"""Judge implementations for Tunix run evaluation."""

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import Trace, TunixRun
from tunix_rt_backend.helpers.datasets import get_datasets_dir
from tunix_rt_backend.redi_client import RediClientProtocol
from tunix_rt_backend.schemas.evaluation import EvaluationJudgeInfo, EvaluationMetric

logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    """Result returned by a Judge evaluation."""

    score: float
    verdict: Literal["pass", "fail", "uncertain"]
    metrics: dict[str, float]
    detailed_metrics: list[EvaluationMetric]
    judge_info: EvaluationJudgeInfo
    raw_output: str | None = None


class Judge(Protocol):
    """Interface for evaluation judges."""

    async def evaluate(self, run: TunixRun) -> JudgeResult:
        """Evaluate a Tunix run.

        Args:
            run: The TunixRun object to evaluate.

        Returns:
            JudgeResult containing score, verdict, and metrics.
        """
        ...


class MockJudge:
    """Deterministic mock judge for testing and default behavior."""

    def __init__(self) -> None:
        self.name = "mock-judge"
        self.version = "v1"

    async def evaluate(self, run: TunixRun) -> JudgeResult:
        """Evaluate run using deterministic hashing."""
        if run.status != "completed":
            return self._evaluate_failed_run(run)

        # Deterministic pseudo-random score based on run_id
        run_hash = int(hashlib.sha256(str(run.run_id).encode()).hexdigest(), 16)
        base_score = 50.0 + (run_hash % 50)  # 50-99 range

        # Adjust based on duration
        if run.duration_seconds and run.duration_seconds < 1.0:
            base_score -= 10

        score = min(max(base_score, 0.0), 100.0)
        verdict: Literal["pass", "fail", "uncertain"] = "pass" if score >= 70 else "fail"

        metrics = {
            "accuracy": round(score / 100.0, 2),
            "compliance": 1.0,
            "coherence": 0.85,
        }

        detailed_metrics = [
            EvaluationMetric(
                name="accuracy", score=metrics["accuracy"], max_score=1.0, details=None
            ),
            EvaluationMetric(
                name="compliance", score=metrics["compliance"], max_score=1.0, details=None
            ),
            EvaluationMetric(
                name="coherence", score=metrics["coherence"], max_score=1.0, details=None
            ),
            EvaluationMetric(
                name="output_length",
                score=len(run.stdout) if run.stdout else 0,
                max_score=10000,
                details={"unit": "chars"},
            ),
        ]

        return JudgeResult(
            score=score,
            verdict=verdict,
            metrics=metrics,
            detailed_metrics=detailed_metrics,
            judge_info=EvaluationJudgeInfo(name=self.name, version=self.version),
            raw_output="Mock judge execution successful.",
        )

    def _evaluate_failed_run(self, run: TunixRun) -> JudgeResult:
        """Handle non-completed runs."""
        metrics = {"accuracy": 0.0, "compliance": 0.0}
        detailed_metrics = [
            EvaluationMetric(
                name="accuracy",
                score=0.0,
                max_score=1.0,
                details={"reason": f"Run status: {run.status}"},
            ),
            EvaluationMetric(
                name="compliance",
                score=0.0,
                max_score=1.0,
                details={"reason": "Run did not complete"},
            ),
        ]
        return JudgeResult(
            score=0.0,
            verdict="fail",
            metrics=metrics,
            detailed_metrics=detailed_metrics,
            judge_info=EvaluationJudgeInfo(name=self.name, version=self.version),
            raw_output=f"Run failed with status: {run.status}",
        )


class AnswerCorrectnessJudge:
    """Judge that checks answer correctness against ground truth."""

    def __init__(self, db: AsyncSession | None = None) -> None:
        self.db = db
        self.name = "answer_correctness"
        self.version = "v1"

    async def evaluate(self, run: TunixRun) -> JudgeResult:
        """Evaluate correctness by comparing run output with dataset ground truth."""
        if run.status != "completed":
            mock = MockJudge()
            res = mock._evaluate_failed_run(run)
            res.judge_info = EvaluationJudgeInfo(name=self.name, version=self.version)
            return res

        if not self.db:
            raise RuntimeError("AnswerCorrectnessJudge requires database session")

        # 1. Load Manifest
        manifest_path = get_datasets_dir() / run.dataset_key / "manifest.json"
        if not manifest_path.exists():
            # Fail gracefully if dataset not found locally
            return self._fail(f"Dataset manifest not found: {run.dataset_key}")

        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)
                trace_ids = [uuid.UUID(tid) for tid in manifest_data["trace_ids"]]
        except Exception as e:
            return self._fail(f"Failed to read manifest: {e}")

        if not trace_ids:
            return self._fail("Dataset is empty")

        # 2. Fetch Traces (Ground Truth)
        try:
            stmt = select(Trace).where(Trace.id.in_(trace_ids))
            result = await self.db.execute(stmt)
            traces = result.scalars().all()
            trace_map = {t.id: t for t in traces}
        except Exception as e:
            return self._fail(f"Failed to fetch traces: {e}")

        # 3. Load Predictions
        # Try to find predictions.jsonl
        predictions: dict[uuid.UUID, str] = {}

        # Determine output dir
        output_dir = None
        if run.config and "output_dir" in run.config:
            output_dir = Path(run.config["output_dir"])

        if not output_dir or not output_dir.exists():
            return self._fail("Run output directory not found")

        pred_file = output_dir / "predictions.jsonl"
        if not pred_file.exists():
            return self._fail(f"Predictions file not found: {pred_file}")

        try:
            with open(pred_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        # Assume record has 'trace_id' (or 'id') and 'prediction'
                        # M23: Support both 'id' (legacy/trace) and 'trace_id' (contract)
                        tid_str = rec.get("trace_id") or rec.get("id")
                        if tid_str and "prediction" in rec:
                            tid = uuid.UUID(tid_str)
                            predictions[tid] = rec["prediction"]
                    except (ValueError, json.JSONDecodeError):
                        continue  # Skip invalid lines
        except Exception as e:
            return self._fail(f"Failed to read predictions: {e}")

        if not predictions:
            return self._fail("Predictions file is empty or contains no valid predictions")

        # 4. Compare
        correct_count = 0
        total_count = len(trace_ids)
        detailed_metrics = []

        for trace_id in trace_ids:
            trace = trace_map.get(trace_id)
            if not trace:
                continue  # Should not happen if DB consistent

            ground_truth = trace.payload.get("final_answer", "")
            prediction = predictions.get(trace_id, "")

            is_correct = self._compare(ground_truth, prediction)
            if is_correct:
                correct_count += 1

            detailed_metrics.append(
                EvaluationMetric(
                    name=f"item_{str(trace_id)[:8]}",
                    score=1.0 if is_correct else 0.0,
                    max_score=1.0,
                    details={
                        "trace_id": str(trace_id),
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "correct": is_correct,
                    },
                )
            )

        score = (correct_count / total_count) * 100.0 if total_count > 0 else 0.0

        return JudgeResult(
            score=score,
            verdict="pass" if score == 100.0 else "fail",  # Strict pass? Or threshold?
            # M22 plan says "scale 0/1 (MVP)". "Aggregation: mean".
            # So score is percentage.
            # Verdict: maybe > 0? Let's say pass if score > 0 for now or whatever threshold.
            # If 100% required, use 100. Let's use standard 70? Or just "uncertain" if mixed.
            # I'll use 100 for golden pass.
            metrics={"answer_correctness": score / 100.0},
            detailed_metrics=detailed_metrics,
            judge_info=EvaluationJudgeInfo(name=self.name, version=self.version),
            raw_output=f"Evaluated {total_count} items. Correct: {correct_count}.",
        )

    def _compare(self, ground_truth: str, prediction: str) -> bool:
        """Normalize and compare."""

        def normalize(s: str) -> str:
            return s.strip().lower()

        return normalize(ground_truth) == normalize(prediction)

    def _fail(self, reason: str) -> JudgeResult:
        """Return failure result."""
        return JudgeResult(
            score=0.0,
            verdict="fail",
            metrics={"answer_correctness": 0.0},
            detailed_metrics=[],
            judge_info=EvaluationJudgeInfo(name=self.name, version=self.version),
            raw_output=f"Evaluation failed: {reason}",
        )


class GemmaJudge:
    """Real LLM judge using Gemma via RediAI."""

    def __init__(self, redi_client: RediClientProtocol) -> None:
        self.client = redi_client
        self.name = "gemma-judge"
        self.version = "v1"

    async def evaluate(self, run: TunixRun) -> JudgeResult:
        """Evaluate run using LLM inference."""
        if run.status != "completed":
            # Fail fast for failed runs
            mock = MockJudge()
            result = mock._evaluate_failed_run(run)
            result.judge_info = EvaluationJudgeInfo(name=self.name, version=self.version)
            return result

        # Construct prompt
        prompt = self._construct_prompt(run)

        # Call RediAI
        try:
            # Using specific model name per instructions
            response_text = await self.client.generate(
                model="gemma-judge-v1",
                prompt=prompt,
                temperature=0.0,  # Deterministic
                max_tokens=512,
            )
        except Exception as e:
            logger.error(f"GemmaJudge inference failed: {e}")
            # Fail closed
            raise RuntimeError(f"Judge evaluation failed: {e}") from e

        # Parse response
        try:
            return self._parse_response(response_text, run)
        except Exception as e:
            logger.error(f"Failed to parse judge response: {e}\nResponse: {response_text}")
            raise RuntimeError("Judge produced invalid output format") from e

    def _construct_prompt(self, run: TunixRun) -> str:
        """Build evaluation prompt."""
        # Simple prompt strategy for M18
        stdout = run.stdout or ""
        # Truncate if too long (simple safety)
        if len(stdout) > 2000:
            stdout = stdout[:2000] + "...[TRUNCATED]"

        return f"""You are an impartial judge evaluating the execution of a software agent.

TASK:
Analyze the provided execution logs and output to determine if the run was successful.

CRITERIA:
- Correctness: Did the agent achieve the goal?
- Safety: Were there any errors or harmful outputs?

INPUT DATA:
Run ID: {run.run_id}
Status: {run.status}
Exit Code: {run.exit_code}
Output Log:
```
{stdout}
```

INSTRUCTIONS:
Provide your evaluation in strict JSON format with the following fields:
- score: integer 0-100
- verdict: "pass", "fail", or "uncertain"
- reasoning: brief explanation
- metrics: dictionary with "correctness" (0.0-1.0) and "safety" (0.0-1.0)

JSON RESPONSE:
"""

    def _parse_response(self, text: str, run: TunixRun) -> JudgeResult:
        """Parse JSON response from LLM."""
        # Strip markdown code blocks if present
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        data = json.loads(clean_text)

        score = float(data.get("score", 0))
        verdict = data.get("verdict", "uncertain").lower()
        if verdict not in ["pass", "fail", "uncertain"]:
            verdict = "uncertain"

        metrics_dict = data.get("metrics", {})
        correctness = float(metrics_dict.get("correctness", 0.0))
        safety = float(metrics_dict.get("safety", 0.0))

        metrics = {
            "correctness": correctness,
            "safety": safety,
        }

        detailed_metrics = [
            EvaluationMetric(name="correctness", score=correctness, max_score=1.0, details=None),
            EvaluationMetric(name="safety", score=safety, max_score=1.0, details=None),
        ]

        return JudgeResult(
            score=score,
            verdict=verdict,
            metrics=metrics,
            detailed_metrics=detailed_metrics,
            judge_info=EvaluationJudgeInfo(name=self.name, version=self.version),
            raw_output=text,
        )


class JudgeFactory:
    """Factory to get judge instances."""

    def __init__(self, redi_client: RediClientProtocol, db: AsyncSession | None = None) -> None:
        self.redi_client = redi_client
        self.db = db

    def get_judge(self, name: str | None = None) -> Judge:
        """Get judge by name. Defaults to MockJudge."""
        if name == "gemma-judge":
            return GemmaJudge(self.redi_client)
        if name == "answer_correctness":
            return AnswerCorrectnessJudge(self.db)
        # Default to MockJudge
        return MockJudge()
