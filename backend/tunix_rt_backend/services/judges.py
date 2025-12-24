"""Judge implementations for Tunix run evaluation."""

import hashlib
import json
import logging
from typing import Literal, Protocol

from pydantic import BaseModel

from tunix_rt_backend.db.models import TunixRun
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

    def __init__(self, redi_client: RediClientProtocol) -> None:
        self.redi_client = redi_client

    def get_judge(self, name: str | None = None) -> Judge:
        """Get judge by name. Defaults to MockJudge."""
        if name == "gemma-judge":
            return GemmaJudge(self.redi_client)
        # Default to MockJudge
        return MockJudge()
