#!/usr/bin/env python3
"""M47 Error Injection Script — Controlled Error Injection for Correction Fidelity Testing.

This script injects known, localized errors into reasoning traces to test whether
the model's verification behavior can detect and correct real errors.

Design Decisions (from M47_answers.md):
- Error types: Arithmetic slips + Unit/scale errors
- Injection rate: ~10% of training samples (~34 traces)
- Error location: 80% intermediate-propagating, 20% final-only
- CORRECT block: Explicit correction template

Output:
- data/stage_c_clean.jsonl (M46 self-correct format, no errors)
- data/stage_c_error.jsonl (errors injected, no VERIFY/CORRECT)
- data/stage_c_error_self_correct.jsonl (errors with explicit corrections)
- data/stage_c_holdout.jsonl (10% held out for eval)
- data/stage_c_holdout_error.jsonl (held out with errors for eval)
- error_manifest.json (full error tracking)

Usage:
    cd research/m47_error_correction_fidelity
    python scripts/inject_errors.py

Author: M47 Error Correction Fidelity Milestone
Date: 2026-01-09
"""

import codecs
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except Exception:
        pass


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

# Source: M45 Stage-C data (original, no VERIFY/CORRECT)
M45_STAGE_C = PROJECT_DIR.parent / "m45_curriculum_reasoning" / "data" / "stage_c.jsonl"

# M46 self-correct data (with VERIFY/CORRECT, for clean baseline)
M46_SELF_CORRECT = PROJECT_DIR.parent / "m46_structured_self_correction" / "data" / "stage_c_self_correct.jsonl"

# Random seed for reproducibility
SEED = 42

# Injection parameters
HOLDOUT_RATE = 0.10  # 10% held out for evaluation
ERROR_INJECTION_RATE = 0.10  # 10% of training samples get errors
INTERMEDIATE_ERROR_RATE = 0.80  # 80% intermediate-propagating, 20% final-only


# ============================================================
# Error Injection Functions
# ============================================================

def inject_arithmetic_error(trace: dict, error_type: str = "intermediate") -> tuple[dict, dict]:
    """Inject an arithmetic error into a trace.
    
    Args:
        trace: Original trace dict
        error_type: "intermediate" (propagates) or "final" (answer only)
        
    Returns:
        (modified_trace, error_info) tuple
    """
    modified = json.loads(json.dumps(trace))  # Deep copy
    steps = modified.get("steps", [])
    
    if not steps:
        return None, None
    
    error_info = {
        "error_class": "arithmetic",
        "error_type": error_type,
    }
    
    if error_type == "intermediate":
        # Find a calculation step to inject error into
        calc_steps = []
        for i, step in enumerate(steps):
            content = step.get("content", "")
            step_type = step.get("type", "")
            # Look for calculation content with = sign
            if "=" in content and any(c.isdigit() for c in content):
                # Try to extract and modify a number
                numbers = re.findall(r'=\s*(-?\d+\.?\d*)', content)
                if numbers:
                    calc_steps.append((i, step, numbers[-1]))  # Last number after =
        
        if calc_steps:
            # Pick a step to modify (prefer earlier steps for propagation)
            step_idx, step, original_value = calc_steps[0]
            
            try:
                orig_num = float(original_value)
                # Inject error: off by 10 (clear, mechanical error)
                if orig_num >= 10:
                    error_num = orig_num + 10
                else:
                    error_num = orig_num + 1
                
                # Format as int if original was int
                if '.' not in original_value:
                    error_value = str(int(error_num))
                    orig_value_str = str(int(orig_num))
                else:
                    error_value = str(error_num)
                    orig_value_str = original_value
                
                # Modify the step content
                old_content = step["content"]
                new_content = old_content.replace(f"= {original_value}", f"= {error_value}")
                if new_content == old_content:
                    new_content = old_content.replace(f"={original_value}", f"={error_value}")
                
                steps[step_idx]["content"] = new_content
                
                # Also modify the final answer to show propagation
                old_answer = modified.get("final_answer", "")
                # Try to modify final answer based on error magnitude
                try:
                    if old_answer.startswith("$"):
                        ans_num = float(old_answer.replace("$", "").replace(",", ""))
                        new_ans = ans_num + (error_num - orig_num)
                        modified["final_answer"] = f"${new_ans:.2f}" if '.' in old_answer else f"${int(new_ans)}"
                    elif old_answer.endswith("km") or old_answer.endswith(" km"):
                        ans_num = float(old_answer.replace(" km", "").replace("km", ""))
                        new_ans = ans_num + (error_num - orig_num)
                        modified["final_answer"] = f"{int(new_ans)} km"
                    else:
                        # Try generic number extraction
                        ans_match = re.search(r'(-?\d+\.?\d*)', old_answer)
                        if ans_match:
                            ans_num = float(ans_match.group(1))
                            new_ans = ans_num + (error_num - orig_num)
                            if '.' not in ans_match.group(1):
                                modified["final_answer"] = old_answer.replace(ans_match.group(1), str(int(new_ans)))
                            else:
                                modified["final_answer"] = old_answer.replace(ans_match.group(1), str(new_ans))
                except:
                    pass  # Keep original answer if parsing fails
                
                error_info.update({
                    "injected_step_idx": step_idx,
                    "original_value": orig_value_str,
                    "injected_value": error_value,
                    "original_final": trace.get("final_answer", ""),
                    "injected_final": modified.get("final_answer", ""),
                    "correction_text": f"Step {step_idx} is wrong: should be {orig_value_str} (not {error_value}). Correct final: {trace.get('final_answer', '')}"
                })
                
                return modified, error_info
                
            except (ValueError, TypeError):
                pass
    
    else:  # final-only error
        # Modify only the final answer
        old_answer = modified.get("final_answer", "")
        try:
            if old_answer.startswith("$"):
                ans_num = float(old_answer.replace("$", "").replace(",", ""))
                new_ans = ans_num + 10
                modified["final_answer"] = f"${new_ans:.2f}" if '.' in old_answer else f"${int(new_ans)}"
            elif old_answer.endswith("km") or old_answer.endswith(" km"):
                ans_num = float(old_answer.replace(" km", "").replace("km", ""))
                new_ans = ans_num + 10
                modified["final_answer"] = f"{int(new_ans)} km"
            else:
                ans_match = re.search(r'(-?\d+\.?\d*)', old_answer)
                if ans_match:
                    ans_num = float(ans_match.group(1))
                    new_ans = ans_num + 10
                    if '.' not in ans_match.group(1):
                        modified["final_answer"] = old_answer.replace(ans_match.group(1), str(int(new_ans)))
                    else:
                        modified["final_answer"] = old_answer.replace(ans_match.group(1), str(new_ans))
            
            error_info.update({
                "injected_step_idx": -1,  # No step modified
                "original_value": trace.get("final_answer", ""),
                "injected_value": modified.get("final_answer", ""),
                "original_final": trace.get("final_answer", ""),
                "injected_final": modified.get("final_answer", ""),
                "correction_text": f"Final answer is wrong: should be {trace.get('final_answer', '')} (not {modified.get('final_answer', '')})"
            })
            
            return modified, error_info
            
        except (ValueError, TypeError):
            pass
    
    return None, None


def inject_unit_error(trace: dict) -> tuple[dict, dict]:
    """Inject a unit/scale error into a trace.
    
    Only for traces that involve unit conversions (template_0: distance/speed/time).
    Scale by wrong factor (×10 instead of correct factor).
    
    Args:
        trace: Original trace dict
        
    Returns:
        (modified_trace, error_info) tuple or (None, None) if not applicable
    """
    template_id = trace.get("meta", {}).get("template_id")
    
    # Only inject unit errors for distance/speed/time problems (template_0)
    if template_id != 0:
        return None, None
    
    modified = json.loads(json.dumps(trace))  # Deep copy
    
    # Modify the final answer by wrong scale (×10)
    old_answer = modified.get("final_answer", "")
    try:
        if "km" in old_answer:
            ans_num = float(old_answer.replace(" km", "").replace("km", ""))
            new_ans = ans_num * 10  # Wrong scale factor
            modified["final_answer"] = f"{int(new_ans)} km"
            
            error_info = {
                "error_class": "unit",
                "error_type": "scale",
                "injected_step_idx": -1,
                "original_value": old_answer,
                "injected_value": modified["final_answer"],
                "original_final": old_answer,
                "injected_final": modified["final_answer"],
                "correction_text": f"Scale error: result should be {old_answer} (not {modified['final_answer']})"
            }
            
            return modified, error_info
    except (ValueError, TypeError):
        pass
    
    return None, None


def add_verify_correct_blocks(trace: dict, error_info: dict = None) -> dict:
    """Add VERIFY and CORRECT blocks to a trace.
    
    Args:
        trace: Trace to augment
        error_info: If provided, use explicit correction; otherwise "No correction needed"
        
    Returns:
        Augmented trace with VERIFY/CORRECT
    """
    modified = json.loads(json.dumps(trace))
    steps = modified.get("steps", [])
    next_idx = len(steps)
    
    # Get template ID for verification template
    template_id = trace.get("meta", {}).get("template_id")
    
    # Verification template based on problem type
    verify_templates = {
        0: "Check by inverse: divide distance by time to verify speed",
        2: "Check by inverse: multiply result by count to verify total",
        3: "Check by inverse: subtract one addend from sum",
        4: "Check by inverse: add result to subtrahend",
        5: "Check by inverse: divide product by one factor",
    }
    verify_text = verify_templates.get(template_id, "Check: verify answer matches expected format")
    
    # Add VERIFY step
    steps.append({
        "i": next_idx,
        "type": "verify",
        "content": f"VERIFY: {verify_text}"
    })
    
    # Add CORRECT step
    if error_info:
        correct_text = error_info.get("correction_text", "Error detected. Recalculating.")
    else:
        correct_text = "No correction needed"
    
    steps.append({
        "i": next_idx + 1,
        "type": "correct",
        "content": f"CORRECT: {correct_text}"
    })
    
    modified["steps"] = steps
    return modified


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("M47 Error Injection — Controlled Error Injection")
    print("=" * 60)
    
    random.seed(SEED)
    
    # Verify sources exist
    if not M45_STAGE_C.exists():
        print(f"[ERROR] M45 stage_c not found: {M45_STAGE_C}")
        return 1
    
    if not M46_SELF_CORRECT.exists():
        print(f"[ERROR] M46 self_correct not found: {M46_SELF_CORRECT}")
        return 1
    
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load M45 Stage-C (original, no VERIFY/CORRECT)
    print(f"\n[LOAD] Loading M45 Stage-C: {M45_STAGE_C}")
    traces = []
    with open(M45_STAGE_C, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    print(f"       Loaded {len(traces)} traces")
    
    # Load M46 self-correct (with VERIFY/CORRECT) for clean baseline
    print(f"[LOAD] Loading M46 Self-Correct: {M46_SELF_CORRECT}")
    m46_traces = []
    with open(M46_SELF_CORRECT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m46_traces.append(json.loads(line))
    print(f"       Loaded {len(m46_traces)} traces")
    
    # Shuffle and split: 10% holdout
    indices = list(range(len(traces)))
    random.shuffle(indices)
    
    holdout_count = int(len(traces) * HOLDOUT_RATE)
    holdout_indices = set(indices[:holdout_count])
    train_indices = indices[holdout_count:]
    
    print(f"\n[SPLIT] Holdout: {holdout_count}, Training: {len(train_indices)}")
    
    # Select 10% of training for error injection
    error_count = int(len(train_indices) * ERROR_INJECTION_RATE)
    error_indices = set(random.sample(train_indices, error_count))
    
    # Split error indices: 80% intermediate, 20% final-only
    error_indices_list = list(error_indices)
    random.shuffle(error_indices_list)
    intermediate_count = int(len(error_indices_list) * INTERMEDIATE_ERROR_RATE)
    intermediate_indices = set(error_indices_list[:intermediate_count])
    final_only_indices = set(error_indices_list[intermediate_count:])
    
    print(f"[INJECT] Error injection: {error_count} total")
    print(f"         Intermediate-propagating: {len(intermediate_indices)}")
    print(f"         Final-only: {len(final_only_indices)}")
    
    # Process traces
    clean_traces = []
    error_traces = []
    error_self_correct_traces = []
    holdout_traces = []
    holdout_error_traces = []
    error_manifest = []
    
    stats = {
        "total": len(traces),
        "holdout": 0,
        "training": 0,
        "clean": 0,
        "error_injected": 0,
        "error_arithmetic": 0,
        "error_unit": 0,
        "error_intermediate": 0,
        "error_final_only": 0,
        "injection_failures": 0,
    }
    
    for i, trace in enumerate(traces):
        if i in holdout_indices:
            # Holdout set
            stats["holdout"] += 1
            holdout_traces.append(trace)
            
            # Also create error-injected version for holdout
            modified, error_info = inject_arithmetic_error(trace, "intermediate")
            if modified and error_info:
                holdout_error_traces.append(modified)
            else:
                holdout_error_traces.append(trace)  # Keep original if injection fails
            
            continue
        
        stats["training"] += 1
        
        if i in error_indices:
            # Inject error
            if i in intermediate_indices:
                error_type = "intermediate"
            else:
                error_type = "final"
            
            # Try arithmetic error first
            modified, error_info = inject_arithmetic_error(trace, error_type)
            
            # If arithmetic fails, try unit error (only for template_0)
            if not modified:
                modified, error_info = inject_unit_error(trace)
                if modified:
                    error_info["error_type"] = error_type  # Preserve type intention
            
            if modified and error_info:
                stats["error_injected"] += 1
                if error_info["error_class"] == "arithmetic":
                    stats["error_arithmetic"] += 1
                else:
                    stats["error_unit"] += 1
                
                if error_type == "intermediate":
                    stats["error_intermediate"] += 1
                else:
                    stats["error_final_only"] += 1
                
                # Record in manifest
                manifest_entry = {
                    "sample_id": i,
                    "trace_index": trace.get("meta", {}).get("index"),
                    "template_id": trace.get("meta", {}).get("template_id"),
                    "prompt": trace.get("prompt", "")[:50],
                    **error_info
                }
                error_manifest.append(manifest_entry)
                
                # Error trace without VERIFY/CORRECT
                error_traces.append(modified)
                
                # Error trace with VERIFY/CORRECT (including correction)
                error_with_correct = add_verify_correct_blocks(modified, error_info)
                error_self_correct_traces.append(error_with_correct)
                
                # Also add clean version to clean dataset
                clean_trace = m46_traces[i] if i < len(m46_traces) else add_verify_correct_blocks(trace)
                clean_traces.append(clean_trace)
            else:
                # Injection failed - use as clean
                stats["injection_failures"] += 1
                stats["clean"] += 1
                clean_trace = m46_traces[i] if i < len(m46_traces) else add_verify_correct_blocks(trace)
                clean_traces.append(clean_trace)
                error_traces.append(trace)  # No error
                error_self_correct_traces.append(add_verify_correct_blocks(trace))
        else:
            # Clean sample
            stats["clean"] += 1
            clean_trace = m46_traces[i] if i < len(m46_traces) else add_verify_correct_blocks(trace)
            clean_traces.append(clean_trace)
            error_traces.append(trace)
            error_self_correct_traces.append(add_verify_correct_blocks(trace))
    
    # Write output files
    print(f"\n[WRITE] Writing datasets...")
    
    # Clean (M46 style)
    clean_path = DATA_DIR / "stage_c_clean.jsonl"
    with open(clean_path, "w", encoding="utf-8") as f:
        for t in clean_traces:
            f.write(json.dumps(t) + "\n")
    print(f"         stage_c_clean.jsonl: {len(clean_traces)} traces")
    
    # Error (no VERIFY/CORRECT)
    error_path = DATA_DIR / "stage_c_error.jsonl"
    with open(error_path, "w", encoding="utf-8") as f:
        for t in error_traces:
            f.write(json.dumps(t) + "\n")
    print(f"         stage_c_error.jsonl: {len(error_traces)} traces")
    
    # Error + Self-Correct
    error_sc_path = DATA_DIR / "stage_c_error_self_correct.jsonl"
    with open(error_sc_path, "w", encoding="utf-8") as f:
        for t in error_self_correct_traces:
            f.write(json.dumps(t) + "\n")
    print(f"         stage_c_error_self_correct.jsonl: {len(error_self_correct_traces)} traces")
    
    # Holdout (clean)
    holdout_path = DATA_DIR / "stage_c_holdout.jsonl"
    with open(holdout_path, "w", encoding="utf-8") as f:
        for t in holdout_traces:
            f.write(json.dumps(t) + "\n")
    print(f"         stage_c_holdout.jsonl: {len(holdout_traces)} traces")
    
    # Holdout (with errors)
    holdout_error_path = DATA_DIR / "stage_c_holdout_error.jsonl"
    with open(holdout_error_path, "w", encoding="utf-8") as f:
        for t in holdout_error_traces:
            f.write(json.dumps(t) + "\n")
    print(f"         stage_c_holdout_error.jsonl: {len(holdout_error_traces)} traces")
    
    # Error manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source": str(M45_STAGE_C),
        "stats": stats,
        "injection_parameters": {
            "holdout_rate": HOLDOUT_RATE,
            "error_injection_rate": ERROR_INJECTION_RATE,
            "intermediate_error_rate": INTERMEDIATE_ERROR_RATE,
        },
        "errors": error_manifest,
    }
    
    manifest_path = PROJECT_DIR / "error_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[MANIFEST] Saved: {manifest_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Error Injection Complete!")
    print("=" * 60)
    print(f"  Total traces: {stats['total']}")
    print(f"  Holdout: {stats['holdout']}")
    print(f"  Training: {stats['training']}")
    print(f"  Clean: {stats['clean']}")
    print(f"  Error-injected: {stats['error_injected']}")
    print(f"    - Arithmetic: {stats['error_arithmetic']}")
    print(f"    - Unit: {stats['error_unit']}")
    print(f"    - Intermediate: {stats['error_intermediate']}")
    print(f"    - Final-only: {stats['error_final_only']}")
    print(f"  Injection failures: {stats['injection_failures']}")
    
    return 0


if __name__ == "__main__":
    exit(main())

