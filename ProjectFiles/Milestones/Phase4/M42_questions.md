# M42 Clarifying Questions

## Project Understanding Summary

Based on my analysis, M42 is the **final submission milestone** with these objectives (from M41 audit):
1. Record demo video (following `docs/DEMO.md` script)
2. Final README polish
3. Lock dependency versions
4. Create submission package
5. Final evidence audit

---

## Clarifying Questions

### 1. Video Recording
- **Q1.1:** Is the video recording something you'll do manually, or do you need automation help (e.g., script timing, evidence screenshots)?
- **Q1.2:** Do you already have a video recording tool set up (OBS, Loom, Camtasia)?
- **Q1.3:** The `docs/video_script_m31.md` and `docs/video_shotlist_m31.md` exist — should these be updated for M42, or is `docs/DEMO.md` the authoritative guide?

### 2. Submission Package
- **Q2.1:** What is the target submission format? (ZIP archive? Kaggle notebook commit? Both?)
- **Q2.2:** The `backend/tools/package_submission.py` exists — should I verify/update it for M42?
- **Q2.3:** Is there a deadline I should be aware of for the submission?

### 3. Dependency Locking
- **Q3.1:** Backend uses `uv.lock` already. Should I also pin versions in `pyproject.toml` explicitly?
- **Q3.2:** Frontend uses `package-lock.json`. Any additional locking needed?
- **Q3.3:** The `.venv-gpu` environment uses PyTorch nightly — should this be documented as a separate requirements file?

### 4. README Polish
- **Q4.1:** What sections of `README.md` need updating? (Problem statement, getting started, demo instructions?)
- **Q4.2:** Should the README reference the demo video URL once recorded?

### 5. Evidence Audit
- **Q5.1:** Which `submission_runs/` folders are required for final submission? (m40_v1, m41_v1, m42_v1?)
- **Q5.2:** Should I create a master evidence index/manifest?

### 6. Scope Boundaries
- **Q6.1:** Are there any code changes expected in M42, or is this purely documentation/packaging?
- **Q6.2:** Should I run a final test suite pass and capture the output?

---

## Regarding Cursor Serialization Issues

You asked about managing Cursor backend serialization disruptions. Here are my recommendations:

### Current Approach (What We Have)
- `M42_toolcalls.md` logs each action with timestamp, purpose, files, and next step
- This provides a recovery checkpoint if the session drops

### Additional Recommendations

1. **PowerShell Log Capture (Yes, add this)**
   Add to `.cursorrules`:
   ```
   # PowerShell Session Recovery
   Before running terminal commands, note the expected output in toolcalls.md.
   After completion, summarize the result. This helps recover context if the session serializes mid-command.
   ```

2. **Checkpoint Comments**
   I can add "CHECKPOINT:" markers in toolcalls.md after completing each major step, making it easier to scan for recovery.

3. **Smaller Atomic Steps**
   Break work into smaller commits/saves so less is lost on serialization failure.

4. **Session State File**
   We could maintain a `M42_session_state.json` with:
   - Current step number
   - Last completed action
   - Next planned action
   - Any pending changes

Would you like me to implement any of these recovery strategies?

---

**Waiting for your responses before proceeding with M42_plan.md.**

