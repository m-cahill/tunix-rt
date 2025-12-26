# Tunix RT - Video Shot List (M31)

**Purpose:** Recording guide with timestamps and visual instructions  
**Total Runtime:** ~2:30

---

## Shot-by-Shot Breakdown

### Shot 1: Title/Opening (0:00 - 0:15)

| Attribute | Value |
|-----------|-------|
| **Duration** | 15 seconds |
| **Visual** | Project README.md or custom title slide |
| **Show** | "Tunix RT - Trace-First Reasoning Training" |
| **Action** | Static or slow zoom |
| **Audio** | Introduction voiceover |

**Script excerpt:**
> "Hi, I'm Michael, and this is Tunix RT..."

---

### Shot 2: Trace Example (0:15 - 0:45)

| Attribute | Value |
|-----------|-------|
| **Duration** | 30 seconds |
| **Visual** | Code editor or terminal showing a trace |
| **Show** | `backend/datasets/golden-v2/dataset.jsonl` |
| **Action** | Highlight prompt → steps → answer |
| **Audio** | Explain trace-first concept |

**What to show:**
```json
{
  "prompt": "What is 15 + 27?",
  "steps": [
    {"content": "First, I add the ones: 5 + 7 = 12..."}
  ],
  "answer": "42"
}
```

**Script excerpt:**
> "Each trace has a prompt, one or more reasoning steps, and a final answer..."

---

### Shot 3: Reproducibility Assets (0:45 - 1:15)

| Attribute | Value |
|-----------|-------|
| **Duration** | 30 seconds |
| **Visual** | Split: manifest.json + submission_freeze.md |
| **Show** | Key fields: seed, commit SHA, dataset version |
| **Action** | Quick scroll through each file |
| **Audio** | Explain reproducibility guarantees |

**Files to open:**
1. `backend/datasets/golden-v2/manifest.json` - Show `trace_count`, `seed`
2. `docs/submission_freeze.md` - Show commit SHA placeholder

**Script excerpt:**
> "We use fixed seeds, versioned datasets with manifests..."

---

### Shot 4: Kaggle Notebook - Setup (1:15 - 1:30)

| Attribute | Value |
|-----------|-------|
| **Duration** | 15 seconds |
| **Visual** | Jupyter notebook |
| **Show** | `notebooks/kaggle_submission.ipynb` open |
| **Action** | Scroll to show structure (7 sections) |
| **Audio** | Introduce notebook workflow |

**Script excerpt:**
> "Let me show you the Kaggle notebook in action..."

---

### Shot 5: Smoke Run (1:30 - 1:45)

| Attribute | Value |
|-----------|-------|
| **Duration** | 15 seconds |
| **Visual** | Notebook cell 4a executing |
| **Show** | Smoke run command and output |
| **Action** | Execute cell, wait for success |
| **Audio** | Explain smoke run purpose |

**Expected output:**
```
🔥 Starting Smoke Run (2 steps)...
...
✅ Smoke run completed successfully!
   Pipeline validated. Ready for full training.
```

**Script excerpt:**
> "First, a smoke run validates the pipeline works..."

---

### Shot 6: Full Run Configuration (1:45 - 2:00)

| Attribute | Value |
|-----------|-------|
| **Duration** | 15 seconds |
| **Visual** | Configuration cell (cell 4) |
| **Show** | MODEL_NAME, DATASET, MAX_STEPS variables |
| **Action** | Highlight key settings |
| **Audio** | Explain training parameters |

**Script excerpt:**
> "For the full run, we train on the golden-v2 dataset with 100 traces..."

**Note:** Don't actually run full training in video (too slow). Show config only.

---

### Shot 7: Submission Summary (2:00 - 2:15)

| Attribute | Value |
|-----------|-------|
| **Duration** | 15 seconds |
| **Visual** | Pre-recorded or mocked summary output |
| **Show** | Final submission summary cell output |
| **Action** | Scroll through summary |
| **Audio** | Highlight key metrics |

**Expected output:**
```
============================================================
         SUBMISSION SUMMARY
============================================================

📦 Model ID: google/gemma-3-1b-it
📁 Dataset:  golden-v2
🔢 Steps:    100
🎲 Seed:     42

📊 Training Metrics (last 5 steps):
   Step 96: loss=0.0832
   Step 97: loss=0.0814
   ...

🎯 Eval Score: [X.XX]

✅ Submission package ready!
```

---

### Shot 8: Closing (2:15 - 2:30)

| Attribute | Value |
|-----------|-------|
| **Duration** | 15 seconds |
| **Visual** | Repository or results summary |
| **Show** | Final score, repo link |
| **Action** | Fade to repository URL |
| **Audio** | Wrap up and thank viewer |

**Script excerpt:**
> "All code is available in the repository. Thanks for watching!"

---

## Recording Checklist

### Before Recording

- [ ] Close unnecessary applications
- [ ] Set screen resolution to 1920x1080
- [ ] Use dark theme in code editor (better visibility)
- [ ] Clear terminal history
- [ ] Pre-run smoke test to ensure it works
- [ ] Prepare a pre-recorded full training output (optional)
- [ ] Test microphone levels

### During Recording

- [ ] Record each shot separately for easier editing
- [ ] Leave 2-3 second pauses between sections
- [ ] Keep mouse movements slow and deliberate
- [ ] Zoom to ~125% on code for readability
- [ ] Don't rush — aim for clarity over speed

### After Recording

- [ ] Trim silence and mistakes
- [ ] Add transitions between shots (simple cuts or fades)
- [ ] Verify final duration is under 3:00
- [ ] Check audio levels are consistent
- [ ] Export at 1080p, 30fps minimum

---

## Technical Setup

| Setting | Recommended |
|---------|-------------|
| **Resolution** | 1920x1080 (1080p) |
| **Frame Rate** | 30 fps |
| **Format** | MP4 (H.264) |
| **Audio** | 48kHz, stereo |
| **Recording Tool** | OBS, Loom, or Camtasia |

---

## Asset Preparation

Files to have open/ready:
1. `notebooks/kaggle_submission.ipynb` in Jupyter
2. `backend/datasets/golden-v2/dataset.jsonl` in editor
3. `docs/submission_freeze.md` in editor
4. Terminal with smoke run ready to execute
5. Pre-recorded submission summary (optional)

---

## Timeline Summary

| Time | Shot | Content |
|------|------|---------|
| 0:00 | 1 | Title/Opening |
| 0:15 | 2 | Trace Example |
| 0:45 | 3 | Reproducibility |
| 1:15 | 4 | Notebook Setup |
| 1:30 | 5 | Smoke Run |
| 1:45 | 6 | Full Run Config |
| 2:00 | 7 | Summary Output |
| 2:15 | 8 | Closing |
| 2:30 | — | END |
