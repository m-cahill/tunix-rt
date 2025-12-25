Here are the decisions for your **M28 clarifying questions**, pinned down so you can proceed without thrash.

---

## 1) UNGAR integration: time-box vs fix fully?

**Time-box it.**
Do a focused fix attempt first, but don’t let it block the milestone.

**Recommended policy**

* **Time-box:** ~2–3 hours (or “half a day max” if you want one deeper push).
* If not resolved in the time-box:

  * switch to **`xfail` with a tracking issue**
  * include a short “what I tried / where it broke” note in the issue
  * keep CI stable and move on

This keeps M28 competition-facing (tuning/compare/leaderboard) and prevents UNGAR from becoming a derail.

---

## 2) Tuning sweep: does M19 infrastructure already support LR/batch size/etc.?

**It should already support it if your M19 schema is generic.** In Ray Tune, search spaces are just a param dict mapping names (e.g., `lr`, `batch_size`) to distributions like `tune.loguniform(...)` and `tune.choice(...)`. ([Ray][1])

**What to do**

* Try a sweep with a *simple* search space first:

  * `lr`: loguniform
  * `batch_size`: choice
  * `weight_decay`: uniform
* If your schema currently only allows a restricted subset (e.g., only `uniform`), then extend the validator to accept a small set of Ray primitives (`choice`, `uniform`, `loguniform`, `randint`)—keep it generic and minimal. ([Ray][1])

**Decision**

* **Assume generic support; extend only if validation blocks you.**

---

## 3) Eval score definition for leaderboard

**Yes — simple mean of `answer_correctness` is sufficient for M28.**

**Definition (recommended)**

* `eval_score = mean(answer_correctness)` over all eval items
* Treat missing/invalid values as 0 (or exclude them, but be consistent—pick one)

Keep it stable, explainable, and usable as a single scalar for sorting and tuning.

**Decision**

* **Use `mean(answer_correctness)` as v1.**

---

## 4) Run comparison UI: new page vs modal?

**Hybrid, minimal-routing approach is best:**

* Primary UX: a **compare panel/modal** launched from the existing run list/history (fast to build, minimal disruption).
* Also add a **shareable deep-link** via query params:

  * `/compare?runA=<id>&runB=<id>`
  * This route can render the same component as the modal/panel.

This gives you both:

* “No new app architecture” simplicity
* shareable comparisons (useful for team/judging/demo)

**Decision**

* **Modal/panel + optional `/compare` deep-link route (thin wrapper).**

---

If you follow the above, M28 stays tightly scoped: **tuning runs, run comparison, leaderboard score**, and **UNGAR doesn’t derail the critical path**.

[1]: https://docs.ray.io/en/latest/tune/api/search_space.html?utm_source=chatgpt.com "Tune Search Space API - Ray Docs"
