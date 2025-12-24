Here are direct answers / recommended decisions for the M20 clarifying questions. 

---

## 1) TunixRun output structure

You **don’t need the exact structure** to proceed safely. For M20, implement promotion as:

* Treat `TunixRun.output_dir` (or equivalent) as an **opaque directory**.
* Validate/promote based on **file pattern discovery** + (optionally) a future “manifest”.

**Validation strategy (robust, format-agnostic):**

* Consider the run promotable if it contains **either**:

  * **LoRA/adapter-style** artifacts:

    * `adapter_config.json` AND one of `adapter_model.bin` / `adapter_model.safetensors`
  * **Full model-style** artifacts:

    * `config.json` AND one of `pytorch_model.bin` / `model.safetensors` / `*.safetensors`
* Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, etc.) should be **optional** for M20, but if present they are copied into the promoted bundle.

**Upgrade path (M21-ish, not required now):**

* Add a `run_manifest.json` written by the run exporter listing *exactly* what was produced and the “type” (adapter/full/checkpoint/etc.). Promotion can prefer the manifest when present.

---

## 2) Required files for promotion

For M20, keep this minimal so you don’t block legitimate runs:

**Hard requirements (choose one track):**

* **Adapter track (preferred if Tunix is doing LoRA / PEFT):**

  * `adapter_config.json`
  * adapter weights (`adapter_model.bin` or `adapter_model.safetensors`)
* **Full model track:**

  * `config.json`
  * model weights (`pytorch_model.bin` or `*.safetensors`)

**Soft requirements (warn, don’t fail):**

* Tokenizer files
* `generation_config.json`
* `README.md` / metadata

**Guardrail:** if none of the hard-requirement patterns match, fail promotion with a message that prints the *found files* (top N) to make debugging immediate.

---

## 3) Storage configuration setting

**Yes. Add it.** This is the clean enterprise-grade move.

* Add `MODEL_REGISTRY_PATH` to backend settings (default: `backend/artifacts/model_registry`).
* Ensure:

  * It is created on boot or on first use.
  * Tests can override it to a temp directory.
  * You never write into the run output dir (promotion should **copy** into registry).

---

## 4) Versioning scheme

**Yes, auto `v1`, `v2`, … is acceptable** and is the simplest stable default.

Recommended rules:

* API accepts optional `version_label`.
* If not provided:

  * compute next version for that artifact by querying existing versions
  * assign `v{n}` (starting at `v1`)
* Validate user-provided labels with something like:

  * `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`
* Enforce uniqueness on `(artifact_id, version_label)`.

Extra guardrail (nice-to-have):

* Also store `version_num` (int) internally for easy sorting, but keep `version_label` as the external identifier.

---

If you want the lowest-risk M20 implementation: **build promotion to accept both adapter and full-model patterns**, add `MODEL_REGISTRY_PATH`, and do auto `vN` versioning by default.
