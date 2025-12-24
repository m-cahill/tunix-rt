# M20 Clarifying Questions

1. **TunixRun Output Structure:** Could you please provide the typical file structure of a successful TunixRun output directory? (e.g., does it contain `adapter_model.bin`, `config.json`, `tokenizer.json`, etc.?) Knowing this will help me implement robust validation during promotion.

2. **Required Files:** Are there any specific files that *must* be present for a promotion to succeed? (e.g. strictly require `adapter_model.bin`?)

3. **Storage Configuration:** Should I add a configurable setting (e.g., `MODEL_REGISTRY_PATH`) to `Settings` for the artifact storage location, defaulting to `backend/artifacts/model_registry`?

4. **Versioning:** The plan suggests "integer auto-increment or user-provided". I plan to implement `v1`, `v2`, etc. automatically if no version label is provided. Is this acceptable?
