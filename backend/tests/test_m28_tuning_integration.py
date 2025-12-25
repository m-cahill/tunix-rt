from tunix_rt_backend.schemas.tunix import TunixManifestRequest, TunixRunRequest


def test_tunix_run_request_weight_decay():
    """Verify weight_decay is supported in TunixRunRequest."""
    req = TunixRunRequest(
        dataset_key="test-v1", model_id="test-model", output_dir="./out", weight_decay=0.05
    )
    assert req.weight_decay == 0.05


def test_tunix_run_request_default_weight_decay():
    """Verify default weight_decay."""
    req = TunixRunRequest(dataset_key="test-v1", model_id="test-model", output_dir="./out")
    assert req.weight_decay == 0.01


def test_manifest_generation_weight_decay():
    """Verify weight_decay is included in manifest."""
    from tunix_rt_backend.integrations.tunix.manifest import build_sft_manifest

    req = TunixManifestRequest(
        dataset_key="test-v1", model_id="test-model", output_dir="./out", weight_decay=0.001
    )

    yaml_out = build_sft_manifest(req, "./data.jsonl")
    assert "weight_decay: 0.001" in yaml_out
