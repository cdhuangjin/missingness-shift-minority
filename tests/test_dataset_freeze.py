"""Phase E1/E2 dataset freeze invariants."""

from src.dataset_registry import freeze_hash, list_hash, load_registry


def _reg():
    return load_registry("configs/datasets_phase_e12.yaml")


def test_registry_has_eight_frozen_datasets():
    reg = _reg()
    datasets = reg["datasets"]
    assert len(datasets) == 8
    names = [d["name"] for d in datasets]
    assert len(set(names)) == 8
    assert "frozen_at" in reg


def test_freeze_hash_deterministic_and_sha1():
    h1 = freeze_hash(_reg())
    h2 = freeze_hash(_reg())
    assert h1 == h2
    assert len(h1) == 40  # sha1 hex


def test_list_hash_deterministic_and_order_sensitive():
    r1 = _reg()
    assert list_hash(r1) == list_hash(_reg())
    # Reorder the registry in memory to check order sensitivity.
    reordered = dict(r1)
    reordered["datasets"] = list(reversed(r1["datasets"]))
    assert list_hash(reordered) != list_hash(r1)


def test_each_dataset_has_required_source_keys():
    reg = _reg()
    for d in reg["datasets"]:
        assert "name" in d and "source" in d and "target" in d
        assert d["source"] in ("uci", "uci_zip")
        if d["source"] == "uci":
            assert "uci_id" in d
        else:
            assert "zip_url" in d and "csv_in_zip" in d
