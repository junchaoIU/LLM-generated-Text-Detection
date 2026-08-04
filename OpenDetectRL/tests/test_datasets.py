"""Tests for dataset loaders."""
import os
import tempfile
from unittest.mock import patch

import pytest

from opendetectrl import list_datasets, load_dataset
from opendetectrl.datasets import CHEATDataset, DatasetSample
from opendetectrl.datasets.cheat import _extract_text, _read_jsonl


class TestRegistry:
    def test_cheat_registered(self):
        names = list_datasets()
        assert "cheat" in names

    def test_load_dataset_returns_instance(self):
        ds = load_dataset("cheat", cache_dir="/tmp/test_cheat", download=False)
        assert isinstance(ds, CHEATDataset)
        assert ds.name == "cheat"
        assert ds.domain == "academic"
        assert "chatgpt" in ds.source_models


class TestCHEATLoader:
    def test_extract_text_abstract_key(self):
        item = {"abstract": "This is a test abstract.", "title": "Test"}
        assert _extract_text(item) == "This is a test abstract."

    def test_extract_text_fallback(self):
        item = {"foo": "hello", "bar": "world"}
        result = _extract_text(item)
        assert "hello" in result and "world" in result

    def test_read_jsonl(self):
        import json

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({"abstract": "text1", "id": 1}) + "\n")
            f.write(json.dumps({"abstract": "text2", "id": 2}) + "\n")
            path = f.name
        try:
            data = _read_jsonl(path)
            assert len(data) == 2
            assert data[0]["abstract"] == "text1"
        finally:
            os.unlink(path)

    def test_load_with_local_files(self):
        """Test loading from local JSONL files (no download needed)."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock JSONL files
            for key, filename in [
                ("human", "ieee-init.jsonl"),
                ("generation", "ieee-chatgpt-generation.jsonl"),
            ]:
                with open(os.path.join(tmpdir, filename), "w", encoding="utf-8") as f:
                    for i in range(5):
                        f.write(json.dumps({"abstract": f"{key}_text_{i}", "id": i}) + "\n")

            ds = CHEATDataset(cache_dir=tmpdir, subsets=["generation"], download=False)
            samples = ds.load(max_samples=6)

            assert len(samples) == 6  # 3 human + 3 AI (balanced subsample)
            human = [s for s in samples if s.label == 0]
            ai = [s for s in samples if s.label == 1]
            assert len(human) == 3
            assert len(ai) == 3
            assert all(s.metadata.get("subset") == "generation" for s in ai)

    def test_load_texts_labels(self):
        """Test the convenience method."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            for key, filename in [
                ("human", "ieee-init.jsonl"),
                ("generation", "ieee-chatgpt-generation.jsonl"),
            ]:
                with open(os.path.join(tmpdir, filename), "w", encoding="utf-8") as f:
                    for i in range(3):
                        f.write(json.dumps({"abstract": f"{key}_{i}"}) + "\n")

            ds = CHEATDataset(cache_dir=tmpdir, subsets=["generation"], download=False)
            texts, labels = ds.load_texts_labels()
            assert len(texts) == 6
            assert len(labels) == 6
            assert set(labels) == {0, 1}
