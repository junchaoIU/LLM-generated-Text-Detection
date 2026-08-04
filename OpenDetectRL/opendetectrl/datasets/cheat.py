"""CHEAT: A Large-scale Dataset for Detecting CHatGPT-writtEn AbsTracts
(Yu et al., IEEE TBD 2025).

Official repo: https://github.com/botianzhe/CHEAT

Dataset structure:
- ``ieee-init.jsonl``: human-written IEEE abstracts (label = 0)
- ``ieee-chatgpt-generation.jsonl``: ChatGPT-generated abstracts (label = 1)
- ``ieee-chatgpt-polish.jsonl``: ChatGPT-polished abstracts (label = 1)
- ``ieee-chatgpt-fusion.jsonl``: human/ChatGPT hybrid abstracts (label = 1)

Each JSONL line is a JSON object. The abstract text is stored under the
``abstract`` key (and optionally ``title`` / ``keywords``).

The dataset contains 35,304 synthetic abstracts in three categories:
Generation, Polish, and Fusion.
"""
import json
import os
from typing import List, Optional

from .base import BaseDataset, DatasetSample
from .registry import register_dataset

REPO_URL = "https://raw.githubusercontent.com/botianzhe/CHEAT/main/data"

FILES = {
    "human": "ieee-init.jsonl",
    "generation": "ieee-chatgpt-generation.jsonl",
    "polish": "ieee-chatgpt-polish.jsonl",
    "fusion": "ieee-chatgpt-fusion.jsonl",
}


def _download_file(filename: str, cache_dir: str) -> str:
    """Download a file from the CHEAT repo if not cached locally."""
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return path
    os.makedirs(cache_dir, exist_ok=True)
    import urllib.request

    url = f"{REPO_URL}/{filename}"
    print(f"Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")
    return path


def _read_jsonl(path: str) -> List[dict]:
    """Read a JSONL file and return a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_text(item: dict) -> str:
    """Extract the abstract text from a CHEAT JSONL entry.

    The official format uses the ``abstract`` key.
    """
    # Try common field names
    for key in ("abstract", "text", "content", "body"):
        if key in item and item[key]:
            return item[key]
    # Fallback: join all string values
    return " ".join(str(v) for v in item.values() if isinstance(v, str))


@register_dataset
class CHEATDataset(BaseDataset):
    """CHEAT: ChatGPT-written academic abstract detection dataset.

    Args:
        cache_dir: Directory for caching downloaded files
            (default: ``~/.cache/opendetectrl/cheat``).
        subsets: Which AI subsets to include: ``"generation"``,
            ``"polish"``, ``"fusion"`` (default: all).
        download: Whether to auto-download missing files (default True).

    The dataset pairs human-written IEEE abstracts with ChatGPT-generated,
    polished, and hybrid-fusion versions. Each human abstract has a
    corresponding AI version.
    """

    name = "cheat"
    domain = "academic"
    source_models = ["chatgpt"]
    languages = ["en"]

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        subsets: Optional[List[str]] = None,
        download: bool = True,
        **kwargs,
    ):
        super().__init__(cache_dir=cache_dir, **kwargs)
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/opendetectrl/cheat")
        self.cache_dir = cache_dir
        self.subsets = subsets or ["generation", "polish", "fusion"]
        self.download = download

    def _get_file(self, key: str) -> str:
        filename = FILES[key]
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            if not self.download:
                raise FileNotFoundError(
                    f"File {filename} not found in {self.cache_dir}. "
                    "Set download=True or place the file manually."
                )
            _download_file(filename, self.cache_dir)
        return path

    def load(
        self, split: str = "all", max_samples: Optional[int] = None
    ) -> List[DatasetSample]:
        """Load CHEAT dataset samples.

        Args:
            split: ``"all"`` returns all samples. CHEAT does not have an
                official train/test split; ``"train"`` and ``"test"`` each
                return the full dataset (use ``max_samples`` for subsampling).
            max_samples: If given, randomly subsample to at most this many
                total samples (balanced across human/AI).

        Returns:
            List of :class:`DatasetSample`.
        """
        import random

        # Load human samples
        human_path = self._get_file("human")
        human_data = _read_jsonl(human_path)
        human_samples = [
            DatasetSample(
                text=_extract_text(item),
                label=0,
                metadata={"source": "human", "id": item.get("id", i)},
            )
            for i, item in enumerate(human_data)
        ]

        # Load AI samples from selected subsets
        ai_samples = []
        for subset in self.subsets:
            if subset not in FILES:
                continue
            ai_path = self._get_file(subset)
            ai_data = _read_jsonl(ai_path)
            ai_samples.extend([
                DatasetSample(
                    text=_extract_text(item),
                    label=1,
                    metadata={
                        "source": "chatgpt",
                        "subset": subset,
                        "id": item.get("id", i),
                    },
                )
                for i, item in enumerate(ai_data)
            ])

        all_samples = human_samples + ai_samples

        # Subsample if needed
        if max_samples is not None and len(all_samples) > max_samples:
            random.seed(42)
            # Balance: half human, half AI
            n_per_class = max_samples // 2
            human = random.sample(human_samples, min(n_per_class, len(human_samples)))
            ai = random.sample(ai_samples, min(max_samples - len(human), len(ai_samples)))
            all_samples = human + ai

        return all_samples
