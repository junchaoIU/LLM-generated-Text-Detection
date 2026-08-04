"""DRGD: Disentangled Representation for Generalizable AI-Text Detection.

Paper: Pu et al., "Breaking the Generator Barrier: Disentangled Representation
for Generalizable AI-Text Detection", ACL 2026.
Repo: https://github.com/PuXiao06/DRGD

Method: a BERT-based encoder with a *progressive* structured framework —
compact latent encoding (information bottleneck), perturbation-based
regularization, and a content/domain disentanglement (AdaIn + gradient
reversal) — so that the final content classifier relies on AI-detection
semantics shared across generators rather than generator-specific
"fingerprints".  AI text is class ``0``, human text is class ``1``.

At inference time the model follows the official ``test_struct`` path:
BERT -> ``[CLS]`` -> content IB module (mean latent) -> ``content_pro`` ->
content classifier ``CE`` -> 2-class logits.  The detector returns the
softmax probability of the AI class (index ``0``), so higher = more AI.
"""
from typing import Optional, Sequence

import numpy as np

from ...core.base import BaseDetector
from ...core.registry import register_detector


# ---------------------------------------------------------------------------
# IB_Module (ported from module/IB.py of the official repo)
# ---------------------------------------------------------------------------
class _IBModule:
    """Information-bottleneck latent module (faithful port of ``IB_Module``)."""

    def __init__(self, input_dim, hidden_dim, latent_dim, kl_beta, sample_size):
        import torch.nn as nn

        self.nn = nn
        intermediate_dim = (hidden_dim + input_dim) // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.ReLU(),
        )
        self.emb2mu = nn.Linear(hidden_dim, latent_dim)
        self.emb2std = nn.Linear(hidden_dim, latent_dim)
        self.mu_p = nn.Parameter(torch.randn(latent_dim))
        self.std_p = nn.Parameter(torch.randn(latent_dim))
        self.kl_beta = kl_beta
        self.sample_size = sample_size
        self.latent_dim = latent_dim

    def estimate(self, emb):
        import torch.nn.functional as F

        mu = self.emb2mu(emb)
        std = F.softplus(self.emb2std(emb))
        return mu, std

    def reparameterize(self, mu, std):
        import torch

        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).to(mu.device)
        return mu + std * z

    def forward(self, x, training=True):
        import torch

        x = self.mlp(x)
        mu, std = self.estimate(x)
        if training:
            z = self.reparameterize(mu, std)
        else:
            z = mu.unsqueeze(0)  # test time uses the mean
        return z, mu, None


# ---------------------------------------------------------------------------
# BertForAIHumanClassification (inference path of module/model.py)
# ---------------------------------------------------------------------------
def _build_model(pretrained_model: str, device):
    """Build the DRGD content classifier and load it onto ``device``."""
    import torch
    import torch.nn as nn
    from transformers import BertModel

    bert = BertModel.from_pretrained(pretrained_model)
    ib_content = _IBModule(
        input_dim=768, hidden_dim=512, latent_dim=32, kl_beta=1e-5, sample_size=5
    )
    content_pro = nn.Linear(32, 32)
    ce = nn.Sequential()
    ce.add_module("cet_fc1", nn.Linear(32, 16))
    ce.add_module("cet_dropout", nn.Dropout(p=0.5))
    ce.add_module("cet_fc2", nn.Linear(16, 2))

    model = nn.Module()
    model.bert = bert
    model.ib_content = ib_content
    model.content_pro = content_pro
    model.ce = ce
    model.to(device)
    model.eval()
    return model


@register_detector("drgd")
class DRGDDetector(BaseDetector):
    """DRGD disentangled-representation detector (Pu et al., ACL 2026).

    Loads a trained DRGD checkpoint (BERT encoder + information-bottleneck
    content module + ``content_pro`` + content classifier ``CE``) and scores
    each text by the softmax probability of the AI class (index ``0``).

    Args:
        model_path: Path to the trained ``*.pth`` state dict (required).
        pretrained_model: BERT backbone used to build the encoder (default
            ``bert-base-uncased``; the official repo initialized from random
            weights).
        max_length: Tokenizer truncation length.
        batch_size: Batch size for inference.
        ai_class_index: Index of the AI output class (official uses ``0``).
        threshold: Decision threshold on the score.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        pretrained_model: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 16,
        ai_class_index: int = 0,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.model_path = model_path
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.ai_class_index = ai_class_index
        if model_path is None:
            raise RuntimeError(
                "DRGDDetector requires a trained checkpoint. Pass "
                "model_path=<path> (a state dict from the official DRGD repo)."
            )

    def _get_model(self):
        if not hasattr(self, "_model"):
            import torch
            from transformers import AutoTokenizer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
            model = _build_model(self.pretrained_model, self.device)
            state = torch.load(self.model_path, map_location=self.device)
            # Strip the "module." prefix / rename keys to match our layout.
            new_state = {}
            for k, v in state.items():
                k = k.replace("module.", "")
                if k.startswith("bert."):
                    new_state[k] = v
                elif k.startswith("ib_content."):
                    new_state["ib_content." + k[len("ib_content."):]] = v
                elif k.startswith("content_pro."):
                    new_state["content_pro." + k[len("content_pro."):]] = v
                elif k.startswith("CE."):
                    new_state["ce." + k[len("CE."):]] = v
            model.load_state_dict(new_state, strict=False)
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def score(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model = self._get_model()
        tokenizer = self._tokenizer
        texts = list(texts)
        scores = []
        for text in texts:
            tokenized = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            # Faithful test_struct path: BERT -> [CLS] -> content IB -> pro -> CE.
            with torch.inference_mode():
                outputs = model.bert(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                )
                cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
                content, _, _ = model.ib_content(cls_output, training=False)
                content = content.squeeze(0)  # (B, 32)
                content = model.content_pro(content)
                contentOut = model.ce(content)
            probs = torch.nn.functional.softmax(contentOut, -1)
            scores.append(probs[0, self.ai_class_index].item())
        return np.asarray(scores, dtype=float)