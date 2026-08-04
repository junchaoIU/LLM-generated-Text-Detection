"""Evaluation utilities and metrics."""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def split_texts_labels(
    texts: Sequence[str],
    labels: Sequence[int],
    test_size: float = 0.5,
    random_state: int = 42,
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """Split texts and labels into train / test halves.

    Most detection benchmarks (CHEAT, HC3, M4, …) ship without an official
    train/test split.  For zero-shot detectors that need a calibration set,
    use this to carve a random 50 % (or any ratio) from the full data.

    Args:
        texts: Input texts.
        labels: Ground-truth labels (1 = AI, 0 = human).
        test_size: Fraction held out for evaluation (default 0.5).
        random_state: Reproducibility seed.

    Returns:
        ``(train_texts, train_labels), (test_texts, test_labels)``
    """
    tr_texts, te_texts, tr_labels, te_labels = train_test_split(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    return (tr_texts, tr_labels), (te_texts, te_labels)


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str = "youden",
) -> float:
    """Find the optimal decision threshold on a calibration set.

    Args:
        scores: Detector scores (higher = more likely AI).
        labels: Ground-truth labels (1 = AI, 0 = human).
        method: Threshold selection strategy.

            - ``"youden"``: maximize Youden's J = TPR - FPR (default).
            - ``"f1"``: maximize F1 score.
            - ``"eer"``: equal error rate — threshold where FPR ≈ FNR.

    Returns:
        Optimal threshold (float).
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(labels)) < 2:
        # Only one class present; return median score as fallback.
        return float(np.median(scores))

    if method == "eer":
        # Equal Error Rate: find threshold where FPR ≈ 1 - TPR (i.e. FPR ≈ FNR).
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1.0 - tpr
        idx = int(np.argmin(np.abs(fpr - fnr)))
        return float(thresholds[idx])

    if method == "f1":
        # Sweep over unique score values to maximize F1.
        candidates = np.unique(scores)
        # Limit to at most 5000 candidate thresholds for efficiency.
        if len(candidates) > 5000:
            candidates = np.quantile(scores, np.linspace(0.01, 0.99, 5000))
        best_t, best_f1 = 0.0, -1.0
        for t in candidates:
            preds = (scores > t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    # Default: Youden's J.
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return float(thresholds[idx])


def compute_metrics(
    labels,
    scores,
    threshold: float = 0.5,
    target_fpr: float = 0.01,
) -> Dict[str, float]:
    """Compute standard binary-classification detection metrics.

    Args:
        labels: Ground-truth labels (1 = AI-generated, 0 = human).
        scores: Detector scores (higher = more likely AI-generated).
        threshold: Decision threshold applied to ``scores``.
        target_fpr: FPR level for ``tpr_at_fpr`` and ``f1_at_fpr`` metrics
            (default 0.01 = 1 %).
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores > threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    # Rank-based metrics require both classes to be present.
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["avg_precision"] = float(average_precision_score(labels, scores))

        # TPR@FPR and F1@FPR: find the operating point on the ROC curve
        # where FPR is closest to (but not exceeding) target_fpr.
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        idx = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
        if idx >= 0:
            t_at_fpr = float(roc_thresholds[idx])
            metrics[f"tpr_at_fpr_{target_fpr}"] = float(tpr[idx])
            preds_at_fpr = (scores > t_at_fpr).astype(int)
            metrics[f"f1_at_fpr_{target_fpr}"] = float(
                f1_score(labels, preds_at_fpr, zero_division=0)
            )
        else:
            metrics[f"tpr_at_fpr_{target_fpr}"] = 0.0
            metrics[f"f1_at_fpr_{target_fpr}"] = 0.0
    return metrics


class Evaluator:
    """Run a detector over texts and compute metrics, with three modes.

    **Mode 1 — Train-only (calibrate)**:
        Pass ``train_texts`` + ``train_labels`` + ``method``.
        Scores on train, finds optimal threshold, reports train metrics.
        Returns ``(threshold, metrics)``.

    **Mode 2 — Test-only (fixed threshold)**:
        Pass ``test_texts`` + ``test_labels`` + ``threshold``.
        Scores on test with the given threshold.
        Returns ``metrics``.

    **Mode 3 — Train + Test (calibrate then evaluate)**:
        Pass both train and test data + ``method``.
        Calibrates threshold on train, evaluates on test.
        Returns ``(threshold, metrics)``.

    Example::

        ev = Evaluator()

        # Mode 1: only train
        threshold, train_metrics = ev.evaluate(
            detector, train_texts=..., train_labels=..., method="youden",
        )

        # Mode 2: only test
        test_metrics = ev.evaluate(
            detector, test_texts=..., test_labels=..., threshold=0.5,
        )

        # Mode 3: both
        threshold, test_metrics = ev.evaluate(
            detector,
            train_texts=..., train_labels=...,
            test_texts=..., test_labels=...,
            method="youden",
        )
    """

    def __init__(self, threshold: float = 0.5, target_fpr: float = 0.01):
        self.threshold = threshold
        self.target_fpr = target_fpr

    def evaluate(
        self,
        detector,
        train_texts: Optional[Sequence[str]] = None,
        train_labels=None,
        test_texts: Optional[Sequence[str]] = None,
        test_labels=None,
        threshold: Optional[float] = None,
        method: str = "youden",
    ):
        """Evaluate a detector in one of three modes.

        Args:
            detector: A :class:`BaseDetector` instance.
            train_texts, train_labels: Training data (for calibration).
            test_texts, test_labels: Test data (for evaluation).
            threshold: Manual threshold (Mode 2). If ``None`` in test-only
                mode, falls back to ``self.threshold``.
            method: Threshold selection strategy — ``"youden"`` (default),
                ``"f1"``, or ``"eer"`` (Modes 1 and 3).

        Returns:
            - Mode 1 (train-only): ``(threshold, train_metrics)``
            - Mode 2 (test-only): ``test_metrics``
            - Mode 3 (train+test): ``(threshold, test_metrics)``
        """
        has_train = train_texts is not None
        has_test = test_texts is not None

        if not has_train and not has_test:
            raise ValueError(
                "Must provide at least train_texts or test_texts."
            )

        # --- Mode 1: train-only ---
        if has_train and not has_test:
            scores = np.asarray(detector.score(train_texts), dtype=float)
            labels = np.asarray(train_labels, dtype=int)
            t = calibrate_threshold(scores, labels, method=method)
            self.threshold = t
            metrics = compute_metrics(
                labels, scores, threshold=t, target_fpr=self.target_fpr
            )
            return t, metrics

        # --- Mode 2: test-only ---
        if has_test and not has_train:
            t = threshold if threshold is not None else self.threshold
            scores = np.asarray(detector.score(test_texts), dtype=float)
            labels = np.asarray(test_labels, dtype=int)
            metrics = compute_metrics(
                labels, scores, threshold=t, target_fpr=self.target_fpr
            )
            return metrics

        # --- Mode 3: train + test ---
        # Calibrate on train
        train_scores = np.asarray(detector.score(train_texts), dtype=float)
        train_labels_arr = np.asarray(train_labels, dtype=int)
        t = calibrate_threshold(train_scores, train_labels_arr, method=method)
        self.threshold = t

        # Evaluate on test
        test_scores = np.asarray(detector.score(test_texts), dtype=float)
        test_labels_arr = np.asarray(test_labels, dtype=int)
        metrics = compute_metrics(
            test_labels_arr, test_scores,
            threshold=t, target_fpr=self.target_fpr,
        )
        return t, metrics
