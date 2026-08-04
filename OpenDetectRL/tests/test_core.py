"""Smoke tests proving the M0 scaffolding works end-to-end."""
import numpy as np

from opendetectrl import (
    BaseDetector,
    DetectionResult,
    list_detectors,
    load_detector,
    register_detector,
)
from opendetectrl.evaluation import (
    Evaluator,
    calibrate_threshold,
    compute_metrics,
    split_texts_labels,
)


@register_detector("toy_length")
class ToyLengthDetector(BaseDetector):
    """Toy detector: longer text -> higher score."""

    def score(self, texts):
        return np.array([len(t) for t in texts], dtype=float)


def test_registry_and_predict():
    assert "toy_length" in list_detectors()
    det = load_detector("toy_length", threshold=5.0)
    results = det.predict(["hi", "this is a much longer sentence"])
    assert all(isinstance(r, DetectionResult) for r in results)
    assert results[0].label == "human"
    assert results[1].label == "ai"


def test_compute_metrics():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.3, 0.8, 0.9])
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0


def test_compute_metrics_tpr_f1_at_fpr():
    """TPR@FPR and F1@FPR should be present and correct for separable data."""
    # 10 human (scores 0.01-0.10) + 10 AI (scores 0.91-1.00)
    scores = np.concatenate([np.arange(1, 11) / 100, np.arange(91, 101) / 100])
    labels = np.array([0] * 10 + [1] * 10)
    m = compute_metrics(labels, scores, threshold=0.5, target_fpr=0.01)
    # Keys should exist
    assert "tpr_at_fpr_0.01" in m
    assert "f1_at_fpr_0.01" in m
    # With perfectly separable data, TPR@FPR=0.01 should be high
    assert m["tpr_at_fpr_0.01"] >= 0.5
    assert m["f1_at_fpr_0.01"] > 0.0


def test_compute_metrics_custom_fpr():
    """target_fpr parameter should produce correct metric keys."""
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    m = compute_metrics(labels, scores, target_fpr=0.05)
    assert "tpr_at_fpr_0.05" in m
    assert "f1_at_fpr_0.05" in m


# ---------------------------------------------------------------------------
# Threshold calibration tests
# ---------------------------------------------------------------------------

def test_calibrate_threshold_youden():
    """Youden's J should pick the threshold that perfectly separates."""
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t = calibrate_threshold(scores, labels, method="youden")
    assert 0.3 <= t <= 0.7  # between or at the class boundary


def test_calibrate_threshold_f1():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t = calibrate_threshold(scores, labels, method="f1")
    assert 0.3 <= t <= 0.7


def test_calibrate_threshold_eer():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t = calibrate_threshold(scores, labels, method="eer")
    assert 0.3 <= t <= 0.7


def test_calibrate_threshold_single_class():
    """If only one class present, should return median score (no crash)."""
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 0, 0])
    t = calibrate_threshold(scores, labels)
    assert t == 0.2  # median


def test_split_texts_labels():
    """Split should respect stratification and sizes."""
    texts = [f"text_{i}" for i in range(100)]
    labels = [0] * 50 + [1] * 50
    (tr_t, tr_l), (te_t, te_l) = split_texts_labels(
        texts, labels, test_size=0.5, random_state=42
    )
    assert len(tr_t) == 50
    assert len(te_t) == 50
    assert sum(tr_l) == 25  # stratified
    assert sum(te_l) == 25


# ---------------------------------------------------------------------------
# Evaluator three-mode tests
# ---------------------------------------------------------------------------

def test_evaluate_mode1_train_only():
    """Mode 1: only train — calibrate threshold, return (threshold, metrics)."""
    detector = load_detector("toy_length")
    texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
    labels = [0, 0, 0, 1, 1, 1]
    evaluator = Evaluator()
    threshold, metrics = evaluator.evaluate(
        detector, train_texts=texts, train_labels=labels, method="youden"
    )
    assert isinstance(threshold, float)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    # The calibrated threshold should separate the two classes
    assert 3.0 <= threshold <= 4.0  # len("ccc")=3, len("dddd")=4


def test_evaluate_mode2_test_only():
    """Mode 2: only test — fixed threshold, return metrics."""
    detector = load_detector("toy_length")
    texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
    labels = [0, 0, 0, 1, 1, 1]
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        detector, test_texts=texts, test_labels=labels, threshold=3.5
    )
    assert isinstance(metrics, dict)
    assert metrics["accuracy"] == 1.0  # threshold 3.5 perfectly separates


def test_evaluate_mode3_train_test():
    """Mode 3: both — calibrate on train, evaluate on test."""
    detector = load_detector("toy_length")
    texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
    labels = [0, 0, 0, 1, 1, 1]
    (tr_t, tr_l), (te_t, te_l) = split_texts_labels(
        texts, labels, test_size=0.5, random_state=42
    )
    evaluator = Evaluator()
    threshold, metrics = evaluator.evaluate(
        detector,
        train_texts=tr_t, train_labels=tr_l,
        test_texts=te_t, test_labels=te_l,
        method="youden",
    )
    assert isinstance(threshold, float)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics


def test_evaluate_no_data_raises():
    """No train or test data should raise ValueError."""
    detector = load_detector("toy_length")
    evaluator = Evaluator()
    try:
        evaluator.evaluate(detector)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
