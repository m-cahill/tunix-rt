#!/usr/bin/env python3
"""M49 Observer Training — Logistic Regression Error Detector.

Trains a lightweight observer model to detect mismatches between 
model-generated answers and expected answers.

The key insight from M48: The generator computes correctly but cannot
detect when its answer differs from expectation. An external observer
CAN detect this mismatch.

Design Decisions (from M49 confirmation):
- TF-IDF features + answer comparison features
- Logistic regression classifier (simple, interpretable)
- Binary classification (error_present: 0/1)

Note: Uses pure numpy/Python implementation to avoid Windows DLL issues.

Output:
- models/observer_model.npz
- metrics/observer_metrics.json
- metrics/predictions.jsonl

Author: M49 Observer Error Detection Milestone
Date: 2026-01-09
"""

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
METRICS_DIR = PROJECT_DIR / "metrics"

# Random seed
SEED = 42


# ============================================================
# Feature Extraction
# ============================================================

def extract_answer_from_output(output: str) -> str:
    """Extract the final answer from model output."""
    output = output.strip()
    
    # Look for patterns like "648 km", "-21", "$100", etc.
    # Check last line first
    lines = output.split('\n')
    last_line = lines[-1].strip() if lines else ""
    
    # Try to extract a number with unit
    km_match = re.search(r'(\d+)\s*km', output)
    if km_match:
        return km_match.group(1) + " km"
    
    # Try to extract a signed number
    num_match = re.search(r'(-?\d+(?:\.\d+)?)\s*$', output)
    if num_match:
        return num_match.group(1)
    
    # Try final result pattern
    result_match = re.search(r'(?:result|answer)[:\s]+(-?\d+(?:\.\d+)?)', output, re.IGNORECASE)
    if result_match:
        return result_match.group(1)
    
    return last_line


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    # Remove units
    answer = re.sub(r'\s*km$', '', answer)
    answer = re.sub(r'\s*hours?$', '', answer)
    answer = re.sub(r'^\$', '', answer)
    # Remove commas
    answer = answer.replace(',', '')
    # Extract number
    match = re.search(r'-?\d+(?:\.\d+)?', answer)
    if match:
        return match.group()
    return answer


def answers_match(expected: str, generated: str) -> bool:
    """Check if two answers match after normalization."""
    exp_norm = normalize_answer(expected)
    gen_norm = normalize_answer(generated)
    
    if not exp_norm or not gen_norm:
        return False
    
    try:
        return float(exp_norm) == float(gen_norm)
    except ValueError:
        return exp_norm == gen_norm


def extract_features(sample: dict) -> list:
    """Extract features for observer classification.
    
    Features:
    1. Answer match (0/1)
    2. Answer numeric difference (normalized)
    3. Output length
    4. Has VERIFY keyword
    5. Has CORRECT keyword
    6. Has "No correction needed"
    """
    output = sample.get("input_text", "")
    expected = sample.get("expected_answer", "")
    
    # Extract generated answer
    generated = extract_answer_from_output(output)
    
    # Feature 1: Answer match
    match = 1.0 if answers_match(expected, generated) else 0.0
    
    # Feature 2: Numeric difference (normalized)
    try:
        exp_num = float(normalize_answer(expected))
        gen_num = float(normalize_answer(generated))
        diff = abs(exp_num - gen_num) / (abs(exp_num) + 1)  # Normalized diff
        diff = min(diff, 10)  # Cap at 10
    except (ValueError, TypeError):
        diff = 5.0  # Default if can't parse
    
    # Feature 3: Output length (normalized)
    length = len(output) / 500.0  # Normalize to ~1
    
    # Feature 4: Has VERIFY
    has_verify = 1.0 if "verify" in output.lower() else 0.0
    
    # Feature 5: Has CORRECT
    has_correct = 1.0 if "correct" in output.lower() else 0.0
    
    # Feature 6: Has "No correction needed"
    no_correction = 1.0 if "no correction needed" in output.lower() else 0.0
    
    return [match, diff, length, has_verify, has_correct, no_correction]


# ============================================================
# Simple Logistic Regression (numpy-only)
# ============================================================

class SimpleLogisticRegression:
    """Pure numpy logistic regression for binary classification."""
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, seed=42):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.seed = seed
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            linear = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear)
            
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# Metrics (numpy-only)
# ============================================================

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix_values(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return int(tn), int(fp), int(fn), int(tp)


def precision_recall_f1(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_values(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def roc_auc_score_simple(y_true, y_proba):
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tpr_values = []
    fpr_values = []
    
    thresholds = np.unique(y_proba)
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix_values(y_true, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    fpr_values = [0] + fpr_values + [1]
    tpr_values = [0] + tpr_values + [1]
    
    sorted_indices = np.argsort(fpr_values)
    fpr_sorted = np.array(fpr_values)[sorted_indices]
    tpr_sorted = np.array(tpr_values)[sorted_indices]
    
    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i-1]
        height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2
        auc += width * height
    return float(auc)


# ============================================================
# Data Loading
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("M49 Observer Training — Answer Mismatch Detection")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[LOAD] Loading datasets...")
    train_data = load_jsonl(DATA_DIR / "observer_train.jsonl")
    val_data = load_jsonl(DATA_DIR / "observer_val.jsonl")
    test_data = load_jsonl(DATA_DIR / "observer_test.jsonl")
    
    print(f"       Train: {len(train_data)}")
    print(f"       Val: {len(val_data)}")
    print(f"       Test: {len(test_data)}")
    
    # Extract features
    print("\n[FEATURES] Extracting features...")
    
    def prepare_data(samples):
        X = []
        y = []
        for s in samples:
            features = extract_features(s)
            X.append(features)
            y.append(s["label"])
        return np.array(X), np.array(y)
    
    X_train, y_train = prepare_data(train_data)
    X_val, y_val = prepare_data(val_data)
    X_test, y_test = prepare_data(test_data)
    
    print(f"         Features: {X_train.shape[1]}")
    print(f"         Train shape: {X_train.shape}")
    
    # Print feature analysis
    print("\n[ANALYSIS] Feature correlations with label:")
    feature_names = ["match", "diff", "length", "has_verify", "has_correct", "no_correction"]
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        print(f"           {name}: {corr:.3f}")
    
    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # Train logistic regression
    print("\n[TRAIN] Training logistic regression...")
    clf = SimpleLogisticRegression(
        learning_rate=0.5,
        n_iterations=2000,
        seed=SEED,
    )
    clf.fit(X_train, y_train)
    print("        Training complete")
    
    # Print learned weights
    print("\n[WEIGHTS] Learned feature weights:")
    for i, name in enumerate(feature_names):
        print(f"           {name}: {clf.weights[i]:.3f}")
    print(f"           bias: {clf.bias:.3f}")
    
    # Evaluate on validation set
    print("\n[EVAL] Validation set:")
    y_val_pred = clf.predict(X_val)
    y_val_proba = clf.predict_proba(X_val)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score_simple(y_val, y_val_proba)
    print(f"        Accuracy: {val_acc:.2%}")
    print(f"        AUC: {val_auc:.3f}")
    
    # Evaluate on test set
    print("\n[EVAL] Test set:")
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score_simple(y_test, y_test_proba)
    precision, recall, f1 = precision_recall_f1(y_test, y_test_pred)
    
    print(f"        Accuracy: {test_acc:.2%}")
    print(f"        AUC: {test_auc:.3f}")
    print(f"        Precision: {precision:.2%}")
    print(f"        Recall: {recall:.2%}")
    print(f"        F1: {f1:.2%}")
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix_values(y_test, y_test_pred)
    print(f"\n        Confusion Matrix:")
    print(f"        TN={tn}, FP={fp}")
    print(f"        FN={fn}, TP={tp}")
    
    # Save model
    model_path = MODELS_DIR / "observer_model.npz"
    np.savez(model_path, weights=clf.weights, bias=np.array([clf.bias]), mean=mean, std=std)
    print(f"\n[SAVE] Model: {model_path}")
    
    # Save metrics
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "approach": "Answer Mismatch Detection",
        "features": feature_names,
        "classifier": "SimpleLogisticRegression",
        "seed": SEED,
        "dataset_sizes": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "validation": {
            "accuracy": float(val_acc),
            "auc": float(val_auc),
        },
        "test": {
            "accuracy": float(test_acc),
            "auc": float(test_auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
        },
        "generator_comparison": {
            "generator_detection_rate": 0.0,
            "observer_detection_rate": float(recall),
            "improvement": f"+{recall*100:.1f}pp",
        },
        "learned_weights": {name: float(clf.weights[i]) for i, name in enumerate(feature_names)},
    }
    
    metrics_path = METRICS_DIR / "observer_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"       Metrics: {metrics_path}")
    
    # Save predictions
    predictions = []
    for i, sample in enumerate(test_data):
        pred = {
            "id": sample["id"],
            "prompt": sample.get("prompt", ""),
            "expected_answer": sample.get("expected_answer", ""),
            "generated_answer": extract_answer_from_output(sample.get("input_text", "")),
            "true_label": int(sample["label"]),
            "predicted_label": int(y_test_pred[i]),
            "confidence": float(y_test_proba[i]),
            "correct": int(sample["label"]) == int(y_test_pred[i]),
            "features": {name: float(X_test[i, j] * std[j] + mean[j]) for j, name in enumerate(feature_names)},
        }
        predictions.append(pred)
    
    predictions_path = METRICS_DIR / "predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"       Predictions: {predictions_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("M49 Observer Training Complete!")
    print("=" * 60)
    print(f"\n  Generator error detection rate: 0.0%")
    print(f"  Observer error detection rate:  {recall*100:.1f}%")
    print(f"  Improvement:                    +{recall*100:.1f}pp")
    print(f"\n  Test AUC: {test_auc:.3f}")
    
    if test_auc >= 0.7:
        print(f"\n  [STRONG SIGNAL] AUC >= 0.7 supports the architectural separation hypothesis")
    elif test_auc >= 0.6:
        print(f"\n  [MEANINGFUL SIGNAL] AUC ~0.6 shows non-trivial detection capability")
    else:
        print(f"\n  [WEAK SIGNAL] AUC ~0.5 suggests observer does not improve on random")
    
    return 0


if __name__ == "__main__":
    exit(main())
