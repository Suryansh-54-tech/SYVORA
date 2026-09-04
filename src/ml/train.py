"""
NYAYANTRA — Machine Learning Model Training & Probability Calibration
==================================================================
Trains and calibrates tabular ML models for post-payment dispute win prediction.

Protocol:
1. Feature Extraction: Fit pipeline on Train, transform Val and Test.
2. Model Comparison: Benchmark Logistic Regression, Random Forest, LightGBM on Validation.
3. Probability Calibration: Calibrate best model on Validation split ONLY (Isotonic vs Sigmoid).
4. Held-Out Evaluation: Final unbiased metrics evaluated strictly on Test split.
5. Model Persistence: Saves calibrated model artifact to models/sentinel_model.joblib.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.ml.features import FeaturePipeline


# ---------------------------------------------------------------------------
# Training & Benchmarking Engine
# ---------------------------------------------------------------------------

class DisputeModelTrainer:
    def __init__(self):
        self.pipeline = FeaturePipeline()
        self.best_uncalibrated_model = None
        self.calibrated_model = None
        self.best_model_name: str = ""
        self.calibration_method: str = ""
        self.feature_names = []

    def load_and_preprocess_data(self) -> Tuple[
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series
    ]:
        """
        Loads raw temporal CSVs and extracts feature matrices.
        Ensures strict chronological isolation (Train -> Val -> Test).
        """
        train_raw = pd.read_csv(config.TRAIN_PATH)
        val_raw = pd.read_csv(config.VAL_PATH)
        test_raw = pd.read_csv(config.TEST_PATH)

        X_train, y_train = self.pipeline.process_split(train_raw)
        X_val, y_val = self.pipeline.process_split(val_raw)
        X_test, y_test = self.pipeline.process_split(test_raw)

        self.feature_names = self.pipeline.feature_names

        return X_train, y_train, X_val, y_val, X_test, y_test

    def benchmark_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Trains candidate models on Train split, evaluates on Validation split.
        """
        candidates = {
            "LogisticRegression": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    C=0.1,
                    random_state=config.RANDOM_SEED
                )
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_leaf=5,
                random_state=config.RANDOM_SEED
            )
        }

        if HAS_LIGHTGBM:
            candidates["LightGBM"] = LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=config.RANDOM_SEED,
                verbose=-1
            )

        val_results = {}
        best_score = -1.0
        best_name = None

        print("\n--- Model Benchmark on Validation Split (Train N={}, Val N={}) ---".format(len(X_train), len(X_val)))
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]

            roc_auc = roc_auc_score(y_val, probs)
            pr_auc = average_precision_score(y_val, probs)
            brier = brier_score_loss(y_val, probs)

            val_results[name] = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "model": model
            }

            print(f"  {name:18s} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")

            # Primary selection criterion: PR-AUC on validation split
            if pr_auc > best_score:
                best_score = pr_auc
                best_name = name

        self.best_model_name = best_name
        self.best_uncalibrated_model = val_results[best_name]["model"]
        print(f"\n  Selected Candidate: {self.best_model_name} (Highest Validation PR-AUC: {best_score:.4f})")

        return val_results

    def calibrate_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[Any, str, float]:
        """
        Calibrates the best model using 5-fold Out-Of-Fold (OOF) Cross-Validation
        on the training split, and selects the optimal calibrator via unbiased
        evaluation on the hold-out Validation split.

        This eliminates in-sample optimism and guarantees that validation
        Brier score selection is completely independent of calibration fitting.
        """
        # Uncalibrated baseline on validation
        uncal_probs = self.best_uncalibrated_model.predict_proba(X_val)[:, 1]
        uncal_brier = brier_score_loss(y_val, uncal_probs)

        # Clone base model hyperparams for calibration estimators
        if self.best_model_name == "RandomForest":
            base_clone_1 = RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_leaf=5, random_state=config.RANDOM_SEED
            )
            base_clone_2 = RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_leaf=5, random_state=config.RANDOM_SEED
            )
        elif self.best_model_name == "LogisticRegression":
            base_clone_1 = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=0.1, random_state=config.RANDOM_SEED)
            )
            base_clone_2 = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=0.1, random_state=config.RANDOM_SEED)
            )
        else:
            base_clone_1 = RandomForestClassifier(random_state=config.RANDOM_SEED)
            base_clone_2 = RandomForestClassifier(random_state=config.RANDOM_SEED)

        # 1. 5-Fold OOF Sigmoid (Platt) Calibration
        cal_sigmoid = CalibratedClassifierCV(
            estimator=base_clone_1,
            method="sigmoid",
            cv=5
        )
        cal_sigmoid.fit(X_train, y_train)
        sig_probs = cal_sigmoid.predict_proba(X_val)[:, 1]
        sig_brier = brier_score_loss(y_val, sig_probs)

        # 2. 5-Fold OOF Isotonic Calibration
        cal_isotonic = CalibratedClassifierCV(
            estimator=base_clone_2,
            method="isotonic",
            cv=5
        )
        cal_isotonic.fit(X_train, y_train)
        iso_probs = cal_isotonic.predict_proba(X_val)[:, 1]
        iso_brier = brier_score_loss(y_val, iso_probs)

        print("\n--- Out-of-Fold Calibration Comparison (Evaluated on Hold-out Validation Split) ---")
        print(f"  Uncalibrated Raw Model         | Val Brier: {uncal_brier:.4f}")
        print(f"  5-Fold OOF Sigmoid / Platt     | Val Brier: {sig_brier:.4f}")
        print(f"  5-Fold OOF Isotonic Regression | Val Brier: {iso_brier:.4f}")

        # Unbiased selection on hold-out validation split
        if iso_brier < sig_brier:
            selected_method = "isotonic"
            selected_calibrator = cal_isotonic
            best_brier = iso_brier
        else:
            selected_method = "sigmoid"
            selected_calibrator = cal_sigmoid
            best_brier = sig_brier

        print(f"  Selected Calibration: {selected_method.upper()} (Lowest Unbiased Validation Brier: {best_brier:.4f})")

        self.calibrated_model = selected_calibrator
        self.calibration_method = selected_method

        return selected_calibrator, selected_method, best_brier

    def evaluate_test_set(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Evaluates the final calibrated model on the UNTOUCHED Test split.
        """
        # Raw uncalibrated probabilities for comparison
        raw_probs = self.best_uncalibrated_model.predict_proba(X_test)[:, 1]
        raw_brier = brier_score_loss(y_test, raw_probs)

        # Final calibrated probabilities
        cal_probs = self.calibrated_model.predict_proba(X_test)[:, 1]
        cal_brier = brier_score_loss(y_test, cal_probs)

        roc_auc = roc_auc_score(y_test, cal_probs)
        pr_auc = average_precision_score(y_test, cal_probs)

        preds = (cal_probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        results = {
            "test_size": len(y_test),
            "decision_threshold": threshold,
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "raw_brier_score": float(raw_brier),
            "calibrated_brier_score": float(cal_brier),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        }

        print("\n" + "=" * 65)
        print("  FINAL UNBIASED HELD-OUT TEST EVALUATION")
        print("=" * 65)
        print(f"  Test Sample Size:          {len(y_test)} records (chronological hold-out)")
        print(f"  Decision Threshold:        tau = {threshold:.2f}")
        print(f"  ROC-AUC:                   {roc_auc:.4f}")
        print(f"  PR-AUC:                    {pr_auc:.4f}")
        print(f"  Raw Brier Score:           {raw_brier:.4f}")
        print(f"  Calibrated Brier Score:    {cal_brier:.4f} (Lower = Better Calibration)")
        print(f"  Accuracy:                  {accuracy:.2%}")
        print(f"  Precision (Win Pred):      {precision:.2%}")
        print(f"  Recall (Win Capture):      {recall:.2%}")
        print(f"  F1-Score:                  {f1:.4f}")
        print("\n  Confusion Matrix:")
        print(f"    [TN (Predicted Lose, Actual Lose)]: {tn:3d}  |  [FP (Predicted Win, Actual Lose)]: {fp:3d}")
        print(f"    [FN (Predicted Lose, Actual Win)]:  {fn:3d}  |  [TP (Predicted Win, Actual Win)]:  {tp:3d}")
        print("=" * 65)

        return results

    def save_artifacts(self, metrics: Dict[str, Any]) -> str:
        """
        Serializes the trained calibrated model, feature list, and metadata.
        """
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        artifact_path = os.path.join(config.MODELS_DIR, "sentinel_model.joblib")

        bundle = {
            "model": self.calibrated_model,
            "base_model": self.best_uncalibrated_model,
            "base_model_name": self.best_model_name,
            "calibration_method": self.calibration_method,
            "feature_names": self.feature_names,
            "metrics": metrics,
        }

        joblib.dump(bundle, artifact_path)
        print(f"\n[OK] Model artifact successfully saved to: {artifact_path}")

        # Also save metrics JSON for easy inspection
        metrics_json_path = os.path.join(config.MODELS_DIR, "test_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] Test metrics summary saved to: {metrics_json_path}")

        return artifact_path


# ---------------------------------------------------------------------------
# Standalone Inference Helper
# ---------------------------------------------------------------------------

class SentinelRiskScorer:
    """
    Loads saved model artifact and scores single or batch dispute feature vectors.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, "sentinel_model.joblib")
        
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.base_model = bundle["base_model"]
        self.feature_names = bundle["feature_names"]
        self.metrics = bundle.get("metrics", {})

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated probability of winning the dispute."""
        # Align features
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict_proba(X_aligned)[:, 1]


# Canonical branding alias
NyayantraScorer = SentinelRiskScorer


def run_training_pipeline():
    print("=" * 65)
    print("  NYAYANTRA -- ML Model Training & Probability Calibration")
    print("=" * 65)

    trainer = DisputeModelTrainer()

    # 1. Load Data
    print("\n[1/5] Loading temporal dataset splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_and_preprocess_data()
    print(f"      Train: X={X_train.shape}, y={y_train.shape}")
    print(f"      Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"      Test:  X={X_test.shape}, y={y_test.shape}")

    # 2. Benchmark Candidates on Validation Split
    print("\n[2/5] Benchmarking candidate models...")
    trainer.benchmark_models(X_train, y_train, X_val, y_val)

    # 3. Calibrate using 5-Fold OOF on Train, Evaluate on Validation Split
    print("\n[3/5] Performing 5-Fold OOF probability calibration & unbiased validation selection...")
    trainer.calibrate_model(X_train, y_train, X_val, y_val)

    # 4. Evaluate strictly on Test Split
    print("\n[4/5] Evaluating calibrated model on untouched Test split...")
    test_metrics = trainer.evaluate_test_set(X_test, y_test, threshold=0.50)

    # 5. Save Artifacts
    print("\n[5/5] Saving model bundle and evaluation summary...")
    trainer.save_artifacts(test_metrics)

    # 6. Verify reload and inference
    print("\n--- Verifying Model Reload & Inference ---")
    scorer = SentinelRiskScorer()
    test_probs = scorer.predict_proba(X_test[:5])
    print(f"  Sample Calibrated Win Probabilities (first 5 test disputes):")
    for i, p in enumerate(test_probs):
        print(f"    Dispute #{i+1}: P(Win) = {p:.2%} | Actual Outcome = {y_test.iloc[i]}")

    print("\n" + "=" * 65)
    print("  Phase 3 ML Training & Calibration Pipeline Complete.")
    print("=" * 65)


if __name__ == "__main__":
    run_training_pipeline()
