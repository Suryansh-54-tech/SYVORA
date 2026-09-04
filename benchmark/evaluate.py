"""
NYAYANTRA — Comprehensive Benchmark Evaluation Suite
=================================================
Reproducible evaluation harness executing on the untouched held-out test split (N=180).

Evaluates:
1. Machine Learning Classification Performance (PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix)
2. Probability Calibration Quality (Raw vs. Calibrated Brier Score, Reliability Bins)
3. Decision-Engine Financial Simulation vs. Ground Truth (Expected Value, Net Rupees Saved, False-Positive Costs)
4. Evidentiary Completeness & Readiness Index Distributions
5. Strategy Comparison: NYAYANTRA Triage vs. Blind Contesting vs. Passive Surrender

Outputs:
- Machine-readable benchmark report: benchmark/benchmark_results.json
- Formatted console output
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ml.features import FeaturePipeline
from src.engine import DecisionEngine, DecisionVerdict


class BenchmarkRunner:
    def __init__(self):
        self.test_df = pd.read_csv(config.TEST_PATH)
        self.pipeline = FeaturePipeline()
        self.engine = DecisionEngine()
        
        # Load saved model bundle
        model_path = os.path.join(config.MODELS_DIR, "sentinel_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
        
        bundle = joblib.load(model_path)
        self.calibrated_model = bundle["model"]
        self.base_model = bundle["base_model"]
        self.feature_names = bundle["feature_names"]

    def run_ml_benchmarks(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates classification and calibration metrics on test split."""
        raw_probs = self.base_model.predict_proba(X_test)[:, 1]
        cal_probs = self.calibrated_model.predict_proba(X_test)[:, 1]

        # Classification metrics at tau = 0.50
        preds_50 = (cal_probs >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_50).ravel()

        raw_brier = float(brier_score_loss(y_test, raw_probs))
        cal_brier = float(brier_score_loss(y_test, cal_probs))

        # Reliability / Calibration bins
        prob_true, prob_pred = calibration_curve(y_test, cal_probs, n_bins=5, strategy="uniform")
        cal_bins = [
            {"predicted_bin_mean": round(float(p), 4), "empirical_true_fraction": round(float(t), 4)}
            for p, t in zip(prob_pred, prob_true)
        ]

        return {
            "sample_size": len(y_test),
            "actual_win_rate": round(float(y_test.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y_test, cal_probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test, cal_probs)), 4),
            "raw_brier_score": round(raw_brier, 4),
            "calibrated_brier_score": round(cal_brier, 4),
            "brier_improvement_pct": round(float((raw_brier - cal_brier) / raw_brier * 100), 2),
            "classification_at_tau_0_50": {
                "threshold": 0.50,
                "precision": round(float(precision_score(y_test, preds_50, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, preds_50, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, preds_50, zero_division=0)), 4),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "calibration_bins": cal_bins,
        }

    def run_decision_engine_benchmarks(self) -> Dict[str, Any]:
        """
        Runs every test dispute through the decision engine and evaluates financial
        outcomes against synthetic ground truth.

        Separates:
        1. Autonomous Direct Return (zero human assumptions)
        2. REVIEW Queue characteristics
        3. Human-in-the-Loop (HITL) Sensitivity Model (70%, 85%, 100% Oracle)
        """
        decisions = []
        verdict_counts = {"CONTEST": 0, "REVIEW": 0, "SURRENDER": 0}
        evidence_scores = []
        missing_counts = {}

        total_dispute_amount = 0.0
        total_winnable_amount = 0.0
        total_actual_losses = 0
        fee = self.engine.arbitration_fee_inr

        # 1. Autonomous Tracking (CONTEST & SURRENDER only)
        auto_contest_count = 0
        auto_contest_recovered_gmv = 0.0
        auto_contest_fees_incurred = 0.0
        auto_surrender_fees_avoided = 0.0
        auto_surrender_missed_gmv = 0.0

        # 2. REVIEW Queue Tracking
        review_count = 0
        review_disputed_gmv = 0.0
        review_winnable_gmv = 0.0
        review_winnable_count = 0
        review_unwinnable_count = 0

        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i].to_dict()
            ground_truth = int(row["dispute_outcome"])
            amount = float(row["txn_amount_inr"])
            total_dispute_amount += amount

            if ground_truth == 1:
                total_winnable_amount += amount
            else:
                total_actual_losses += 1

            # Run engine evaluation
            eval_res = self.engine.evaluate_dispute(row, include_shap=False)
            verdict = eval_res["decision"]
            verdict_counts[verdict] += 1
            
            score = eval_res["evidence_analysis"]["readiness_score"]
            evidence_scores.append(score)

            for m in eval_res["evidence_analysis"]["missing_elements"]:
                missing_counts[m] = missing_counts.get(m, 0) + 1

            # Categorize economics by verdict
            if verdict == "CONTEST":
                auto_contest_count += 1
                if ground_truth == 1:
                    auto_contest_recovered_gmv += amount
                else:
                    auto_contest_fees_incurred += fee  # False positive fee
            elif verdict == "SURRENDER":
                if ground_truth == 0:
                    auto_surrender_fees_avoided += fee  # True negative fee saved
                else:
                    auto_surrender_missed_gmv += amount  # False negative missed GMV
            elif verdict == "REVIEW":
                review_count += 1
                review_disputed_gmv += amount
                if ground_truth == 1:
                    review_winnable_gmv += amount
                    review_winnable_count += 1
                else:
                    review_unwinnable_count += 1

            decisions.append({
                "dispute_id": row["dispute_id"],
                "amount_inr": amount,
                "ground_truth": ground_truth,
                "verdict": verdict,
                "win_prob": eval_res["financial_analysis"]["calibrated_win_probability"],
                "expected_value": eval_res["financial_analysis"]["expected_value_inr"],
                "evidence_score": score,
            })

        # --- Baseline Strategies ---
        net_strat_a_passive = -total_winnable_amount
        net_strat_b_blind = total_winnable_amount - (total_actual_losses * fee)

        # --- Metric 1: Autonomous Direct Return (Pure System, Zero Human Assumptions) ---
        net_autonomous_direct = auto_contest_recovered_gmv - auto_contest_fees_incurred

        # --- Metric 3: HITL Sensitivity Analysis ---
        def calc_hitl_net(accuracy_rate: float) -> Dict[str, float]:
            recovered = auto_contest_recovered_gmv + (accuracy_rate * review_winnable_gmv)
            # In human review, errors cause unneeded fees on loss cases
            fees = auto_contest_fees_incurred + ((1.0 - accuracy_rate) * review_unwinnable_count * fee)
            net = recovered - fees
            return {
                "assumed_human_accuracy_pct": round(accuracy_rate * 100, 1),
                "total_net_financial_outcome_inr": round(net, 2),
                "advantage_over_blind_contesting_inr": round(net - net_strat_b_blind, 2),
            }

        hitl_sensitivity = {
            "human_accuracy_70pct": calc_hitl_net(0.70),
            "human_accuracy_85pct": calc_hitl_net(0.85),
            "human_accuracy_100pct_oracle": {
                **calc_hitl_net(1.00),
                "note": "Theoretical upper bound / oracle assumption (assumes 100% human review precision)"
            },
        }

        return {
            "verdict_distribution": {
                "CONTEST": verdict_counts["CONTEST"],
                "REVIEW": verdict_counts["REVIEW"],
                "SURRENDER": verdict_counts["SURRENDER"],
                "CONTEST_pct": round(verdict_counts["CONTEST"] / len(self.test_df) * 100, 2),
                "REVIEW_pct": round(verdict_counts["REVIEW"] / len(self.test_df) * 100, 2),
                "SURRENDER_pct": round(verdict_counts["SURRENDER"] / len(self.test_df) * 100, 2),
            },
            "evidence_quality_distribution": {
                "mean_readiness_score": round(float(np.mean(evidence_scores)), 2),
                "median_readiness_score": round(float(np.median(evidence_scores)), 2),
                "min_readiness_score": int(np.min(evidence_scores)),
                "max_readiness_score": int(np.max(evidence_scores)),
                "missing_evidence_frequency": missing_counts,
            },
            "financial_simulation_metrics": {
                "total_disputed_gmv_inr": round(total_dispute_amount, 2),
                "total_winnable_gmv_inr": round(total_winnable_amount, 2),
                "arbitration_fee_parameter_inr": fee,
                "baselines": {
                    "strategy_a_passive_surrender_net_inr": round(net_strat_a_passive, 2),
                    "strategy_b_blind_contest_all_net_inr": round(net_strat_b_blind, 2),
                },
                "autonomous_direct_return": {
                    "total_auto_contested_cases": auto_contest_count,
                    "recovered_gmv_inr": round(auto_contest_recovered_gmv, 2),
                    "false_positive_fees_incurred_inr": round(auto_contest_fees_incurred, 2),
                    "net_autonomous_return_inr": round(net_autonomous_direct, 2),
                    "true_negative_fees_avoided_inr": round(auto_surrender_fees_avoided, 2),
                    "false_negative_missed_recovery_inr": round(auto_surrender_missed_gmv, 2),
                },
                "review_queue_metrics": {
                    "total_review_cases": review_count,
                    "review_disputed_gmv_inr": round(review_disputed_gmv, 2),
                    "review_winnable_gmv_inr": round(review_winnable_gmv, 2),
                    "review_winnable_cases_count": review_winnable_count,
                    "review_unwinnable_cases_count": review_unwinnable_count,
                },
                "hitl_sensitivity_analysis": hitl_sensitivity,
            },
            "decisions_log": decisions,
        }

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Executes full benchmark suite and saves machine-readable report."""
        print("=" * 65)
        print("  NYAYANTRA -- Comprehensive Benchmark Evaluation Suite")
        print("=" * 65)

        # 1. Feature processing
        X_test, y_test = self.pipeline.process_split(self.test_df)

        print("\n[1/3] Computing ML Classification & Calibration Metrics...")
        ml_results = self.run_ml_benchmarks(X_test, y_test)

        print("\n[2/3] Simulating Decision Engine Economics on Held-Out Test Set...")
        decision_results = self.run_decision_engine_benchmarks()

        report = {
            "benchmark_metadata": {
                "dataset": "data/test.csv",
                "test_split_size": len(self.test_df),
                "split_type": "Chronological Hold-Out",
                "random_seed": config.RANDOM_SEED,
            },
            "ml_performance": ml_results,
            "decision_engine_performance": {
                "verdict_distribution": decision_results["verdict_distribution"],
                "evidence_quality": decision_results["evidence_quality_distribution"],
                "financial_simulation": decision_results["financial_simulation_metrics"],
            }
        }

        # 3. Save report to disk
        out_path = os.path.join(config.PROJECT_ROOT, "benchmark", "benchmark_results.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[3/3] [OK] Saved machine-readable benchmark to: {out_path}")

        # 4. Print structured summary
        self.print_summary(report)

        return report

    def print_summary(self, report: Dict[str, Any]) -> None:
        ml = report["ml_performance"]
        cls50 = ml["classification_at_tau_0_50"]
        dec = report["decision_engine_performance"]
        verdict = dec["verdict_distribution"]
        fin = dec["financial_simulation"]

        print("\n" + "=" * 65)
        print("  NYAYANTRA BENCHMARK SUMMARY (HELD-OUT TEST SET, N=180)")
        print("=" * 65)

        print("\n--- 1. MACHINE LEARNING & CALIBRATION METRICS ---")
        print(f"  PR-AUC:                   {ml['pr_auc']:.4f} (Primary Imbalanced Ranking Metric)")
        print(f"  ROC-AUC:                  {ml['roc_auc']:.4f}")
        print(f"  Raw Brier Score:          {ml['raw_brier_score']:.4f}")
        print(f"  Calibrated Brier Score:   {ml['calibrated_brier_score']:.4f} (Improvement: {ml['brier_improvement_pct']}%)")
        print(f"  Precision (at tau=0.50):  {cls50['precision']:.2%}")
        print(f"  Recall (at tau=0.50):     {cls50['recall']:.2%}")
        print(f"  F1-Score (at tau=0.50):   {cls50['f1_score']:.4f}")
        print(f"  Confusion (at tau=0.50):  TN={cls50['true_negatives']}, FP={cls50['false_positives']}, FN={cls50['false_negatives']}, TP={cls50['true_positives']}")

        print("\n--- 2. TRIAGE VERDICT DISTRIBUTION ---")
        print(f"  CONTEST (Auto-defend):    {verdict['CONTEST']:3d} ({verdict['CONTEST_pct']}%)")
        print(f"  REVIEW (Human Queue):     {verdict['REVIEW']:3d} ({verdict['REVIEW_pct']}%)")
        print(f"  SURRENDER (Save Fee):     {verdict['SURRENDER']:3d} ({verdict['SURRENDER_pct']}%)")

        print("\n--- 3. FINANCIAL SIMULATION & STRATEGY COMPARISON ---")
        base = fin["baselines"]
        auto = fin["autonomous_direct_return"]
        rev = fin["review_queue_metrics"]
        hitl = fin["hitl_sensitivity_analysis"]

        print(f"  Total Disputed GMV:       INR {fin['total_disputed_gmv_inr']:,.2f}")
        print(f"  Total Winnable GMV:       INR {fin['total_winnable_gmv_inr']:,.2f}")
        print(f"  -------------------------------------------------------------")
        print(f"  [Baseline A] Passive Surrender All:  INR {base['strategy_a_passive_surrender_net_inr']:,.2f}")
        print(f"  [Baseline B] Blind Contest All:      INR {base['strategy_b_blind_contest_all_net_inr']:,.2f}")
        print(f"  -------------------------------------------------------------")
        print(f"  [Metric 1] Autonomous Direct Return (CONTEST-Only, 0 Human Assumptions):")
        print(f"    - Net Financial Return:            +INR {auto['net_autonomous_return_inr']:,.2f}")
        print(f"    - Recovered GMV:                   +INR {auto['recovered_gmv_inr']:,.2f} ({auto['total_auto_contested_cases']} auto-contested disputes)")
        print(f"    - False Positive Fees Paid:        -INR {auto['false_positive_fees_incurred_inr']:,.2f}")
        print(f"    - True Negative Fees Avoided:      +INR {auto['true_negative_fees_avoided_inr']:,.2f} (via smart SURRENDER)")
        print(f"  -------------------------------------------------------------")
        print(f"  [Metric 2] REVIEW Queue (Escalated to Operations):")
        print(f"    - Review Case Volume:              {rev['total_review_cases']} disputes ({rev['total_review_cases']/len(self.test_df):.1%})")
        print(f"    - Disputed GMV in Review:          INR {rev['review_disputed_gmv_inr']:,.2f}")
        print(f"    - Winnable GMV in Review:          INR {rev['review_winnable_gmv_inr']:,.2f}")
        print(f"  -------------------------------------------------------------")
        print(f"  [Metric 3] Human-in-the-Loop (HITL) Sensitivity Analysis:")
        print(f"    - At 70% Human Precision:          INR {hitl['human_accuracy_70pct']['total_net_financial_outcome_inr']:,.2f} (Advantage: +INR {hitl['human_accuracy_70pct']['advantage_over_blind_contesting_inr']:,.2f})")
        print(f"    - At 85% Human Precision:          INR {hitl['human_accuracy_85pct']['total_net_financial_outcome_inr']:,.2f} (Advantage: +INR {hitl['human_accuracy_85pct']['advantage_over_blind_contesting_inr']:,.2f})")
        print(f"    - At 100% Precision (Oracle):      INR {hitl['human_accuracy_100pct_oracle']['total_net_financial_outcome_inr']:,.2f} (Theoretical Upper Bound)")

        print("\n--- 4. EVIDENTIARY READINESS ---")
        ev = dec["evidence_quality"]
        print(f"  Mean Evidence Readiness:  {ev['mean_readiness_score']}/100")
        print(f"  Median Readiness:         {ev['median_readiness_score']}/100")
        print(f"  Top Missing Evidence:     {list(ev['missing_evidence_frequency'].keys())[:2]}")

        print("\n" + "=" * 65)


def main():
    runner = BenchmarkRunner()
    runner.run_all_benchmarks()


if __name__ == "__main__":
    main()
