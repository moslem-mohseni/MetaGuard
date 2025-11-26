#!/usr/bin/env python
"""
Model Benchmark Script for MetaGuard

Author: Moslem Mohseni

This script benchmarks different models for fraud detection.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metaguard.models import (
    RandomForestModel,
    EnsembleModel,
    create_default_ensemble,
    is_xgboost_available,
)
from metaguard.features import FeatureEngineer, extract_features


def generate_synthetic_data(n_samples: int = 10000, fraud_ratio: float = 0.1) -> pd.DataFrame:
    """Generate synthetic transaction data for benchmarking.

    Args:
        n_samples: Number of samples to generate.
        fraud_ratio: Ratio of fraudulent transactions.

    Returns:
        DataFrame with synthetic transaction data.
    """
    np.random.seed(42)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # Normal transactions
    normal = pd.DataFrame({
        "amount": np.random.exponential(200, n_normal).clip(1, 5000),
        "hour": np.random.choice(range(8, 22), n_normal),  # Business hours
        "user_age_days": np.random.exponential(180, n_normal).clip(30, 1000),
        "transaction_count": np.random.poisson(5, n_normal).clip(0, 30),
        "is_fraud": 0,
    })

    # Fraudulent transactions
    fraud = pd.DataFrame({
        "amount": np.random.exponential(2000, n_fraud).clip(500, 10000),
        "hour": np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], n_fraud),  # Night hours
        "user_age_days": np.random.exponential(10, n_fraud).clip(1, 30),
        "transaction_count": np.random.poisson(30, n_fraud).clip(10, 100),
        "is_fraud": 1,
    })

    # Combine and shuffle
    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def benchmark_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Benchmark a single model.

    Args:
        model: Model instance to benchmark.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        model_name: Name for the model.

    Returns:
        Dictionary with benchmark results.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print("=" * 60)

    results: dict[str, Any] = {"model": model_name}

    # Training time
    start = time.time()
    metrics = model.train(X_train, y_train, validation_data=(X_test, y_test))
    train_time = time.time() - start

    results["train_time"] = train_time
    results.update(metrics)

    # Inference time (single prediction)
    start = time.time()
    for _ in range(100):
        _ = model.predict(X_test[:1])
    single_inference = (time.time() - start) / 100 * 1000  # ms

    results["single_inference_ms"] = single_inference

    # Batch inference time
    start = time.time()
    _ = model.predict(X_test)
    batch_inference = (time.time() - start) * 1000  # ms

    results["batch_inference_ms"] = batch_inference
    results["samples_per_second"] = len(X_test) / (batch_inference / 1000)

    # Feature importance
    try:
        importance = model.get_feature_importance()
        results["top_feature"] = max(importance, key=importance.get)
    except (NotImplementedError, ValueError):
        results["top_feature"] = "N/A"

    # Print results
    print(f"\nTraining Time: {train_time:.2f}s")
    print(f"Train Accuracy: {results.get('train_accuracy', 0):.4f}")
    print(f"Train AUC-ROC: {results.get('train_auc_roc', 0):.4f}")
    print(f"Val Accuracy: {results.get('val_accuracy', 0):.4f}")
    print(f"Val AUC-ROC: {results.get('val_auc_roc', 0):.4f}")
    print(f"Val F1: {results.get('val_f1', 0):.4f}")
    print(f"Single Inference: {single_inference:.2f}ms")
    print(f"Batch Inference: {batch_inference:.2f}ms ({len(X_test)} samples)")
    print(f"Throughput: {results['samples_per_second']:.0f} samples/sec")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MetaGuard models")
    parser.add_argument(
        "--samples", type=int, default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--fraud-ratio", type=float, default=0.1,
        help="Ratio of fraudulent transactions"
    )
    parser.add_argument(
        "--derived-features", action="store_true",
        help="Include derived features"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file for results"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MetaGuard Model Benchmark")
    print("=" * 60)
    print(f"Samples: {args.samples}")
    print(f"Fraud Ratio: {args.fraud_ratio:.1%}")
    print(f"Derived Features: {args.derived_features}")

    # Generate data
    print("\nGenerating synthetic data...")
    df = generate_synthetic_data(args.samples, args.fraud_ratio)
    print(f"Generated {len(df)} samples ({df['is_fraud'].sum()} fraud)")

    # Extract features
    print("\nExtracting features...")
    y = df["is_fraud"].values
    feature_cols = ["amount", "hour", "user_age_days", "transaction_count"]

    if args.derived_features:
        X, feature_names = extract_features(df[feature_cols].to_dict("records"))
        print(f"Features: {len(feature_names)} ({feature_names})")
    else:
        X = df[feature_cols].values
        feature_names = feature_cols
        print(f"Features: {len(feature_names)} (base only)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Models to benchmark
    models = [
        ("RandomForest (default)", RandomForestModel()),
        ("RandomForest (deep)", RandomForestModel(n_estimators=200, max_depth=20)),
        ("RandomForest (shallow)", RandomForestModel(n_estimators=50, max_depth=5)),
    ]

    # Add XGBoost if available
    if is_xgboost_available():
        from metaguard.models import XGBoostModel
        models.extend([
            ("XGBoost (default)", XGBoostModel()),
            ("XGBoost (tuned)", XGBoostModel(n_estimators=200, max_depth=8, learning_rate=0.05)),
        ])
    else:
        print("\nNote: XGBoost not available. Install with: pip install xgboost")

    # Add ensemble
    models.append(("Ensemble (3 RF)", create_default_ensemble()))

    # Update feature names for models
    for name, model in models:
        if args.derived_features:
            model.feature_names = feature_names

    # Run benchmarks
    all_results = []
    for name, model in models:
        try:
            results = benchmark_model(model, X_train, y_train, X_test, y_test, name)
            all_results.append(results)
        except Exception as e:
            print(f"\nError benchmarking {name}: {e}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary_df = pd.DataFrame(all_results)
    summary_cols = [
        "model", "val_accuracy", "val_auc_roc", "val_f1",
        "train_time", "single_inference_ms", "samples_per_second"
    ]
    available_cols = [c for c in summary_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))

    # Best model
    if all_results:
        best = max(all_results, key=lambda x: x.get("val_auc_roc", 0))
        print(f"\nBest Model: {best['model']} (AUC-ROC: {best.get('val_auc_roc', 0):.4f})")

    # Save results
    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
