#!/usr/bin/env python
"""
Custom Configuration Example

Author: Moslem Mohseni

This example demonstrates how to configure MetaGuard.
"""

import os
from metaguard import SimpleDetector
from metaguard.utils.config import MetaGuardConfig, get_config, set_config, reset_config


def main():
    print("=" * 60)
    print("MetaGuard - Configuration Example")
    print("=" * 60)

    # Example 1: Default configuration
    print("\n1. Default Configuration")
    print("-" * 40)

    reset_config()  # Ensure we start fresh
    config = get_config()

    print(f"Model Path: {config.model_path}")
    print(f"Risk Threshold: {config.risk_threshold}")
    print(f"ML Weight: {config.ml_weight}")
    print(f"Log Level: {config.log_level}")
    print(f"Batch Size: {config.batch_size}")

    # Example 2: Custom configuration via class
    print("\n2. Custom Configuration")
    print("-" * 40)

    custom_config = MetaGuardConfig(
        risk_threshold=0.4,  # More sensitive (default 0.5)
        ml_weight=0.8,       # Higher ML weight (default 0.7)
        log_level="DEBUG",
    )

    print(f"Custom Risk Threshold: {custom_config.risk_threshold}")
    print(f"Custom ML Weight: {custom_config.ml_weight}")
    print(f"Custom Log Level: {custom_config.log_level}")

    # Example 3: Detector with custom config
    print("\n3. Detector with Custom Config")
    print("-" * 40)

    detector = SimpleDetector(config=custom_config)

    transaction = {
        "amount": 500,
        "hour": 12,
        "user_age_days": 50,
        "transaction_count": 15,
    }

    result = detector.detect(transaction)

    print(f"Transaction: {transaction}")
    print(f"Is Suspicious: {result['is_suspicious']}")
    print(f"Risk Score: {result['risk_score']:.2%}")
    print(f"Detector Threshold: {detector.config.risk_threshold}")

    # Example 4: Global configuration
    print("\n4. Global Configuration")
    print("-" * 40)

    # Set global config
    set_config(MetaGuardConfig(risk_threshold=0.3))
    print(f"Global threshold set to: 0.3")

    # New detectors will use global config
    detector1 = SimpleDetector()
    print(f"Detector1 threshold: {detector1.config.risk_threshold}")

    # Explicit config overrides global
    detector2 = SimpleDetector(config=MetaGuardConfig(risk_threshold=0.7))
    print(f"Detector2 threshold (explicit): {detector2.config.risk_threshold}")

    # Reset global config
    reset_config()
    detector3 = SimpleDetector()
    print(f"Detector3 threshold (after reset): {detector3.config.risk_threshold}")

    # Example 5: Environment variables
    print("\n5. Environment Variables")
    print("-" * 40)

    # Set environment variables (in real use, set these before running)
    os.environ["METAGUARD_RISK_THRESHOLD"] = "0.6"
    os.environ["METAGUARD_LOG_LEVEL"] = "WARNING"

    # Create new config (will read from env)
    reset_config()
    env_config = MetaGuardConfig()

    print(f"From METAGUARD_RISK_THRESHOLD: {env_config.risk_threshold}")
    print(f"From METAGUARD_LOG_LEVEL: {env_config.log_level}")

    # Clean up
    del os.environ["METAGUARD_RISK_THRESHOLD"]
    del os.environ["METAGUARD_LOG_LEVEL"]
    reset_config()

    # Example 6: Comparison with different thresholds
    print("\n6. Threshold Comparison")
    print("-" * 40)

    transaction = {
        "amount": 800,
        "hour": 10,
        "user_age_days": 40,
        "transaction_count": 12,
    }

    thresholds = [0.3, 0.5, 0.7]

    print(f"Transaction: amount=${transaction['amount']}, age={transaction['user_age_days']}d")
    print(f"\n{'Threshold':>10} {'Suspicious':>12} {'Risk Score':>12}")
    print("-" * 40)

    for threshold in thresholds:
        config = MetaGuardConfig(risk_threshold=threshold)
        detector = SimpleDetector(config=config)
        result = detector.detect(transaction)
        suspicious = "Yes" if result["is_suspicious"] else "No"
        print(f"{threshold:>10.1f} {suspicious:>12} {result['risk_score']:>11.2%}")

    print("\n" + "=" * 60)
    print("Configuration example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
