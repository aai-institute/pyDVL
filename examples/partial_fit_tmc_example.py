"""
Example demonstrating partial_fit functionality with TMC Shapley valuation.

This example shows how to use PartialFitModelUtility to speed up TMC Shapley
computations by using incremental training instead of retraining from scratch
for each subset.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from pydvl.valuation import Dataset, MinUpdates
from pydvl.valuation.methods.shapley import TMCShapleyValuation
from pydvl.valuation.scorers import SupervisedScorer
from pydvl.valuation.utility.modelutility import (
    ModelUtility,
    PartialFitModelUtility,
)


def main():
    """Run example comparing ModelUtility vs PartialFitModelUtility."""
    print("=" * 80)
    print("Partial Fit TMC Shapley Example")
    print("=" * 80)

    # Create a synthetic classification dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_data = Dataset.from_arrays(X_train, y_train)
    test_data = Dataset.from_arrays(X_test, y_test)

    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Features: {X_train.shape[1]}")

    # Setup models
    print("\n2. Setting up models...")
    # Use SGDClassifier which supports partial_fit
    model_standard = SGDClassifier(
        loss="log_loss", random_state=42, max_iter=100, tol=1e-3
    )
    model_partial = SGDClassifier(
        loss="log_loss", random_state=42, max_iter=100, tol=1e-3
    )

    # Create scorers
    scorer = SupervisedScorer("accuracy", test_data, default=0.0, range=(0.0, 1.0))

    # Create utilities
    print("\n3. Creating utilities...")
    print("   a) Standard ModelUtility (retrains from scratch each time)")
    utility_standard = ModelUtility(
        model_standard, scorer, catch_errors=True, show_warnings=False
    )

    print("   b) PartialFitModelUtility (uses incremental training)")
    utility_partial = PartialFitModelUtility(
        model_partial, scorer, catch_errors=True, show_warnings=False
    )

    # Run TMC Shapley with both utilities
    print("\n4. Running TMC Shapley valuation...")
    n_updates = 20  # Small number for demonstration

    print(f"\n   Running with standard utility ({n_updates} updates)...")
    valuation_standard = TMCShapleyValuation(
        utility_standard,
        is_done=MinUpdates(n_updates),
        show_warnings=False,
        progress=False,
    )
    valuation_standard.fit(train_data)
    result_standard = valuation_standard.result

    print(f"   Running with partial_fit utility ({n_updates} updates)...")
    valuation_partial = TMCShapleyValuation(
        utility_partial,
        is_done=MinUpdates(n_updates),
        show_warnings=False,
        progress=False,
    )
    valuation_partial.fit(train_data)
    result_partial = valuation_partial.result

    # Compare results
    print("\n5. Results comparison:")
    print("   " + "-" * 60)
    print(f"   {'Method':<30} {'Mean Value':<15} {'Std Value':<15}")
    print("   " + "-" * 60)
    print(
        f"   {'Standard ModelUtility':<30} "
        f"{np.mean(result_standard.values):<15.6f} "
        f"{np.std(result_standard.values):<15.6f}"
    )
    print(
        f"   {'PartialFitModelUtility':<30} "
        f"{np.mean(result_partial.values):<15.6f} "
        f"{np.std(result_partial.values):<15.6f}"
    )
    print("   " + "-" * 60)

    # Check correlation between results
    correlation = np.corrcoef(result_standard.values, result_partial.values)[0, 1]
    print(f"\n   Correlation between results: {correlation:.4f}")

    if correlation > 0.95:
        print("   ✓ Results are highly correlated!")
    elif correlation > 0.80:
        print("   ~ Results show good correlation.")
    else:
        print("   ⚠ Results show some divergence (expected with few updates).")

    print("\n6. Top 5 most valuable data points (PartialFitModelUtility):")
    print("   " + "-" * 40)
    top_indices = np.argsort(result_partial.values)[-5:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank}. Index {idx}: value = {result_partial.values[idx]:.6f}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  • PartialFitModelUtility automatically uses partial_fit when available")
    print("  • It falls back to regular fit() for models without partial_fit")
    print("  • Results should be similar to standard ModelUtility")
    print("  • Performance benefits increase with larger datasets and more updates")
    print("=" * 80)


if __name__ == "__main__":
    main()



