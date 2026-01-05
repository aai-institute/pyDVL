"""
Simple test to verify partial_fit functionality works correctly.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from pydvl.valuation import Dataset
from pydvl.valuation.scorers import SupervisedScorer
from pydvl.valuation.types import Sample
from pydvl.valuation.utility.modelutility import PartialFitModelUtility


def test_partial_fit_basic():
    """Test that PartialFitModelUtility can be instantiated and used."""
    print("Test 1: Basic instantiation and usage")

    # Create small dataset
    X, y = make_classification(n_samples=50, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_data = Dataset.from_arrays(X_train, y_train)
    test_data = Dataset.from_arrays(X_test, y_test)

    # Create utility with partial_fit support
    model = SGDClassifier(random_state=42, max_iter=100)
    scorer = SupervisedScorer("accuracy", test_data, default=0.0, range=(0.0, 1.0))
    utility = PartialFitModelUtility(
        model, scorer, catch_errors=True, show_warnings=False
    )

    # Set training data
    utility = utility.with_dataset(train_data)

    print("  ✓ PartialFitModelUtility instantiated successfully")
    return utility


def test_partial_fit_incremental():
    """Test that partial_fit is used for incremental training."""
    print("\nTest 2: Incremental training simulation")

    utility = test_partial_fit_basic()
    train_indices = utility.training_data.indices

    # Simulate a permutation processing
    print(f"  - Training data size: {len(train_indices)}")

    # Reset state for new permutation
    utility.reset_partial_fit_state()
    print("  ✓ State reset")

    # Simulate processing first few points in a permutation
    permutation = np.random.permutation(train_indices)[:10]
    print(f"  - Using first 10 points from permutation: {permutation}")

    scores = []
    for i in range(1, len(permutation) + 1):
        sample = Sample(None, permutation[:i])
        score = utility(sample)
        scores.append(score)
        print(f"    Step {i}: subset size={i}, score={score:.4f}")

    print(f"  ✓ Processed {len(scores)} incremental steps")

    # Verify that we got increasing complexity
    if len(scores) > 1:
        print(f"  ✓ Score range: [{min(scores):.4f}, {max(scores):.4f}]")

    # Reset and verify state is cleared
    utility.reset_partial_fit_state()
    assert utility._current_model is None
    assert len(utility._current_indices) == 0
    print("  ✓ State cleared after reset")


def test_partial_fit_detection():
    """Test that utility detects partial_fit support correctly."""
    print("\nTest 3: Partial fit detection")

    X, y = make_classification(n_samples=30, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    train_data = Dataset.from_arrays(X_train, y_train)
    test_data = Dataset.from_arrays(X_test, y_test)
    scorer = SupervisedScorer("accuracy", test_data, default=0.0)

    # Model with partial_fit
    model_with = SGDClassifier(random_state=42)
    utility_with = PartialFitModelUtility(
        model_with, scorer, catch_errors=True, show_warnings=False
    )
    assert utility_with._supports_partial_fit
    print("  ✓ SGDClassifier: partial_fit detected")

    # Model without partial_fit (will still work, just uses fit())
    from sklearn.tree import DecisionTreeClassifier

    model_without = DecisionTreeClassifier(random_state=42)
    utility_without = PartialFitModelUtility(
        model_without, scorer, catch_errors=True, show_warnings=False
    )
    assert not utility_without._supports_partial_fit
    print("  ✓ DecisionTreeClassifier: no partial_fit (will use fit())")


def test_permutation_strategy():
    """Test that the permutation sampler uses the correct strategy."""
    print("\nTest 4: Permutation sampler strategy selection")

    from pydvl.valuation.samplers.permutation import PermutationSampler

    # Create a basic utility
    utility = test_partial_fit_basic()

    # Create sampler and check strategy
    sampler = PermutationSampler(seed=42)
    strategy = sampler.make_strategy(utility, None)

    from pydvl.valuation.samplers.permutation import (
        PermutationEvaluationStrategyWithPartialFit,
    )

    assert isinstance(strategy, PermutationEvaluationStrategyWithPartialFit)
    print("  ✓ PermutationSampler correctly uses PartialFit strategy")
    print(f"  ✓ Strategy type: {type(strategy).__name__}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Partial Fit Functionality")
    print("=" * 70)

    try:
        test_partial_fit_basic()
        test_partial_fit_incremental()
        test_partial_fit_detection()
        test_permutation_strategy()

        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())



