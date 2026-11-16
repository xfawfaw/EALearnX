"""Unit tests for 3-state forward-only GaussianHMM regime classifier."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from python.src.ml_regime import CANONICAL_REGIMES, RegimeModel


def _synthetic_regime_matrix(n_per_regime: int = 120) -> np.ndarray:
    """Generate synthetic [log_ret_1, atr_pct_14, adx_14] matrix with 3 clear regimes."""
    rng = np.random.default_rng(7)
    # Ranging: low return noise, low ATR, low ADX
    ranging = np.column_stack([
        rng.normal(0.0, 0.0005, n_per_regime),
        rng.normal(0.0010, 0.0002, n_per_regime),
        rng.normal(14.0, 2.0, n_per_regime),
    ])
    # Trending: positive drift, moderate ATR, high ADX
    trending = np.column_stack([
        rng.normal(0.0012, 0.0007, n_per_regime),
        rng.normal(0.0020, 0.0003, n_per_regime),
        rng.normal(34.0, 3.0, n_per_regime),
    ])
    # Volatile: wide return swings, high ATR, moderate ADX
    volatile = np.column_stack([
        rng.normal(0.0, 0.0035, n_per_regime),
        rng.normal(0.0065, 0.0008, n_per_regime),
        rng.normal(22.0, 4.0, n_per_regime),
    ])
    return np.vstack([ranging, trending, volatile])


def test_regime_fit_maps_states_semantically():
    """Verify HMM states map deterministically to VOLATILE (max ATR), TRENDING (higher ADX), RANGING."""
    X = _synthetic_regime_matrix()
    model = RegimeModel.fit(X)
    probs, labels = model.predict_proba_causal(X)

    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert labels[60] == "RANGING"
    assert labels[180] == "TRENDING"
    assert labels[300] == "VOLATILE"

    # Canonical columns: 0=TRENDING, 1=RANGING, 2=VOLATILE
    assert probs[60, 1] > 0.8
    assert probs[180, 0] > 0.8
    assert probs[300, 2] > 0.8


def test_forward_only_filtering_has_no_future_leakage():
    """Causal forward recursion must guarantee historical probabilities are strictly invariant to future bars."""
    X = _synthetic_regime_matrix()
    model = RegimeModel.fit(X[:240])

    probs_full, _ = model.predict_proba_causal(X)
    probs_prefix, _ = model.predict_proba_causal(X[:180])

    np.testing.assert_allclose(probs_full[:180], probs_prefix, rtol=1e-9, atol=1e-9)


def test_predict_latest():
    """Verify predict_latest returns the final regime label and formatted probability dict."""
    X = _synthetic_regime_matrix()
    model = RegimeModel.fit(X)

    label, prob_dict = model.predict_latest(X)
    probs, labels = model.predict_proba_causal(X)

    assert label == labels[-1]
    assert set(prob_dict.keys()) == set(CANONICAL_REGIMES)
    assert pytest.approx(sum(prob_dict.values()), abs=1e-6) == 1.0
    assert pytest.approx(prob_dict["VOLATILE"], abs=1e-6) == float(probs[-1, 2])


def test_constant_degenerate_column_handled_gracefully():
    """Verify near-constant input columns do not crash fitting due to singular covariance."""
    X = _synthetic_regime_matrix()
    # Force column 0 (returns) to be completely flat (zero variance)
    X[:, 0] = 0.0

    model = RegimeModel.fit(X, random_state=42)
    probs, labels = model.predict_proba_causal(X)

    assert probs.shape == (len(X), 3)
    assert not np.isnan(probs).any()
    assert len(labels) == len(X)


def test_input_validation():
    """Verify invalid shapes or empty inputs are rejected cleanly."""
    X = _synthetic_regime_matrix()
    model = RegimeModel.fit(X)

    # Empty prediction array returns empty outputs
    empty_probs, empty_labels = model.predict_proba_causal(np.empty((0, 3)))
    assert empty_probs.shape == (0, 3)
    assert empty_labels == []

    # Wrong number of columns
    with pytest.raises(ValueError, match="Expected X_regime with 3 columns"):
        model.predict_proba_causal(np.zeros((10, 2)))

    # Empty array for predict_latest raises ValueError
    with pytest.raises(ValueError, match="empty input"):
        model.predict_latest(np.empty((0, 3)))


def test_model_serialization_pickle():
    """Verify RegimeModel can be pickled and unpickled with exact identical predictions."""
    X = _synthetic_regime_matrix()
    model = RegimeModel.fit(X)
    probs_orig, labels_orig = model.predict_proba_causal(X)

    data = pickle.dumps(model)
    restored: RegimeModel = pickle.loads(data)

    probs_restored, labels_restored = restored.predict_proba_causal(X)
    np.testing.assert_allclose(probs_orig, probs_restored, atol=1e-9)
    assert labels_orig == labels_restored
