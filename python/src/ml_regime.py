"""3-State Forward-Only GaussianHMM Market Regime Classifier for XAUUSD.

Fits a 3-state Gaussian Hidden Markov Model on [log_ret_1, atr_pct_14, adx_14]
and maps states deterministically to semantic regimes:
- Highest fitted mean atr_pct_14 -> VOLATILE
- Among remaining 2 states, higher fitted mean adx_14 -> TRENDING, remaining -> RANGING

Computes regime probabilities strictly via causal log-space forward recursion (alpha pass)
with zero future lookahead (never backward smoothing).
"""

from __future__ import annotations

import logging
from typing import Sequence
import warnings

# Suppress upstream hmmlearn DeprecationWarning on numpy 2.x array shape mutation
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"hmmlearn.*")

from hmmlearn.hmm import GaussianHMM
import numpy as np
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

from python.src.schemas import RegimeLabel

logger = logging.getLogger(__name__)

CANONICAL_REGIMES: tuple[RegimeLabel, ...] = ("TRENDING", "RANGING", "VOLATILE")
REGIME_COLUMNS: tuple[str, ...] = ("log_ret_1", "atr_pct_14", "adx_14")
EPS: float = 1e-12


class RegimeModel:
    """3-State GaussianHMM regime classifier with strictly causal forward filtering."""

    def __init__(
        self,
        scaler: StandardScaler,
        hmm: GaussianHMM,
        state_indices: tuple[int, int, int],
        random_state: int = 42,
    ) -> None:
        """Initialize with fitted scaler, HMM, and canonical state mapping indices.

        Parameters
        ----------
        scaler : StandardScaler
            Fitted standard scaler for [log_ret_1, atr_pct_14, adx_14].
        hmm : GaussianHMM
            Fitted 3-state GaussianHMM.
        state_indices : tuple[int, int, int]
            HMM state indices mapping to (TRENDING, RANGING, VOLATILE).
        random_state : int
            Seed used for reproducible jitter/fitting.
        """
        self.scaler = scaler
        self.hmm = hmm
        self.state_indices = state_indices
        self.random_state = random_state

    @classmethod
    def fit(cls, X_regime: np.ndarray, random_state: int = 42) -> RegimeModel:
        """Fit StandardScaler and 3-state GaussianHMM on regime features.

        Parameters
        ----------
        X_regime : np.ndarray
            Feature matrix of shape (N, 3) corresponding to [log_ret_1, atr_pct_14, adx_14].
        random_state : int
            Random seed for HMM initialization and deterministic jitter fallback.

        Returns
        -------
        RegimeModel
            Fitted regime model with semantic state mapping.
        """
        X_arr = np.asarray(X_regime, dtype=float).copy()
        if X_arr.ndim != 2 or X_arr.shape[1] != 3:
            raise ValueError(f"Expected X_regime with 3 columns [log_ret_1, atr_pct_14, adx_14], got shape {X_arr.shape}")
        if len(X_arr) < 3:
            raise ValueError(f"Expected at least 3 samples to fit RegimeModel, got {len(X_arr)}")

        # Handle NaNs
        if np.isnan(X_arr).any():
            X_arr = np.nan_to_num(X_arr, nan=0.0)

        # Handle near-constant/degenerate columns safely (deterministic jitter)
        for col_idx in range(3):
            col_std = float(np.std(X_arr[:, col_idx]))
            if col_std < 1e-9:
                jitter_rng = np.random.default_rng(random_state + col_idx)
                X_arr[:, col_idx] += 1e-8 * jitter_rng.normal(0.0, 1.0, size=len(X_arr))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        hmm = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=random_state,
        )
        hmm.fit(X_scaled)

        # Map HMM states semantically back in unscaled domain
        unscaled_means = scaler.inverse_transform(hmm.means_)

        # 1. Highest fitted mean atr_pct_14 -> VOLATILE
        volatile_idx = int(np.argmax(unscaled_means[:, 1]))

        # 2. Among remaining 2 states, higher fitted mean adx_14 -> TRENDING, remaining -> RANGING
        remaining = [s for s in range(3) if s != volatile_idx]
        if unscaled_means[remaining[0], 2] >= unscaled_means[remaining[1], 2]:
            trending_idx = remaining[0]
            ranging_idx = remaining[1]
        else:
            trending_idx = remaining[1]
            ranging_idx = remaining[0]

        state_indices = (trending_idx, ranging_idx, volatile_idx)
        return cls(scaler=scaler, hmm=hmm, state_indices=state_indices, random_state=random_state)

    def predict_proba_causal(self, X_regime: np.ndarray) -> tuple[np.ndarray, list[RegimeLabel]]:
        """Compute regime probabilities using strictly causal forward recursion (alpha pass).

        Guarantees zero future lookahead: probabilities at time t depend solely on
        observations up to time t, completely invariant to any future rows appended.

        Parameters
        ----------
        X_regime : np.ndarray
            Feature matrix of shape (N, 3) corresponding to [log_ret_1, atr_pct_14, adx_14].

        Returns
        -------
        tuple[np.ndarray, list[RegimeLabel]]
            - probs: np.ndarray of shape (N, 3) in canonical order [TRENDING, RANGING, VOLATILE],
              with row sums strictly equal to 1.0.
            - labels: list[RegimeLabel] with argmax regime label per row.
        """
        X_arr = np.asarray(X_regime, dtype=float)
        if X_arr.shape == (0, 3) or (X_arr.ndim == 2 and X_arr.shape[0] == 0 and X_arr.shape[1] == 3) or X_arr.size == 0:
            return np.empty((0, 3), dtype=float), []

        if X_arr.ndim != 2 or X_arr.shape[1] != 3:
            raise ValueError(f"Expected X_regime with 3 columns, got shape {X_arr.shape}")

        if np.isnan(X_arr).any():
            X_arr = np.nan_to_num(X_arr, nan=0.0)

        X_scaled = self.scaler.transform(X_arr)
        log_emissions = self.hmm._compute_log_likelihood(X_scaled)
        n_samples, n_states = log_emissions.shape

        log_start = np.log(np.maximum(self.hmm.startprob_, EPS))
        log_trans = np.log(np.maximum(self.hmm.transmat_, EPS))

        log_alpha = np.empty((n_samples, n_states), dtype=float)
        # t = 0 (first bar)
        log_alpha[0] = log_start + log_emissions[0]
        log_alpha[0] -= logsumexp(log_alpha[0])

        # Forward recursive filtering
        for t in range(1, n_samples):
            for j in range(n_states):
                log_alpha[t, j] = log_emissions[t, j] + logsumexp(log_alpha[t - 1] + log_trans[:, j])
            log_alpha[t] -= logsumexp(log_alpha[t])

        raw_probs = np.exp(log_alpha)
        # Numerical normalization
        row_sums = raw_probs.sum(axis=1, keepdims=True)
        raw_probs = raw_probs / np.maximum(row_sums, EPS)

        # Reorder into canonical order [TRENDING, RANGING, VOLATILE]
        trending_idx, ranging_idx, volatile_idx = self.state_indices
        canonical_probs = np.column_stack([
            raw_probs[:, trending_idx],
            raw_probs[:, ranging_idx],
            raw_probs[:, volatile_idx],
        ])

        # Enforce exact row sums
        canonical_probs = canonical_probs / np.maximum(canonical_probs.sum(axis=1, keepdims=True), EPS)

        max_idx = np.argmax(canonical_probs, axis=1)
        labels: list[RegimeLabel] = [CANONICAL_REGIMES[int(i)] for i in max_idx]

        return canonical_probs, labels

    def predict_latest(self, X_regime: np.ndarray) -> tuple[RegimeLabel, dict[str, float]]:
        """Return the latest regime label and probability dictionary for the current bar.

        Parameters
        ----------
        X_regime : np.ndarray
            Feature matrix of shape (N, 3). Must contain at least one observation.

        Returns
        -------
        tuple[RegimeLabel, dict[str, float]]
            - latest_label: RegimeLabel ("TRENDING", "RANGING", or "VOLATILE")
            - prob_dict: dict mapping each canonical regime name to its float probability.
        """
        if len(X_regime) == 0:
            raise ValueError("Cannot predict latest regime on empty input array.")

        probs, labels = self.predict_proba_causal(X_regime)
        latest_label = labels[-1]
        latest_probs = probs[-1]

        prob_dict: dict[str, float] = {
            "TRENDING": float(latest_probs[0]),
            "RANGING": float(latest_probs[1]),
            "VOLATILE": float(latest_probs[2]),
        }
        return latest_label, prob_dict
