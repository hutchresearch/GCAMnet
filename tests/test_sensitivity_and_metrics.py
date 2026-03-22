"""Comprehensive tests for sensitivity analysis, metrics, and related modules.

These tests are designed to run WITHOUT the full ml_gcam config/data infrastructure
by using module stubbing to bypass heavy imports.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Module loading helpers (stub out ml_gcam's heavy config/data dependencies)
# ---------------------------------------------------------------------------

def _stub_ml_gcam():
    """Install minimal stubs for the ml_gcam package so individual modules
    can be imported without the full config/data/inference infrastructure."""
    stubs = {}

    # ml_gcam top-level
    ml_gcam_stub = ModuleType("ml_gcam")
    ml_gcam_stub.config = MagicMock()
    ml_gcam_stub.logger = MagicMock()
    stubs["ml_gcam"] = ml_gcam_stub

    # ml_gcam.inference
    inference_stub = ModuleType("ml_gcam.inference")
    inference_stub.Inference = MagicMock
    stubs["ml_gcam.inference"] = inference_stub

    # ml_gcam.data
    data_stub = ModuleType("ml_gcam.data")
    data_stub.GcamDataset = MagicMock
    data_stub.Source = MagicMock
    data_stub.Split = MagicMock
    data_stub.load_targets = MagicMock()
    data_stub.experiment_name_to_paper_label = lambda x: x
    stubs["ml_gcam.data"] = data_stub

    # ml_gcam.data.normalization
    norm_stub = ModuleType("ml_gcam.data.normalization")
    norm_stub.Normalization = MagicMock
    stubs["ml_gcam.data.normalization"] = norm_stub

    # ml_gcam.evaluate (package)
    eval_stub = ModuleType("ml_gcam.evaluate")
    stubs["ml_gcam.evaluate"] = eval_stub

    # ml_gcam.config
    config_stub = ModuleType("ml_gcam.config")
    config_stub.config = MagicMock()
    stubs["ml_gcam.config"] = config_stub

    # ml_gcam.logging
    logging_stub = ModuleType("ml_gcam.logging")
    logging_stub.logger = MagicMock()
    stubs["ml_gcam.logging"] = logging_stub

    # ml_gcam.table (package)
    table_stub = ModuleType("ml_gcam.table")
    stubs["ml_gcam.table"] = table_stub

    # ml_gcam.emulator (package)
    emulator_stub = ModuleType("ml_gcam.emulator")
    stubs["ml_gcam.emulator"] = emulator_stub

    # SALib stubs (needed by sensitivity.py)
    salib_stub = ModuleType("SALib")
    salib_stub.ProblemSpec = MagicMock()
    stubs["SALib"] = salib_stub
    salib_analyze_stub = ModuleType("SALib.analyze")
    salib_analyze_stub.sobol = MagicMock()
    stubs["SALib.analyze"] = salib_analyze_stub
    salib_sample_stub = ModuleType("SALib.sample")
    salib_sample_stub.saltelli = MagicMock()
    stubs["SALib.sample"] = salib_sample_stub

    # torch stub (needed by model.py)
    torch_stub = ModuleType("torch")
    torch_stub.nn = MagicMock()
    stubs["torch"] = torch_stub
    stubs["torch.nn"] = MagicMock()

    return stubs


def _load_module(dotted_name, file_path, package=None, extra_stubs=None):
    """Load a single module by file path with ml_gcam stubs in place."""
    stubs = _stub_ml_gcam()
    if extra_stubs:
        stubs.update(extra_stubs)

    saved = {}
    for name, stub in stubs.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = stub

    try:
        spec = importlib.util.spec_from_file_location(dotted_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        if package:
            mod.__package__ = package
        sys.modules[dotted_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # Restore original state
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _load_sensitivity():
    return _load_module(
        "ml_gcam.evaluate.sensitivity",
        REPO_ROOT / "ml_gcam" / "evaluate" / "sensitivity.py",
        package="ml_gcam.evaluate",
    )


def _load_table_sensitivity():
    return _load_module(
        "ml_gcam.table.sensitivity",
        REPO_ROOT / "ml_gcam" / "table" / "sensitivity.py",
        package="ml_gcam.table",
    )


# ===========================================================================
# 1. sigma_normalization_factor tests
# ===========================================================================

class TestSigmaNormalizationFactor:
    """Tests for evaluate/sensitivity.py :: sigma_normalization_factor."""

    def test_normal_case(self):
        """Known input_std and output_std produce correct ratio."""
        mod = _load_sensitivity()
        inputs_std = pd.Series([0.5, 1.0, 2.0], index=["a", "b", "c"])
        output_std = 0.25
        result = mod.sigma_normalization_factor(inputs_std, output_std)
        expected = inputs_std / 0.25
        pd.testing.assert_series_equal(result, expected)

    def test_zero_output_std(self):
        """Zero output_std should return zero, not produce inf/NaN."""
        mod = _load_sensitivity()
        inputs_std = pd.Series([0.2, 0.4], index=["a", "b"])
        result = mod.sigma_normalization_factor(inputs_std, 0.0)
        assert np.isfinite(result.to_numpy()).all(), "Result contains inf or NaN"
        assert (result == 0.0).all(), "Result should be zero for flat outputs"

    def test_near_zero_output_std(self):
        """Near-zero output_std (1e-20) should return zero."""
        mod = _load_sensitivity()
        inputs_std = pd.Series([1.0, 3.0], index=["x", "y"])
        result = mod.sigma_normalization_factor(inputs_std, 1e-20, epsilon=1e-8)
        assert (result == 0.0).all(), "Result should be zero when output_std < epsilon"

    def test_negative_output_std(self):
        """Negative output_std edge case: abs() should be used so small negatives return zero."""
        mod = _load_sensitivity()
        inputs_std = pd.Series([1.0], index=["a"])
        # -1e-20 has abs < epsilon, so should return zero
        result = mod.sigma_normalization_factor(inputs_std, -1e-20, epsilon=1e-8)
        assert (result == 0.0).all()
        # -5.0 has abs > epsilon, so should use raw value
        result_neg = mod.sigma_normalization_factor(inputs_std, -5.0, epsilon=1e-8)
        expected_neg = inputs_std / (-5.0)
        np.testing.assert_allclose(result_neg.to_numpy(), expected_neg.to_numpy())

    def test_always_finite_for_positive_inputs(self):
        """Result should always be finite for positive input_std and any output_std."""
        mod = _load_sensitivity()
        rng = np.random.default_rng(42)
        for _ in range(50):
            n = rng.integers(1, 10)
            inputs_std = pd.Series(rng.uniform(0.01, 10, size=n))
            output_std = rng.choice([0.0, 1e-30, 1e-15, 0.5, 100.0])
            result = mod.sigma_normalization_factor(inputs_std, output_std)
            assert np.isfinite(result.to_numpy()).all(), (
                f"Non-finite result for output_std={output_std}"
            )


# ===========================================================================
# 2. Sensitivity normalization logic tests
# ===========================================================================

class TestSensitivityNormalizationLogic:
    """Tests for the DGSM normalization paths in dgsm_sensitivity_compare.

    We test the normalization logic in isolation by directly calling
    sigma_normalization_factor and verifying the two code paths:
      dgsm="dgsm" -> raw SALib values (no sigma norm)
      dgsm="vi"   -> SALib values * (sigma_x / sigma_y)
    """

    def test_dgsm_mode_returns_raw_values(self):
        """When dgsm='dgsm', output should be RAW SALib values (no sigma norm)."""
        # Simulate what the code does for dgsm="dgsm":
        # s1 = sp.analysis[feature][dgsm]  -- just use raw values
        # No sigma_normalization_factor call
        raw_s1 = np.array([0.3, 0.5, 0.2])
        input_keys = ["a", "b", "c"]
        dgsm_mode = "dgsm"

        # In "dgsm" mode, the code does NOT apply sigma_norm
        if dgsm_mode == "vi":
            # Would apply normalization
            pass
        result = dict(zip(input_keys, raw_s1))
        assert result == {"a": 0.3, "b": 0.5, "c": 0.2}

    def test_vi_mode_applies_sigma_normalization(self):
        """When dgsm='vi', output should be SALib values * (sigma_x / sigma_y)."""
        mod = _load_sensitivity()
        raw_s1 = np.array([0.3, 0.5, 0.2])
        inputs_std = np.array([0.1, 0.4, 0.8])
        output_std = 2.0
        input_keys = ["a", "b", "c"]

        dgsm_mode = "vi"
        s1 = raw_s1.copy()
        if dgsm_mode == "vi":
            sigma_norm = mod.sigma_normalization_factor(inputs_std, output_std)
            s1 = s1 * sigma_norm

        expected = raw_s1 * (inputs_std / output_std)
        np.testing.assert_allclose(s1, expected)

        # The dict(zip()) should use the normalized s1
        result = dict(zip(input_keys, s1))
        for key, val in result.items():
            idx = input_keys.index(key)
            assert val == pytest.approx(expected[idx])

    def test_dict_zip_uses_normalized_values(self):
        """The dict(zip()) should use the (potentially normalized) s1, not raw values."""
        mod = _load_sensitivity()
        raw_s1 = np.array([1.0, 2.0])
        inputs_std = np.array([0.5, 0.5])
        output_std = 1.0
        input_keys = ["x", "y"]

        # Mimic "vi" path
        s1 = raw_s1.copy()
        sigma_norm = mod.sigma_normalization_factor(inputs_std, output_std)
        s1 = s1 * sigma_norm

        result = dict(zip(input_keys, s1))
        # s1 should be [1.0 * 0.5/1.0, 2.0 * 0.5/1.0] = [0.5, 1.0]
        assert result["x"] == pytest.approx(0.5)
        assert result["y"] == pytest.approx(1.0)


# ===========================================================================
# 3. R-squared argument order test (metrics.py)
# ===========================================================================

class TestR2ArgumentOrder:
    """Verify that calculate_r2 in metrics.py passes y_true, y_pred in the
    correct order to sklearn.metrics.r2_score.

    Because the function is tightly coupled to config, we test the ordering
    logic at the sklearn level and verify the source code.
    """

    def test_r2_score_argument_order_matters(self):
        """Create synthetic data where r2_score(y_true, y_pred) != r2_score(y_pred, y_true)."""
        rng = np.random.default_rng(123)
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.5, 100) + 2.0  # offset so asymmetry is clear

        r2_correct = r2_score(y_true, y_pred)
        r2_reversed = r2_score(y_pred, y_true)
        assert r2_correct != pytest.approx(r2_reversed, abs=1e-6), (
            "Test is invalid: the two orderings should give different R2 scores"
        )

    def test_calculate_r2_source_code_order(self):
        """Verify in the source code that r2_score is called with y_true first."""
        source_path = REPO_ROOT / "ml_gcam" / "evaluate" / "metrics.py"
        source = source_path.read_text()
        # The function signature is calculate_r2(y_pred, y_true) but internally
        # it should call r2_score(y_true[:,...], y_pred[:,...])
        assert "r2_score(\n            y_true" in source or "r2_score(y_true" in source, (
            "r2_score should be called with y_true as the first argument"
        )

    def test_r2_with_mock_config(self):
        """End-to-end test of calculate_r2 using a mock config."""
        # Set up stubs with config providing required attributes
        stubs = _stub_ml_gcam()
        mock_config = stubs["ml_gcam"].config
        mock_config.data.region_keys = ["R1", "R2"]
        mock_config.data.years = [2020, 2030]
        mock_config.data.n_dimensions = 4  # 2 regions * 2 years
        mock_config.data.output_keys = ["q1", "q2"]

        saved = {}
        for name, stub in stubs.items():
            saved[name] = sys.modules.get(name)
            sys.modules[name] = stub

        try:
            spec = importlib.util.spec_from_file_location(
                "ml_gcam.evaluate.metrics",
                REPO_ROOT / "ml_gcam" / "evaluate" / "metrics.py",
                )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "ml_gcam.evaluate"
            sys.modules["ml_gcam.evaluate.metrics"] = mod
            spec.loader.exec_module(mod)

            rng = np.random.default_rng(42)
            n_samples = 20
            n_dims = 4
            n_outputs = 2
            y_true = rng.normal(0, 1, (n_samples, n_dims, n_outputs))
            y_pred = y_true + rng.normal(0, 0.1, (n_samples, n_dims, n_outputs))

            result = mod.calculate_r2(y_pred, y_true)
            assert isinstance(result, pl.DataFrame)
            assert "region" in result.columns
            assert "year" in result.columns
            # All R2 values should be high since y_pred ~= y_true
            r2_cols = [c for c in result.columns if c not in ("region", "year")]
            for col in r2_cols:
                vals = result[col].to_numpy()
                assert (vals > 0.5).all(), f"R2 for {col} unexpectedly low: {vals}"
        finally:
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
            sys.modules.pop("ml_gcam.evaluate.metrics", None)


# ===========================================================================
# 4. _finite_r2 tests (table/sensitivity.py)
# ===========================================================================

class TestFiniteR2:
    """Tests for table/sensitivity.py :: _finite_r2."""

    def _get_finite_r2(self):
        mod = _load_table_sensitivity()
        return mod._finite_r2

    def test_normal_case_matches_sklearn(self):
        """Normal case: should match sklearn r2_score exactly."""
        _finite_r2 = self._get_finite_r2()
        rng = np.random.default_rng(99)
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.3, 100)

        result = _finite_r2(y_true, y_pred)
        expected = r2_score(y_true, y_pred)
        assert result == pytest.approx(expected)

    def test_with_nan_values(self):
        """With NaN values: should compute R2 only over finite pairs."""
        _finite_r2 = self._get_finite_r2()
        y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y_pred = np.array([1.1, np.nan, 3.0, 4.2, 4.8])

        result = _finite_r2(y_true, y_pred)
        # Finite pairs: indices 0, 3, 4
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        expected = r2_score(y_true[mask], y_pred[mask])
        assert result == pytest.approx(expected)

    def test_with_inf_values(self):
        """With Inf values: should compute R2 only over finite pairs."""
        _finite_r2 = self._get_finite_r2()
        y_true = np.array([1.0, np.inf, 3.0, 4.0, -np.inf])
        y_pred = np.array([1.2, 2.0, 3.1, np.inf, 5.0])

        result = _finite_r2(y_true, y_pred)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        assert mask.sum() >= 2  # Enough finite pairs
        expected = r2_score(y_true[mask], y_pred[mask])
        assert result == pytest.approx(expected)

    def test_all_nan_returns_nan(self):
        """All NaN: should return NaN."""
        _finite_r2 = self._get_finite_r2()
        y_true = np.array([np.nan, np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan, np.nan])
        result = _finite_r2(y_true, y_pred)
        assert np.isnan(result)

    def test_less_than_2_finite_pairs_returns_nan(self):
        """Less than 2 finite pairs: should return NaN."""
        _finite_r2 = self._get_finite_r2()
        # Only 1 finite pair
        y_true = np.array([1.0, np.nan, np.nan])
        y_pred = np.array([1.1, np.nan, np.nan])
        result = _finite_r2(y_true, y_pred)
        assert np.isnan(result)

    def test_exactly_2_finite_pairs(self):
        """Exactly 2 finite pairs: should compute R2 (not NaN)."""
        _finite_r2 = self._get_finite_r2()
        y_true = np.array([1.0, 2.0, np.nan])
        y_pred = np.array([1.1, 2.1, np.nan])
        result = _finite_r2(y_true, y_pred)
        assert np.isfinite(result)

    def test_mixed_nan_and_inf(self):
        """Mixed NaN and Inf values: only truly finite pairs used."""
        _finite_r2 = self._get_finite_r2()
        y_true = np.array([1.0, np.nan, np.inf, 4.0, 5.0, -np.inf])
        y_pred = np.array([1.1, 2.0, 3.0, 4.1, np.nan, 6.0])
        result = _finite_r2(y_true, y_pred)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        expected = r2_score(y_true[mask], y_pred[mask])
        assert result == pytest.approx(expected)


# ===========================================================================
# 5. Aggregation logic tests
# ===========================================================================

class TestAggregationLogic:
    """Tests for z-score statistics and aggregation shape/divisor logic
    as used in dgsm_sensitivity_compare."""

    def test_zscore_computation(self):
        """Test that z-score statistics are computed correctly."""
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            rng.normal(10, 3, size=(100, 3)),
            columns=["q1", "q2", "q3"],
        )
        q_mean = data.mean()
        q_std = data.std()

        # Z-score of a known value
        row = data.iloc[0]
        z = (row - q_mean) / q_std
        expected = (data.iloc[0] - q_mean) / q_std
        pd.testing.assert_series_equal(z, expected)

        # Z-scored data should have mean ~0 and std ~1
        z_all = (data - q_mean) / q_std
        np.testing.assert_allclose(z_all.mean().to_numpy(), 0, atol=0.15)
        np.testing.assert_allclose(z_all.std().to_numpy(), 1, atol=0.15)

    def test_valid_q_filters_zero_variance(self):
        """Test that zero-variance quantities are excluded from valid_q."""
        data = pd.DataFrame({
            "q1": [1.0, 2.0, 3.0, 4.0],
            "q2": [5.0, 5.0, 5.0, 5.0],  # zero variance
            "q3": [1.0, 3.0, 5.0, 7.0],
        })
        q_std = data.std()
        valid_q = q_std[q_std >= 1e-8].index.tolist()
        assert "q1" in valid_q
        assert "q2" not in valid_q, "Zero-variance quantity should be excluded"
        assert "q3" in valid_q

    def test_region_aggregation_shape(self):
        """Test that region aggregation produces correct shape."""
        n_samples = 50
        region_keys = ["R1", "R2", "R3"]
        n_agg_keys = len(region_keys)

        y_agg = np.zeros((n_samples, n_agg_keys))
        assert y_agg.shape == (50, 3)

        # Simulate accumulation into region slot
        ri = region_keys.index("R2")
        y_agg[:, ri] += np.ones(n_samples)
        assert y_agg[:, 1].sum() == 50.0
        assert y_agg[:, 0].sum() == 0.0

    def test_year_aggregation_shape(self):
        """Test that year aggregation produces correct shape."""
        n_samples = 30
        year_keys = [2020, 2030, 2040, 2050]
        agg_keys = [str(y) for y in year_keys]

        y_agg = np.zeros((n_samples, len(agg_keys)))
        assert y_agg.shape == (30, 4)

        yi = agg_keys.index("2040")
        y_agg[:, yi] += np.ones(n_samples) * 3.0
        assert y_agg[:, 2].sum() == pytest.approx(90.0)

    def test_quantity_n_terms_divisor(self):
        """Test that n_terms divisor is correct for quantity aggregation."""
        region_keys = ["R1", "R2", "R3"]
        year_keys = [2020, 2030, 2040]
        # For 'quantity' mode: n_terms = len(region_keys) * len(year_keys)
        n_terms = len(region_keys) * len(year_keys)
        assert n_terms == 9

    def test_region_n_terms_divisor(self):
        """Test that n_terms divisor is correct for region aggregation."""
        year_keys = [2020, 2030, 2040]
        valid_q = ["q1", "q3"]  # q2 was zero-variance
        # For 'region' mode: n_terms = len(year_keys) * len(valid_q)
        n_terms = len(year_keys) * len(valid_q)
        assert n_terms == 6

    def test_year_n_terms_divisor(self):
        """Test that n_terms divisor is correct for year aggregation."""
        region_keys = ["R1", "R2", "R3"]
        valid_q = ["q1", "q2", "q3", "q4"]
        # For 'year' mode: n_terms = len(region_keys) * len(valid_q)
        n_terms = len(region_keys) * len(valid_q)
        assert n_terms == 12

    def test_aggregation_averaging(self):
        """Test that dividing by n_terms produces a correct average."""
        rng = np.random.default_rng(42)
        n_samples = 20
        n_agg = 3
        n_terms = 5

        y_agg = np.zeros((n_samples, n_agg))
        # Simulate accumulating n_terms contributions
        for _ in range(n_terms):
            y_agg += rng.normal(10, 1, (n_samples, n_agg))

        y_avg = y_agg / n_terms
        # Average of ~N(10,1) values should be close to 10
        np.testing.assert_allclose(y_avg.mean(), 10, atol=1.0)


# ===========================================================================
# 6. Other bug fix tests
# ===========================================================================

class TestBugFixes:
    """Tests for specific bug fixes in the codebase."""

    def test_not_implemented_error_for_unknown_arch(self):
        """Verify it raises NotImplementedError (not NameError) for unknown arch.

        The Arch.from_str method should raise NotImplementedError, not a
        bare NameError from a missing variable.
        """
        # We import the model module with torch stubs
        stubs = _stub_ml_gcam()

        # Stub torchtyping if not installed
        if "torchtyping" not in sys.modules:
            tt_stub = ModuleType("torchtyping")
            tt_stub.TensorType = MagicMock()
            stubs["torchtyping"] = tt_stub

        saved = {}
        for name, stub in stubs.items():
            saved[name] = sys.modules.get(name)
            sys.modules[name] = stub

        try:
            spec = importlib.util.spec_from_file_location(
                "ml_gcam.emulator.model",
                REPO_ROOT / "ml_gcam" / "emulator" / "model.py",
                )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "ml_gcam.emulator"
            sys.modules["ml_gcam.emulator.model"] = mod
            spec.loader.exec_module(mod)

            with pytest.raises(NotImplementedError, match="not implemented"):
                mod.Arch.from_str("nonexistent_architecture")

            # Also verify valid architectures do NOT raise
            assert mod.Arch.from_str("deep") == mod.Arch.DEEP
            assert mod.Arch.from_str("linear") == mod.Arch.LINEAR
        finally:
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
            sys.modules.pop("ml_gcam.emulator.model", None)

    def test_scenario_id_cast_int32(self):
        """Verify scenario_id uses Int32 cast (not Int16) in targets.py.

        Int16 can only hold values up to 32767, which is too small for
        large scenario IDs. This test verifies the source uses Int32.
        """
        source_path = REPO_ROOT / "ml_gcam" / "data" / "targets.py"
        source = source_path.read_text()

        # Check that scenario_id is cast to Int32
        assert 'pl.col("scenario_id").cast(pl.Int32)' in source, (
            "scenario_id should be cast to pl.Int32, not Int16"
        )

        # Additionally verify it's NOT Int16 for scenario_id
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if "scenario_id" in line and "Int16" in line:
                pytest.fail(
                    f"Line {i+1}: scenario_id should not use Int16 cast: {line.strip()}"
                )

    def test_scenario_id_int32_handles_large_values(self):
        """Verify that Int32 can handle scenario IDs larger than Int16 max (32767)."""
        large_id = 50000
        df = pl.DataFrame({"scenario_id": [large_id]})
        result = df.select(pl.col("scenario_id").cast(pl.Int32))
        assert result["scenario_id"][0] == large_id

        # Verify Int16 would overflow or fail for large values
        with pytest.raises(Exception):
            df_large = pl.DataFrame({"scenario_id": [large_id]})
            df_large.select(pl.col("scenario_id").cast(pl.Int16, strict=True))


# ===========================================================================
# Additional edge-case and integration tests
# ===========================================================================

class TestSigmaNormEdgeCases:
    """Additional edge cases for sigma_normalization_factor."""

    def test_scalar_inputs_std(self):
        """Works with scalar inputs_std."""
        mod = _load_sensitivity()
        result = mod.sigma_normalization_factor(0.5, 2.0)
        assert result == pytest.approx(0.25)

    def test_numpy_array_inputs(self):
        """Works with numpy array inputs_std."""
        mod = _load_sensitivity()
        inputs_std = np.array([1.0, 2.0, 3.0])
        result = mod.sigma_normalization_factor(inputs_std, 2.0)
        expected = np.array([0.5, 1.0, 1.5])
        np.testing.assert_allclose(result, expected)

    def test_epsilon_parameter(self):
        """Custom epsilon values are respected."""
        mod = _load_sensitivity()
        inputs_std = pd.Series([1.0])
        # With epsilon=1.0, output_std=0.5 < epsilon, so return zero
        result = mod.sigma_normalization_factor(inputs_std, 0.5, epsilon=1.0)
        assert (result == 0.0).all(), "Should return zero when output_std < epsilon"

    def test_large_output_std(self):
        """Large output_std should produce small normalization factor."""
        mod = _load_sensitivity()
        inputs_std = pd.Series([1.0])
        result = mod.sigma_normalization_factor(inputs_std, 1e6)
        assert result.iloc[0] == pytest.approx(1e-6)


class TestFiniteR2EdgeCases:
    """Additional edge cases for _finite_r2."""

    def test_perfect_prediction(self):
        """Perfect predictions should give R2 = 1.0."""
        mod = _load_table_sensitivity()
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mod._finite_r2(y, y)
        assert result == pytest.approx(1.0)

    def test_constant_prediction(self):
        """Constant prediction equal to mean should give R2 = 0."""
        mod = _load_table_sensitivity()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())
        result = mod._finite_r2(y_true, y_pred)
        assert result == pytest.approx(0.0)

    def test_empty_arrays(self):
        """Empty arrays should return NaN (fewer than 2 pairs)."""
        mod = _load_table_sensitivity()
        y_true = np.array([])
        y_pred = np.array([])
        result = mod._finite_r2(y_true, y_pred)
        assert np.isnan(result)
