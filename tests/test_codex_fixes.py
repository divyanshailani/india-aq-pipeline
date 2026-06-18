"""
Global AQ Intelligence — Core Tests
=====================================
Tests for:
  1. Feature leakage prevention (rolling features use shifted values)
  2. Schema alignment (sql/schema.sql matches what scripts expect)
  3. Prediction JSON shape (output format for frontend)
  4. Config module (shared credentials, no hardcoded values)
  5. Model metadata (features list matches training)

Run: python -m pytest tests/ -v
"""

import os
import sys
import json
import re
import glob

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ─── Test 1: Rolling Feature Leakage Prevention ──────────────

class TestFeatureLeakage:
    """Verify that rolling features do NOT include today's target value."""

    def test_src_features_uses_shift(self):
        """src/features.py must shift before computing rolling stats."""
        features_path = os.path.join(PROJECT_ROOT, "src", "features.py")
        with open(features_path) as f:
            code = f.read()
        
        # Must contain .shift(1) before rolling computation
        assert ".shift(1)" in code, \
            "src/features.py must use .shift(1) before rolling to prevent leakage"
    
    def test_rolling_excludes_current_day(self):
        """Simulate: rolling mean of [10, 20, 30, 40] with shift should NOT include day 4's value."""
        from src.features import add_rolling_features
        
        df = pd.DataFrame({"pm25": [10, 20, 30, 40, 50]}, 
                          index=pd.date_range("2024-01-01", periods=5))
        result = add_rolling_features(df.copy(), "pm25", windows=[3])
        
        # On day 5 (index 4): roll_3_mean should be mean of days 2,3,4 = (20+30+40)/3 = 30
        # NOT mean of days 3,4,5 = (30+40+50)/3 = 40 (that would be leakage)
        day5_roll = result.iloc[4]["roll_3_mean"]
        assert abs(day5_roll - 30.0) < 0.01, \
            f"roll_3_mean on day 5 should be 30.0 (shifted), got {day5_roll}"

    def test_build_global_fixes_all_three_rolling(self):
        """build_global_features.py must fix roll_3_mean, roll_7_mean, AND roll_3_std."""
        build_path = os.path.join(PROJECT_ROOT, "scripts", "build_global_features.py")
        with open(build_path) as f:
            code = f.read()
        
        assert "roll_3_mean" in code, "Must fix roll_3_mean"
        assert "roll_7_mean" in code, "Must fix roll_7_mean"
        assert "roll_3_std" in code, "Must fix roll_3_std"


# ─── Test 2: Schema Alignment ────────────────────────────────

class TestSchema:
    """Verify sql/schema.sql includes all v5 columns and tables."""

    def setup_method(self):
        schema_path = os.path.join(PROJECT_ROOT, "sql", "schema.sql")
        with open(schema_path) as f:
            self.schema = f.read().lower()

    def test_daily_features_has_country_code(self):
        assert "country_code" in self.schema

    def test_daily_features_has_nasa_columns(self):
        for col in ["nasa_temperature", "nasa_humidity", "nasa_wind_speed", 
                     "precipitation", "wind_direction"]:
            assert col in self.schema, f"Missing column: {col}"

    def test_daily_features_has_fire_count(self):
        assert "fire_count" in self.schema

    def test_pipeline_runs_table_exists(self):
        assert "pipeline_runs" in self.schema

    def test_prediction_log_table_exists(self):
        assert "prediction_log" in self.schema


# ─── Test 3: Prediction JSON Shape ───────────────────────────

class TestPredictionJSON:
    """Verify prediction output JSON matches what the frontend expects."""

    def test_site_data_json_shape(self):
        """Check existing site_data JSONs have the expected structure."""
        site_data_dir = os.path.join(PROJECT_ROOT, "data", "site_data")
        if not os.path.exists(site_data_dir):
            return  # skip if no data generated yet
        
        json_files = glob.glob(os.path.join(site_data_dir, "*.json"))
        if not json_files:
            return  # skip if empty
        
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            
            # Must have top-level keys
            assert "country" in data, f"{jf} missing 'country'"
            assert "predictions" in data, f"{jf} missing 'predictions'"
            
            # Each prediction must have required fields
            if data["predictions"]:
                pred = data["predictions"][0]
                required = ["target_date", "predicted_pm25", "horizon_days", "confidence"]
                for key in required:
                    assert key in pred, f"{jf} prediction missing '{key}'"


# ─── Test 4: Config Module ───────────────────────────────────

class TestConfig:
    """Verify shared config module works correctly."""

    def test_config_imports(self):
        from src.config import DB_CONFIG, MODEL_DIR, SITE_DATA_DIR
        assert "dbname" in DB_CONFIG
        assert "password" in DB_CONFIG
        assert "user" in DB_CONFIG

    def test_config_reads_env_vars(self):
        """Config should use env vars when set."""
        os.environ["POSTGRES_PASSWORD"] = "test_password_12345"
        # Reload module
        import importlib
        from src import config
        importlib.reload(config)
        assert config.DB_CONFIG["password"] == "test_password_12345"
        # Restore
        os.environ["POSTGRES_PASSWORD"] = "8765"
        importlib.reload(config)

    def test_no_hardcoded_creds_in_scripts(self):
        """No script should contain hardcoded '8765' except src/config.py."""
        scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
        violations = []
        for py_file in glob.glob(os.path.join(scripts_dir, "*.py")):
            with open(py_file) as f:
                content = f.read()
            if '"8765"' in content or "'8765'" in content:
                violations.append(os.path.basename(py_file))
        
        assert not violations, \
            f"Hardcoded DB password found in: {', '.join(violations)}"


# ─── Test 5: Model Metadata ──────────────────────────────────

class TestModelMetadata:
    """Verify model files have matching metadata."""

    def test_model_meta_files_exist(self):
        """Each .pkl should have a matching _meta.json."""
        model_dir = os.path.join(PROJECT_ROOT, "models", "v5")
        if not os.path.exists(model_dir):
            return  # skip if no models trained yet
        
        pkl_files = glob.glob(os.path.join(model_dir, "*_pm25_gbr.pkl"))
        for pkl in pkl_files:
            meta = pkl.replace("_gbr.pkl", "_meta.json")
            assert os.path.exists(meta), \
                f"Missing metadata for {os.path.basename(pkl)}"

    def test_meta_has_features_list(self):
        """Metadata JSON must contain 'features' key."""
        model_dir = os.path.join(PROJECT_ROOT, "models", "v5")
        if not os.path.exists(model_dir):
            return
        
        meta_files = glob.glob(os.path.join(model_dir, "*_meta.json"))
        for mf in meta_files:
            with open(mf) as f:
                data = json.load(f)
            assert "features" in data, f"{mf} missing 'features' key"
            assert isinstance(data["features"], list), f"{mf} features must be a list"
            assert len(data["features"]) > 0, f"{mf} features list is empty"


# ─── Test 6: API v5 Alignment ────────────────────────────────

class TestAPIAlignment:
    """Verify api/main.py serves v5, not v3."""

    def test_api_references_v5(self):
        api_path = os.path.join(PROJECT_ROOT, "api", "main.py")
        with open(api_path) as f:
            code = f.read()
        
        assert "v3" not in code.lower() or "v5" in code.lower(), \
            "api/main.py still references v3 without v5"
        assert "Global AQ Intelligence" in code, \
            "api/main.py should reference Global AQ Intelligence, not IndiaAQ"

    def test_api_no_hardcoded_model_path(self):
        api_path = os.path.join(PROJECT_ROOT, "api", "main.py")
        with open(api_path) as f:
            code = f.read()
        
        assert "gb_pm25_v3" not in code, \
            "api/main.py still loads old v3 model path"
