import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
import onnx
import onnxruntime as ort
from onnxmltools.convert import convert_xgboost
from onnxconverter_common.data_types import FloatTensorType

# V12 strict feature list
V12_FEATURES = [
    'month', 'day_of_week', 'is_weekend', 'day_of_year', 
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30', 
    'roll_3_mean', 'roll_7_mean', 'roll_3_std', 'roll_14_mean', 
    'roll_30_mean', 'roll_14_std', 'om_temperature', 'om_wind_speed', 
    'om_precipitation', 'om_aerosol_optical_depth', 'rolling_3day_precip', 
    'aod_volatility_index', 'latitude', 'longitude'
]

def load_test_data():
    """Loads parquet data to verify ONNX predictions against XGBoost."""
    parquet_path = "data/daily_features_full.parquet"
    if not os.path.exists(parquet_path):
        print(f"Test data not found at {parquet_path}. Please fetch it.")
        return None
    
    df = pd.read_parquet(parquet_path)
    
    # We only need a sample (e.g., 5000 rows) with NaNs in AOD to test
    X = df[V12_FEATURES]
    return X.tail(5000)

def main():
    models_dir = "models/v12"
    json_models = glob.glob(f"{models_dir}/*/*/model.json")
    
    if not json_models:
        print("No XGBoost model.json files found in models/v12/")
        return

    print(f"Found {len(json_models)} XGBoost models to convert.")
    
    X_test = load_test_data()
    if X_test is None:
        print("Skipping NaN verification because data is unavailable.")

    for json_path in json_models:
        print(f"\nProcessing {json_path}...")
        
        # Load native XGBoost model
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(json_path)
        
        # FIX: onnxmltools requires feature names to be in 'f0', 'f1' format
        # rather than the actual pandas string names.
        booster = xgb_model.get_booster()
        booster.feature_names = [f'f{i}' for i in range(len(V12_FEATURES))]
        
        # FIX 2: onnxmltools crashes on XGBoost 2.0 boolean/categorical splits 
        # that omit 'split_condition'. We monkey-patch get_dump to inject it.
        original_get_dump = booster.get_dump
        def patched_get_dump(**kwargs):
            import json
            dumps = original_get_dump(**kwargs)
            patched_dumps = []
            for dump in dumps:
                tree = json.loads(dump)
                def patch_node(node):
                    if 'leaf' not in node and 'split_condition' not in node:
                        node['split_condition'] = 0.5
                    if 'children' in node:
                        for child in node['children']:
                            patch_node(child)
                patch_node(tree)
                patched_dumps.append(json.dumps(tree))
            return patched_dumps
        booster.get_dump = patched_get_dump
        
        # Convert to ONNX
        # Target opset 15 is generally safe and supports the latest NaN logic
        initial_types = [('X', FloatTensorType([None, len(V12_FEATURES)]))]
        onnx_model = convert_xgboost(
            xgb_model, 
            initial_types=initial_types, 
            target_opset=15
        )
        
        onnx_path = json_path.replace(".json", ".onnx")
        
        # Save ONNX
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        print(f"Saved {onnx_path}")
        
        # Verification Step
        if X_test is not None:
            # Create Inference Session
            onnx_session = ort.InferenceSession(onnx_path)
            
            # Verify specifically on rows where AOD is null - that's your 33% risk surface
            null_aod_mask = X_test['om_aerosol_optical_depth'].isna()
            num_nulls = null_aod_mask.sum()
            
            if num_nulls > 0:
                X_null = X_test[null_aod_mask]
                
                # XGBoost native predictions (handles NaN seamlessly)
                booster.feature_names = V12_FEATURES
                xgb_preds = xgb_model.predict(X_null)
                
                # ONNX predictions (requires float32 matrix)
                # Note: ONNX Runtime interprets np.nan natively if opset and converter are correct.
                X_null_np = X_null.values.astype(np.float32)
                onnx_preds = onnx_session.run(None, {'X': X_null_np})[0]
                
                # The crucial assertion
                try:
                    assert np.allclose(xgb_preds, onnx_preds.flatten(), atol=1e-3), "NaN handling divergence"
                    print(f"✅ NaN Verification Passed (Tested on {num_nulls} null AOD rows)")
                except AssertionError as e:
                    print(f"❌ ASSERTION FAILED: {e} on {json_path}")
                    # Print max diff to understand severity
                    max_diff = np.max(np.abs(xgb_preds - onnx_preds.flatten()))
                    print(f"   Max Difference: {max_diff}")
            else:
                print("⚠️ No NaN rows found in test sample to verify.")

if __name__ == "__main__":
    main()
