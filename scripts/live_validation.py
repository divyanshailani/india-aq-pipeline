import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import date
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_CONFIG

def run_live_validation(conn=None):
    """
    Executes the Collision Engine logic.
    Merges today's actual PM2.5 values with past predictions for today.
    Inserts results into validation_ledger safely (idempotent).
    Returns the total number of validations in the ledger.
    """
    # Manage connection if not passed
    close_conn = False
    if conn is None:
        conn = psycopg2.connect(**DB_CONFIG)
        close_conn = True
        
    try:
        today = date.today()
        
        # Step A: Query today's actual PM2.5 values
        actuals_query = f"""
            SELECT s.country_code AS country, d.date AS target_date, d.value AS actual_pm25
            FROM daily_features d
            JOIN stations s ON d.station_id = s.id
            WHERE d.date = '{today}'
              AND d.parameter = 'pm25'
              AND d.value IS NOT NULL
        """
        df_actuals = pd.read_sql(actuals_query, conn)
        
        # We need a clean representation: groupby country and target_date to average across stations
        # Alternatively, the predictions might be at station level. Wait, the schema
        # the user requested for validation_ledger doesn't have station_id. It has country.
        # This implies we average across the country. Let's do that.
        if not df_actuals.empty:
            df_actuals = df_actuals.groupby(['country', 'target_date'])['actual_pm25'].mean().reset_index()
        
        # Step B: Query past predictions for today
        preds_query = f"""
            SELECT country_code AS country, run_date AS anchor_date, 
                   target_date, 'h' || horizon_days AS horizon, 
                   predicted_value AS predicted_pm25
            FROM prediction_log
            WHERE target_date = '{today}'
              AND predicted_value IS NOT NULL
        """
        df_preds = pd.read_sql(preds_query, conn)
        
        if not df_preds.empty:
            df_preds = df_preds.groupby(['country', 'anchor_date', 'target_date', 'horizon'])['predicted_pm25'].mean().reset_index()
            
        # Step C: Inner Join (Merge)
        if not df_actuals.empty and not df_preds.empty:
            df_collision = pd.merge(df_preds, df_actuals, on=['country', 'target_date'], how='inner')
            
            # Step D: Calculate math
            df_collision['error_delta'] = abs(df_collision['predicted_pm25'] - df_collision['actual_pm25'])
            
            # Step E: Insert into validation_ledger idempotently
            insert_query = """
                INSERT INTO validation_ledger 
                (country, anchor_date, target_date, horizon, predicted_pm25, actual_pm25, error_delta)
                VALUES %s
                ON CONFLICT (country, target_date, horizon) DO NOTHING
            """
            
            values = [
                (
                    row.country, 
                    row.anchor_date, 
                    row.target_date, 
                    row.horizon, 
                    float(row.predicted_pm25), 
                    float(row.actual_pm25), 
                    float(row.error_delta)
                ) 
                for row in df_collision.itertuples(index=False)
            ]
            
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, insert_query, values)
            conn.commit()
            print(f"  [Validation] Successfully merged {len(values)} country-level collision records.")
        else:
            print("  [Validation] No overlapping data to collide today.")

        # Finally, count total rows for Vercel Sync
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM validation_ledger")
            count = cur.fetchone()[0]
            
        return count
            
    finally:
        if close_conn:
            conn.close()

if __name__ == "__main__":
    count = run_live_validation()
    print(f"Total Validation Ledger Rows: {count}")
