"""
Ultra-fast NASA + Fire merge using pandas to_csv + COPY FROM.
"""
import os, io, psycopg2, pandas as pd

DB = dict(dbname="indiaaq", user="postgres", password="8765", host="localhost", port=5432)
ROOT = os.path.join(os.path.dirname(__file__), "..")

def copy_df_to_temp(cur, conn, df, table_name, columns, create_sql):
    """Bulk load a DataFrame into a temp table using COPY."""
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    cur.execute(create_sql)
    conn.commit()
    
    buf = io.StringIO()
    df[columns].to_csv(buf, sep="\t", header=False, index=False, na_rep="\\N")
    buf.seek(0)
    cur.copy_from(buf, table_name, columns=columns, null="\\N")
    conn.commit()
    print(f"  Loaded {len(df):,} rows into {table_name}")

def main():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    # Kill stuck connections
    cur.execute("""
        SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
        WHERE datname='indiaaq' AND pid != pg_backend_pid() 
        AND state != 'idle' AND query_start < now() - interval '1 minute'
    """)
    conn.commit()

    # Step 1: Add columns
    print("Step 1: Adding columns...")
    for col, dtype in [("country_code","TEXT"),("nasa_temperature","DOUBLE PRECISION"),
                        ("nasa_humidity","DOUBLE PRECISION"),("nasa_wind_speed","DOUBLE PRECISION"),
                        ("precipitation","DOUBLE PRECISION"),("wind_direction","DOUBLE PRECISION"),
                        ("fire_count","INTEGER")]:
        try:
            cur.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {dtype}")
            conn.commit()
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()
    print("  Done")

    # Step 2: Country code
    print("\nStep 2: Backfilling country_code...")
    cur.execute("""
        UPDATE daily_features df SET country_code = s.country_code
        FROM stations s WHERE df.station_id = s.id AND df.country_code IS NULL
    """)
    print(f"  Updated {cur.rowcount:,} rows")
    conn.commit()

    # Step 3: NASA weather
    weather_path = os.path.join(ROOT, "data", "weather_nasa_power.csv")
    extra_path = os.path.join(ROOT, "data", "weather_nasa_power_extra.csv")

    if os.path.exists(weather_path):
        print("\nStep 3: NASA weather merge...")
        w = pd.read_csv(weather_path)
        w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
        
        if os.path.exists(extra_path):
            e = pd.read_csv(extra_path)
            e["date"] = pd.to_datetime(e["date"]).dt.strftime("%Y-%m-%d")
            w = w.merge(e, on=["date", "station_id"], how="left")

        # Rename for temp table
        w = w.rename(columns={"nasa_temperature":"t","nasa_humidity":"h","nasa_wind_speed":"ws"})
        if "precipitation" not in w.columns:
            w["precipitation"] = None
        if "wind_direction" not in w.columns:
            w["wind_direction"] = None

        copy_df_to_temp(cur, conn, w, "tmp_w",
            ["station_id","date","t","h","ws","precipitation","wind_direction"],
            """CREATE TEMP TABLE tmp_w (
                station_id INT, date DATE, t FLOAT8, h FLOAT8, ws FLOAT8, 
                precipitation FLOAT8, wind_direction FLOAT8)""")

        print("  Bulk UPDATE join...")
        cur.execute("""
            UPDATE daily_features df
            SET nasa_temperature=t.t, nasa_humidity=t.h, nasa_wind_speed=t.ws,
                precipitation=t.precipitation, wind_direction=t.wind_direction
            FROM tmp_w t WHERE df.station_id=t.station_id AND df.date=t.date
            AND df.nasa_temperature IS NULL
        """)
        print(f"  Weather merged: {cur.rowcount:,} rows")
        conn.commit()

    # Step 4: Fire data
    fire_path = os.path.join(ROOT, "data", "fire_counts_firms.csv")
    if os.path.exists(fire_path):
        print("\nStep 4: FIRMS fire merge...")
        f = pd.read_csv(fire_path)
        f["date"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
        f = f.rename(columns={"fire_count":"fc"})

        copy_df_to_temp(cur, conn, f, "tmp_f",
            ["station_id","date","fc"],
            "CREATE TEMP TABLE tmp_f (station_id INT, date DATE, fc INT)")

        cur.execute("""
            UPDATE daily_features df SET fire_count=t.fc
            FROM tmp_f t WHERE df.station_id=t.station_id AND df.date=t.date
            AND df.fire_count IS NULL
        """)
        print(f"  Fire merged: {cur.rowcount:,} rows")
        conn.commit()

    # Step 5: Fix leakage
    print("\nStep 5: Fix roll_3_mean leakage...")
    cur.execute("""
        UPDATE daily_features SET roll_3_mean = 
            (COALESCE(lag_1,0)+COALESCE(lag_2,0)+COALESCE(lag_3,0)) /
            NULLIF((CASE WHEN lag_1 IS NOT NULL THEN 1 ELSE 0 END)+
                   (CASE WHEN lag_2 IS NOT NULL THEN 1 ELSE 0 END)+
                   (CASE WHEN lag_3 IS NOT NULL THEN 1 ELSE 0 END),0)
        WHERE lag_1 IS NOT NULL
    """)
    print(f"  Fixed {cur.rowcount:,} rows")
    conn.commit()

    # Final
    print("\n" + "=" * 50)
    cur.execute("""
        SELECT country_code, parameter, COUNT(*),
               SUM(CASE WHEN nasa_temperature IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN fire_count IS NOT NULL THEN 1 ELSE 0 END)
        FROM daily_features GROUP BY country_code, parameter
        ORDER BY country_code, parameter
    """)
    total = 0
    print(f"{'Country':>8} {'Param':>6} {'Rows':>10} {'Weather':>10} {'Fire':>8}")
    print("-" * 50)
    for r in cur.fetchall():
        total += r[2]
        print(f"{r[0] or '?':>8} {r[1]:>6} {r[2]:>10,} {r[3]:>10,} {r[4]:>8,}")
    print(f"{'TOTAL':>8} {'':>6} {total:>10,}")
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
