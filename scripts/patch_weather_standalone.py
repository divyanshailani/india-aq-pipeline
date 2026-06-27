import time
import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values

def patch_weather_batch():
    conn = psycopg2.connect(
        host='globalaqiserver.postgres.database.azure.com',
        user='postgresadmin',
        password='[REDACTED]',
        dbname='indiaaq',
        sslmode='require',
        connect_timeout=15
    )
    
    print("🔍 Detecting missing weather/AOD rows...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT df.station_id, s.latitude, s.longitude, MIN(df.date), MAX(df.date), COUNT(*)
            FROM daily_features df
            JOIN stations s ON df.station_id = s.id
            WHERE df.om_temperature IS NULL OR df.om_precipitation IS NULL
            GROUP BY df.station_id, s.latitude, s.longitude
            ORDER BY COUNT(*) DESC
        """)
        stations = cur.fetchall()
        
    if not stations:
        print("✅ No missing weather data found!")
        return

    total_rows = sum(s[5] for s in stations)
    print(f"🚨 Found {total_rows} missing rows across {len(stations)} stations.")
    print("🚀 Using BATCH FETCHING to avoid 429 Rate Limits...\n")

    for idx, (sid, lat, lon, min_date, max_date, count) in enumerate(stations):
        start_str = min_date.strftime("%Y-%m-%d")
        end_str = max_date.strftime("%Y-%m-%d")
        print(f"[{idx+1}/{len(stations)}] Station {sid}: Fetching {start_str} to {end_str} ({count} rows)")
        
        w_url = "https://archive-api.open-meteo.com/v1/archive"
        w_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str,
            "end_date": end_str,
            "daily": "temperature_2m_mean,precipitation_sum,windspeed_10m_max",
            "timezone": "auto"
        }
        
        a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        a_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "aerosol_optical_depth",
            "timezone": "auto"
        }
        
        try:
            w_res = requests.get(w_url, params=w_params, timeout=10).json()
            a_res = requests.get(a_url, params=a_params, timeout=10).json()
            
            w_daily = w_res.get("daily", {})
            a_hourly = a_res.get("hourly", {})
            
            if not w_daily or not a_hourly:
                continue
                
            w_df = pd.DataFrame(w_daily)
            w_df['date'] = pd.to_datetime(w_df['time']).dt.date
            
            a_df = pd.DataFrame(a_hourly)
            a_df['date'] = pd.to_datetime(a_df['time']).dt.date
            a_daily = a_df.groupby('date')['aerosol_optical_depth'].mean().reset_index()
            
            merged = pd.merge(w_df, a_daily, on='date', how='inner')
            
            update_data = []
            for _, row in merged.iterrows():
                update_data.append((
                    row["temperature_2m_mean"], row["windspeed_10m_max"], row["precipitation_sum"],
                    row["aerosol_optical_depth"], sid, row["date"]
                ))
            
            with conn.cursor() as cur:
                execute_values(cur, """
                    UPDATE daily_features
                    SET om_temperature = data.temp,
                        om_wind_speed = data.wind,
                        om_precipitation = data.precip,
                        om_aerosol_optical_depth = data.aod
                    FROM (VALUES %s) AS data(temp, wind, precip, aod, sid, dt)
                    WHERE station_id = data.sid AND date = data.dt
                """, update_data)
            conn.commit()
            print(f"  ✅ Updated {len(update_data)} rows for station {sid}.")
            
            time.sleep(1.0)
            
        except Exception as e:
            print(f"  ❌ Error fetching for station {sid}: {str(e)}")
            time.sleep(2.0)
            continue

    print("\n🎉 Bulk patch complete!")
    conn.close()

if __name__ == "__main__":
    patch_weather_batch()
