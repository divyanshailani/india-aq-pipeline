"""
Update daily_features table with NASA POWER weather data.
Replaces Open-Meteo's bad data with satellite-quality weather.
"""
import psycopg2
import pandas as pd

DB_CONFIG = {
    "dbname": "indiaaq", "user": "postgres",
    "password": "8765", "host": "localhost", "port": "5432"
}

def main():
    nasa = pd.read_csv("data/weather_nasa_power.csv")
    nasa['date'] = pd.to_datetime(nasa['date']).dt.date
    print(f"NASA rows: {len(nasa):,}")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    updated = 0
    total = len(nasa)

    for idx, row in nasa.iterrows():
        if pd.isna(row['nasa_temperature']):
            continue

        cur.execute("""
            UPDATE daily_features
            SET temperature = %s, humidity = %s, wind_speed = %s
            WHERE station_id = %s AND date = %s AND parameter = 'pm25'
        """, (
            row['nasa_temperature'],
            row['nasa_humidity'],
            row['nasa_wind_speed'],
            int(row['station_id']),
            row['date']
        ))
        updated += cur.rowcount

        if (idx + 1) % 50000 == 0:
            conn.commit()
            print(f"  [{idx+1:,}/{total:,}] {updated:,} rows updated...")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n✅ Done! {updated:,} rows updated with NASA weather.")

if __name__ == "__main__":
    main()
