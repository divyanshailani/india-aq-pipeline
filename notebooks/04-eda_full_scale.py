
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import matplotlib.dates as mdates

conn = psycopg2.connect(
    dbname = "indiaaq",
    user = "postgres",
    password = 8765,
    host = "localhost",
    port = 5432
)


sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)


# Top 20 Most Polluted Stations
query = """
SELECT s.name, ROUND(AVG(r.value)::numeric, 2) AS avg_pm25
FROM clean_measurements r 
JOIN stations s ON s.id = r.station_id
WHERE r.parameter = 'pm25'
GROUP BY s.name
ORDER BY avg_pm25 DESC
LIMIT 20;
"""

df_top = pd.read_sql(query, conn)

plt.figure(figsize=(12, 8))
plt.barh(df_top["name"], df_top["avg_pm25"], color="crimson")
plt.xlabel("Average PM2.5 (µg/m³)")
plt.title("Top 20 Most Polluted Stations in India (2021 - 2025) (PM2.5)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Monthly PM2.5 Trends ( All India Average)
query_monthly = """
SELECT DATE_TRUNC('month', datetime_utc) as month,
         ROUND(AVG(value)::numeric, 2) AS avg_pm25
FROM clean_measurements
WHERE parameter = 'pm25'
GROUP BY 1
ORDER BY 1;
"""

df_monthly = pd.read_sql(query_monthly, conn)
plt.plot(df_monthly["month"], df_monthly["avg_pm25"], marker = "o", color = "orangered")
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))  # Jan, Feb...
plt.xticks(rotation=0)
plt.xlabel("Month")
plt.ylabel("Average PM2.5 (µg/m³)")
plt.title("Monthly PM2.5 Trends in India (2021 - 2025)")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig("reports/monthly_pm25_trends.png", dpi = 150)
plt.show()

# parameter co-relation ( Daily values for a single station)
query_corr = """
SELECT DATE(datetime_utc) as date, parameter, AVG(value) as avg_value
FROM clean_measurements
WHERE station_id = 1
GROUP BY date, parameter;
"""

df_corr = pd.read_sql(query_corr, conn)
df_pivot = df_corr.pivot(index="date", columns="parameter", values="avg_value")
plt.figure(figsize=(10, 8))
sns.heatmap(df_pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation of Pollutants at Station 1 (2021 - 2025)")
plt.tight_layout()
plt.savefig("reports/pollutant_correlation.png", dpi = 150)
plt.show()

# station map of india
query_map = """
SELECT s.name, s.latitude, s.longitude, ROUND(AVG(r.value)::numeric, 2) AS avg_pm25
FROM clean_measurements r
JOIN stations s ON s.id = r.station_id
WHERE r.parameter = 'pm25'
AND s.latitude IS NOT NULL
GROUP BY s.name, s.latitude, s.longitude;
"""

df_map = pd.read_sql(query_map, conn)
plt.figure(figsize=(10, 12))
scatter = plt.scatter(df_map["longitude"], df_map["latitude"], c=df_map["avg_pm25"], cmap="RdYlGn_r", s=50, edgecolor="black", linewidth=0.5)
plt.colorbar(scatter, label="Average PM2.5 (µg/m³)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("India AQ Station Map — PM2.5 Severity (2021-2025)")
plt.tight_layout()
plt.savefig("reports/india_pm25_map.png", dpi = 150)
plt.show()

# PM2.5 Distribution (Histogram)

query_hist = """
SELECT value FROM clean_measurements
WHERE parameter = 'pm25' AND value < 500;
"""

df_hist = pd.read_sql(query_hist, conn)
plt.figure(figsize=(12, 6))
plt.hist(df_hist["value"], bins=100, color="orangered", alpha=0.8, edgecolor="black")
plt.axvline(x=40, color="green", linestyle="--", label="WHO Safe Limit (40)")
plt.axvline(x=60, color="orange", linestyle="--", label="India NAAQS (60)")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("Frequency")
plt.title("PM2.5 Distribution Across India (2021-2025)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/pm25_distribution.png", dpi=150)
plt.show()

# Winter vs Summer comparision

query_season = """
SELECT EXTRACT(MONTH FROM datetime_utc) as month_num,
       ROUND(AVG(value)::numeric, 2) as avg_pm25
FROM clean_measurements
WHERE parameter = 'pm25'
GROUP BY 1
ORDER BY 1;
"""
df_season = pd.read_sql(query_season, conn)
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df_season["month_name"] = [months[int(m)-1] for m in df_season["month_num"]]

plt.figure(figsize=(12, 6))
colors = ["#d32f2f" if m in [10,11,12,1,2] else "#4caf50" for m in df_season["month_num"]]
plt.bar(df_season["month_name"], df_season["avg_pm25"], color=colors)
plt.axhline(y=40, color="green", linestyle="--", label="WHO Limit")
plt.xlabel("Month")
plt.ylabel("Average PM2.5 (µg/m³)")
plt.title("Seasonal PM2.5 Pattern — Winter (Red) vs Summer (Green)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/seasonal_comparison.png", dpi=150)
plt.show()