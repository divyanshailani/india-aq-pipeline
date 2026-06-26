import datetime
from src.api_fallback_manager import ApiFallbackManager

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_weather_for_date(fallback_manager: ApiFallbackManager, lat: float, lon: float, target_date: str):
    """
    Fetches the daily weather for a specific latitude, longitude, and date.
    Returns a dictionary of the extracted features or raises RuntimeError if it fails.
    """
    target_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    today = datetime.date.today()
    
    # If the date is older than 90 days, we MUST use the archive API.
    # To be safe, we use the archive API for anything older than 7 days, 
    # since the forecast API is best for recent/future data.
    if (today - target_dt).days > 7:
        url = OPEN_METEO_ARCHIVE_URL
    else:
        url = OPEN_METEO_FORECAST_URL
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_date,
        "end_date": target_date,
        "daily": ["temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum", "relative_humidity_2m_mean"],
        "timezone": "auto"
    }

    # This will throw a RuntimeError if it exhausts all backoff retries
    data = fallback_manager.request_with_fallback(
        url=url,
        params=params,
        is_openaq=False
    )
    
    daily = data.get("daily", {})
    if not daily or "time" not in daily or len(daily["time"]) == 0:
        raise ValueError(f"Open-Meteo returned empty daily array for {lat},{lon} on {target_date}")
    
    return {
        "om_temperature": daily.get("temperature_2m_mean", [None])[0],
        "om_wind_speed": daily.get("wind_speed_10m_max", [None])[0],
        "om_precipitation": daily.get("precipitation_sum", [None])[0],
        "humidity": daily.get("relative_humidity_2m_mean", [None])[0]
    }
