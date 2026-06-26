import datetime
from src.api_fallback_manager import ApiFallbackManager

# Open-Meteo Air Quality API
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

def fetch_aod_for_date(fallback_manager: ApiFallbackManager, lat: float, lon: float, target_date: str):
    """
    Fetches the daily Aerosol Optical Depth (AOD) for a specific latitude, longitude, and date.
    Returns a dictionary of the extracted features or raises RuntimeError if it fails.
    """
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_date,
        "end_date": target_date,
        "hourly": ["aerosol_optical_depth"],
        "timezone": "auto"
    }

    # This will throw a RuntimeError if it exhausts all backoff retries
    data = fallback_manager.request_with_fallback(
        url=OPEN_METEO_AQ_URL,
        params=params,
        is_openaq=False
    )
    
    hourly = data.get("hourly", {})
    aod_array = hourly.get("aerosol_optical_depth", [])
    
    if not aod_array:
        raise ValueError(f"Open-Meteo AOD returned empty array for {lat},{lon} on {target_date}")
    
    # Filter out None values
    valid_aod = [v for v in aod_array if v is not None]
    
    if not valid_aod:
        # If all hours are missing AOD, return 0.0 or None (let the caller decide, we'll return 0.0 for safety)
        mean_aod = 0.0
    else:
        mean_aod = sum(valid_aod) / len(valid_aod)
        
    return {
        "om_aerosol_optical_depth": mean_aod
    }
