"""Umumiy sozlamalar va yordamchi funksiyalar."""
import numpy as np
import pandas as pd

# ---- SOZLAMALAR (o'zingizning joyingizga moslang) ----
CITY = "Toshkent"
LAT, LON = 41.31, 69.24
TZ = "Asia/Tashkent"

WINDY_MS = 8.0        # kuniga maksimal shamol shu qiymatdan oshsa -> "windy" (m/s)
RAIN_MM = 1.0         # yog'in shu qiymatdan oshsa -> "rain" (mm)
SNOW_CM = 0.1         # qor shu qiymatdan oshsa -> "snow" (sm)
CLOUDY_PCT = 40       # bulutlilik shundan oshsa -> "cloudy"

DAILY_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "snowfall_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "pressure_msl_mean", "relative_humidity_2m_mean",
    "cloud_cover_mean", "shortwave_radiation_sum",
]

BASE = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "pressure_msl_mean", "relative_humidity_2m_mean",
    "cloud_cover_mean", "shortwave_radiation_sum",
]
LAGGED = ["pressure_msl_mean", "wind_speed_10m_max",
          "temperature_2m_mean", "relative_humidity_2m_mean"]


def diagnose(row):
    """Bugungi ob-havoni qoida bo'yicha aniqlash (DIAGNOSIS)."""
    if row["snowfall_sum"] >= SNOW_CM:
        return "snow"
    if row["precipitation_sum"] >= RAIN_MM:
        return "rain"
    if row["wind_speed_10m_max"] >= WINDY_MS:
        return "windy"
    if row["cloud_cover_mean"] >= CLOUDY_PCT:
        return "cloudy"
    return "sunny"


def build_features(df):
    """Bugungi va oldingi kunlardan belgilar (features) yasash."""
    X = df[BASE].copy()
    for lag in (1, 2, 3):
        for c in LAGGED:
            X[f"{c}_lag{lag}"] = df[c].shift(lag)
    X["pressure_change"] = df["pressure_msl_mean"].diff()
    X["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    doy = pd.to_datetime(df["time"]).dt.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return X
