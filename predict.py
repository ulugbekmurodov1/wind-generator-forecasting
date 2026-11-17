"""3-qadam: bugunni diagnoz qilish va ertangi kunni bashorat qilish."""
import joblib
import pandas as pd
import requests
from common import (LAT, LON, TZ, CITY, DAILY_VARS, diagnose, build_features)

URL = "https://api.open-meteo.com/v1/forecast"

# Shamol generatori uchun taxminiy hisob
ROTOR_DIAMETER_M = 1.0   # pervanek diametri (o'zingiznikiga almashtiring)
CP = 0.30                # quvvat koeffitsienti (real: 0.2-0.4)
AIR_DENSITY = 1.2        # kg/m3
AVG_FACTOR = 0.5         # kunlik maksimum -> o'rtacha shamol (taxminiy)
CUT_IN_MS = 3.0          # generator ishga tushadigan minimal tezlik


def main():
    params = {"latitude": LAT, "longitude": LON, "daily": ",".join(DAILY_VARS),
              "past_days": 7, "forecast_days": 1, "timezone": TZ,
              "wind_speed_unit": "ms"}
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"])

    X = build_features(df).iloc[[-1]]
    today = df.iloc[-1]
    clf = joblib.load("models/clf.joblib")
    reg = joblib.load("models/wind_reg.joblib")

    print(f"=== {CITY}: {today['time']} ===")
    print("Bugungi diagnoz:", diagnose(today))
    proba = clf.predict_proba(X)[0]
    print("\nErtangi kun ehtimolliklari:")
    for c, p in sorted(zip(clf.classes_, proba), key=lambda t: -t[1]):
        print(f"  {c:7s} {p:6.1%}")

    wind = float(reg.predict(X)[0])
    v = wind * AVG_FACTOR
    area = 3.14159 * (ROTOR_DIAMETER_M / 2) ** 2
    power = 0.5 * AIR_DENSITY * area * CP * v ** 3 if v >= CUT_IN_MS else 0.0
    print(f"\nErtangi maksimal shamol: {wind:.1f} m/s")
    print(f"Taxminiy o'rtacha shamol: {v:.1f} m/s -> generator quvvati ~{power:.1f} W")


if __name__ == "__main__":
    main()
