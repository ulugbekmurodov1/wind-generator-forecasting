"""1-qadam: Open-Meteo'dan tarixiy ob-havo ma'lumotini yuklash (API kalit kerak emas)."""
import datetime as dt
import os
import pandas as pd
import requests
from common import LAT, LON, TZ, DAILY_VARS, CITY

URL = "https://archive-api.open-meteo.com/v1/archive"


def main():
    end = dt.date.today() - dt.timedelta(days=6)   # arxiv ~5 kun kechikadi
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": "2005-01-01", "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": TZ, "wind_speed_unit": "ms",
    }
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/weather.csv", index=False)
    print(f"{CITY}: {len(df)} kunlik ma'lumot saqlandi -> data/weather.csv")


if __name__ == "__main__":
    main()
