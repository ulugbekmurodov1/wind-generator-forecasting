"""2-qadam: modelni o'qitish va baholash."""
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, mean_absolute_error)
from common import diagnose, build_features


def main():
    df = pd.read_csv("data/weather.csv", parse_dates=["time"])
    df["diagnosis"] = df.apply(diagnose, axis=1)
    print("Diagnoz taqsimoti:\n", df["diagnosis"].value_counts(), "\n")

    X = build_features(df)
    y_cls = df["diagnosis"].shift(-1)            # ertangi ob-havo turi
    y_wind = df["wind_speed_10m_max"].shift(-1)  # ertangi maksimal shamol
    mask = X.notna().all(axis=1) & y_cls.notna() & y_wind.notna()
    X, y_cls, y_wind = X[mask], y_cls[mask], y_wind[mask]
    today_diag = df.loc[mask, "diagnosis"]

    split = int(len(X) * 0.8)   # vaqt bo'yicha bo'lish (kelajakni "ko'rmaslik" uchun)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 min_samples_leaf=3, random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_cls.iloc[:split])
    pred = clf.predict(Xte)

    base_acc = accuracy_score(y_cls.iloc[split:], today_diag.iloc[split:])
    print(f"Baseline (ertaga = bugungidek): {base_acc:.2%}")
    print(f"Model aniqligi:                 {accuracy_score(y_cls.iloc[split:], pred):.2%}\n")
    print(classification_report(y_cls.iloc[split:], pred, zero_division=0))

    reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                random_state=42, n_jobs=-1)
    reg.fit(Xtr, y_wind.iloc[:split])
    mae = mean_absolute_error(y_wind.iloc[split:], reg.predict(Xte))
    print(f"Shamol tezligi xatosi (MAE): {mae:.2f} m/s")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(clf, "models/clf.joblib")
    joblib.dump(reg, "models/wind_reg.joblib")
    ConfusionMatrixDisplay.from_predictions(y_cls.iloc[split:], pred, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Saqlandi: models/, results/confusion_matrix.png")


if __name__ == "__main__":
    main()
