"""Web app."""
import json
import pickle
from pathlib import Path

import flask
from flask import Flask, render_template, request
from geopy.geocoders import Nominatim
import pandas as pd

from training import prediction

app = flask.Flask(__name__)


PREDICT_CITIES = [
    "Delhi",
    "Mumbai",
    "Kolkata",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Ahmedabad",
    "Pune",
    "Lucknow",
    "Jaipur",
]

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Lucknow": (26.8467, 80.9462),
    "Jaipur": (26.9124, 75.7873),
}


def _city_options(selected: str = "Delhi"):
    return [{"name": c, "sel": "selected" if c == selected else ""} for c in PREDICT_CITIES]


DEFAULT_FEATURE_NAMES = ["temp", "max_temp", "wind_speed", "cloudcover", "precip", "humidity"]
HYDROLOGY_ARTIFACTS_DIR = Path("artifacts/hydrology")
CITY_BASIN_MAP_PATH = Path("hydrology/city_to_basin.json")
DEFAULT_CITY = "Delhi"


def _load_model_artifacts():
    threshold = 0.5
    horizon_days = 15
    feature_names = DEFAULT_FEATURE_NAMES

    bundle_path = Path("model_bundle.pkl")
    if bundle_path.exists():
        with bundle_path.open("rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and "model" in bundle:
            return (
                bundle["model"],
                float(bundle.get("threshold", threshold)),
                int(bundle.get("horizon_days", horizon_days)),
                bundle.get("feature_names", feature_names),
            )

    with open("model.pickle", "rb") as f:
        model_obj = pickle.load(f)

    if isinstance(model_obj, dict) and "model" in model_obj:
        return (
            model_obj["model"],
            float(model_obj.get("threshold", threshold)),
            int(model_obj.get("horizon_days", horizon_days)),
            model_obj.get("feature_names", feature_names),
        )

    return model_obj, threshold, horizon_days, feature_names


def _predict_flood_risk(model, features, threshold):
    # Probabilistic output is preferred; fallback to hard class when unavailable.
    if hasattr(model, "predict_proba"):
        risk_probability = float(model.predict_proba([features])[0][1])
    else:
        risk_probability = float(model.predict([features])[0])

    label = "Unsafe" if risk_probability >= threshold else "Safe"
    return label, risk_probability


def _load_city_basin_map():
    if not CITY_BASIN_MAP_PATH.exists():
        return {}
    try:
        return json.loads(CITY_BASIN_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


CITY_TO_BASIN = _load_city_basin_map()


def _resolve_hydrology_paths_for_city(cityname: str):
    basin_id = CITY_TO_BASIN.get(cityname)
    if not basin_id:
        return None, None, None, f"No basin mapping found for city '{cityname}' in {CITY_BASIN_MAP_PATH}."

    model_path = HYDROLOGY_ARTIFACTS_DIR / f"{basin_id}_hydrology_model_bundle.pkl"
    feature_row_path = HYDROLOGY_ARTIFACTS_DIR / f"{basin_id}_latest_feature_row.parquet"

    # Backward-compatible fallback for earlier single-basin runs.
    if basin_id == "jalandhar" and not feature_row_path.exists():
        feature_row_path = HYDROLOGY_ARTIFACTS_DIR / "latest_feature_row.parquet"

    if not model_path.exists():
        return (
            basin_id,
            model_path,
            feature_row_path,
            f"Missing hydrology model bundle: {model_path}",
        )
    if not feature_row_path.exists():
        return (
            basin_id,
            model_path,
            feature_row_path,
            f"Missing hydrology feature row: {feature_row_path}",
        )
    return basin_id, model_path, feature_row_path, None


def _load_hydrology_predictions(cityname: str):
    basin_id, model_path, feature_row_path, error = _resolve_hydrology_paths_for_city(cityname)
    if error:
        return {"city": cityname, "error": error}

    try:
        with model_path.open("rb") as f:
            artifact = pickle.load(f)

        feature_columns = artifact["feature_columns"]
        feature_df = pd.read_parquet(feature_row_path)
        if len(feature_df) != 1:
            return {
                "city": cityname,
                "basin_id": basin_id,
                "error": f"Hydrology feature row must contain exactly one row: {feature_row_path}",
            }

        missing = [c for c in feature_columns if c not in feature_df.columns]
        if missing:
            return {
                "city": cityname,
                "basin_id": basin_id,
                "error": f"Hydrology feature row is missing columns: {', '.join(missing)}",
            }

        x = feature_df[feature_columns]
        predictions = []
        for lead_hours in sorted(artifact["models_by_lead"].keys()):
            model = artifact["models_by_lead"][lead_hours]
            threshold = float(artifact["results_by_lead"][lead_hours]["test_metrics"]["threshold"])
            prob = float(model.predict_proba(x)[:, 1][0])
            predictions.append(
                {
                    "lead_hours": int(lead_hours),
                    "risk_probability": round(prob, 6),
                    "threshold": round(threshold, 6),
                    "alert": "UNSAFE" if prob >= threshold else "SAFE",
                }
            )
        return {"city": cityname, "basin_id": basin_id, "predictions": predictions}
    except Exception as exc:
        return {
            "city": cityname,
            "basin_id": basin_id,
            "error": f"Hydrology prediction unavailable: {exc}",
        }


model, decision_threshold, horizon_days, model_feature_names = _load_model_artifacts()
geolocator = Nominatim(user_agent="floodml-app")


@app.route("/")
@app.route("/index.html")
def index() -> str:
    """Base page."""
    return flask.render_template("index.html")


@app.route("/plots.html")
def plots():
    return render_template("plots.html")


@app.route("/heatmaps.html")
def heatmaps():
    return render_template("heatmaps.html")


@app.route("/predicts.html")
def predicts():
    selected_city = DEFAULT_CITY
    hydrology_data = _load_hydrology_predictions(selected_city)
    return render_template(
        "predicts.html",
        cities=_city_options(selected_city),
        cityname="Information about the city",
        pred=None,
        risk_pct=None,
        threshold_pct=round(decision_threshold * 100, 1),
        horizon_days=horizon_days,
        hydrology_data=hydrology_data,
    )


@app.route("/predicts.html", methods=["GET", "POST"])
def get_predicts():
    cityname = DEFAULT_CITY
    cities = _city_options(DEFAULT_CITY)
    try:
        cityname = request.form["city"]
        cities = _city_options(cityname)

        if cityname in CITY_COORDS:
            latitude, longitude = CITY_COORDS[cityname]
        else:
            location = geolocator.geocode(cityname)
            if not location:
                raise ValueError("Unable to geocode city: {}".format(cityname))
            latitude = location.latitude
            longitude = location.longitude
        final = prediction.get_data(latitude, longitude)

        if len(final) != len(model_feature_names):
            raise ValueError(
                "Model expects {} features but received {}".format(len(model_feature_names), len(final))
            )

        pred, risk_probability = _predict_flood_risk(model, final, decision_threshold)
        hydrology_data = _load_hydrology_predictions(cityname)

        return render_template(
            "predicts.html",
            cityname="Information about " + cityname,
            cities=cities,
            temp=round(final[0], 2),
            maxt=round(final[1], 2),
            wspd=round(final[2], 2),
            cloudcover=round(final[3], 2),
            percip=round(final[4], 2),
            humidity=round(final[5], 2),
            pred=pred,
            risk_pct=round(risk_probability * 100, 1),
            threshold_pct=round(decision_threshold * 100, 1),
            horizon_days=horizon_days,
            hydrology_data=hydrology_data,
        )
    except Exception:
        hydrology_data = _load_hydrology_predictions(DEFAULT_CITY)
        app.logger.exception("Prediction failed for city %s", cityname if "cityname" in locals() else "UNKNOWN")
        return render_template(
            "predicts.html",
            cities=cities,
            cityname="Oops, we weren't able to retrieve data for that city.",
            pred=None,
            risk_pct=None,
            threshold_pct=round(decision_threshold * 100, 1),
            horizon_days=horizon_days,
            hydrology_data=hydrology_data,
        )


if __name__ == "__main__":
    app.run(debug=True)
