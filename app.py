from datetime import datetime
from flask import Flask, request, jsonify
from monitoring import detect_drift, log_prediction_event

app = Flask(__name__)

SUPPORTED_COUNTRIES = {
    "united_kingdom": 12500.0,
    "united_states": 18400.0,
    "germany": 9200.0,
    "france": 8100.0,
    "netherlands": 6400.0,
}

REFERENCE_REVENUE = [11000, 11800, 12100, 12400, 12900, 13200, 12700, 13500]


def validate_payload(data: dict, require_country: bool = True) -> tuple[bool, str]:
    """Validate incoming JSON request fields."""
    if not isinstance(data, dict):
        return False, "Request body must be a valid JSON object."

    target_date = data.get("target_date")
    if not target_date or not isinstance(target_date, str):
        return False, "Missing or invalid 'target_date' (expected YYYY-MM-DD)."

    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        return False, "Invalid 'target_date' format. Use YYYY-MM-DD."

    if require_country:
        country = data.get("country")
        if not country or not isinstance(country, str):
            return False, "Missing or invalid 'country'."
        normalized = country.strip().lower().replace(" ", "_")
        if normalized not in SUPPORTED_COUNTRIES:
            return False, f"Unsupported country '{country}'."

    return True, ""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "aavail-revenue-forecasting"}), 200


@app.route("/predict", methods=["POST"])
def predict_country():
    """Predict revenue for a specific country and target date."""
    payload = request.get_json(silent=True)
    is_valid, error_msg = validate_payload(payload, require_country=True)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    country_key = payload["country"].strip().lower().replace(" ", "_")
    target_date = payload["target_date"]
    base_forecast = SUPPORTED_COUNTRIES[country_key]

    drift_status = detect_drift(REFERENCE_REVENUE, [base_forecast])
    log_prediction_event(
        country=country_key,
        target_date=target_date,
        prediction=base_forecast,
        drift_info=drift_status,
    )

    return jsonify(
        {
            "country": country_key,
            "target_date": target_date,
            "predicted_revenue": base_forecast,
            "drift_check": drift_status,
        }
    ), 200


@app.route("/predict_all", methods=["POST"])
def predict_all_countries():
    """Predict revenue across all countries combined for a target date."""
    payload = request.get_json(silent=True)
    is_valid, error_msg = validate_payload(payload, require_country=False)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    target_date = payload["target_date"]
    total_forecast = float(sum(SUPPORTED_COUNTRIES.values()))

    drift_status = detect_drift(REFERENCE_REVENUE, list(SUPPORTED_COUNTRIES.values()))
    log_prediction_event(
        country="all_countries",
        target_date=target_date,
        prediction=total_forecast,
        drift_info=drift_status,
    )

    return jsonify(
        {
            "country": "all_countries",
            "target_date": target_date,
            "predicted_revenue": total_forecast,
            "breakdown_by_country": SUPPORTED_COUNTRIES,
            "drift_check": drift_status,
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
