"""
app.py - Flask web application for the Disease Prediction System.

Routes
------
/                  Serve the prediction UI.
/api/dataset-info  Return symptom & disease lists as JSON.
/predict           Accept symptoms (JSON) and return predicted disease.
/predict-nlp       Accept free-text, extract symptoms via NLP, and predict.
"""

from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

from src.data import symptoms_dict, diseases_list

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained ensemble model
# ---------------------------------------------------------------------------
MODEL_PATH = "models/ensemble_model_augmented.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as exc:
    model = None
    print(f"⚠️ Model not found. Run `python -m scripts.train` first. ({exc})")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main prediction page."""
    symptoms_list = sorted(s.title() for s in symptoms_dict.keys())
    return render_template("index.html", symptoms=symptoms_list)


@app.route("/api/dataset-info")
def dataset_info():
    """Return dataset coverage info for the information page."""
    symptoms_list = sorted(s.title() for s in symptoms_dict.keys())
    diseases_names = sorted(diseases_list.values())
    return jsonify({
        "total_symptoms": len(symptoms_list),
        "total_diseases": len(diseases_names),
        "symptoms": symptoms_list,
        "diseases": diseases_names,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Predict a disease from the submitted symptom list."""
    if not model:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    data = request.get_json()
    user_symptoms = data.get("symptoms", [])

    if not user_symptoms:
        return jsonify({"error": "Please select at least one symptom."}), 400

    # Build binary input vector
    user_symptoms_lower = [s.lower() for s in user_symptoms]
    input_vector = np.zeros(len(symptoms_dict))
    for s in user_symptoms_lower:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    pred_index = model.predict(np.array([input_vector]))[0]
    result = diseases_list.get(pred_index, "Unknown")

    return jsonify({
        "disease": result,
        "symptoms_count": len(user_symptoms),
    })


@app.route("/predict-nlp", methods=["POST"])
def predict_nlp():
    """Extract symptoms from free text via NLP, then predict disease.

    Expects JSON: {"text": "...", "backend": "transformer"|"llm"}
    Returns JSON:  {
        "disease": "...",
        "matched_symptoms": [...],
        "confidence": float,
        "backend": "transformer"|"llm",
        "symptoms_count": int
    }
    """
    if not model:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    data = request.get_json()
    text = data.get("text", "").strip()
    backend = data.get("backend", "transformer")

    if not text:
        return jsonify({"error": "Please describe your symptoms."}), 400

    # Create the chosen NLP extractor
    try:
        from src.nlp import create_extractor
        extractor = create_extractor(backend)
    except Exception as exc:
        return jsonify({"error": f"NLP backend error: {exc}"}), 500

    # Extract symptoms from free text
    symptom_names = list(symptoms_dict.keys())
    extraction = extractor.extract_symptoms(text, symptom_names)

    if extraction.get("error"):
        return jsonify({"error": extraction["error"]}), 500

    matched = extraction.get("matched_symptoms", [])
    if not matched:
        return jsonify({
            "error": "Could not identify any symptoms from your description. "
                     "Please try being more specific or use the symptom selector.",
            "extraction": extraction,
        }), 400

    # Build binary input vector from extracted symptoms
    input_vector = np.zeros(len(symptoms_dict))
    for s in matched:
        s_lower = s.lower()
        if s_lower in symptoms_dict:
            input_vector[symptoms_dict[s_lower]] = 1

    pred_index = model.predict(np.array([input_vector]))[0]
    result = diseases_list.get(pred_index, "Unknown")

    return jsonify({
        "disease": result,
        "matched_symptoms": [s.title() for s in matched],
        "confidence": extraction.get("confidence", 0.0),
        "details": extraction.get("details", []),
        "backend": extraction.get("backend", backend),
        "symptoms_count": len(matched),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
