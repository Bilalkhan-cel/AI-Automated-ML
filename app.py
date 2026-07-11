from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import os
import uuid
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

app = Flask(__name__)
app.secret_key = "your-secret-key"

UPLOAD_FOLDER = "temp_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_REGISTRY = {
    "regression": {
        "linear_regression": {"class": LinearRegression, "params": {}},
        "random_forest": {"class": RandomForestRegressor, "params": {"n_estimators": 100, "max_depth": None}},
        "decision_tree": {"class": DecisionTreeRegressor, "params": {"max_depth": None}},
        "svr": {"class": SVR, "params": {"C": 1.0, "kernel": "rbf"}},
        "knn": {"class": KNeighborsRegressor, "params": {"n_neighbors": 5}},
    },
    "classification": {
        "logistic_regression": {"class": LogisticRegression, "params": {"C": 1.0, "max_iter": 100}},
        "random_forest": {"class": RandomForestClassifier, "params": {"n_estimators": 100, "max_depth": None}},
        "decision_tree": {"class": DecisionTreeClassifier, "params": {"max_depth": None}},
        "svc": {"class": SVC, "params": {"C": 1.0, "kernel": "rbf"}},
        "knn": {"class": KNeighborsClassifier, "params": {"n_neighbors": 5}},
    }
}


@app.route("/")
def index():
    return render_template("index.html")

import math

def sanitize_records(records):
    clean = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                clean_row[k] = None
            else:
                clean_row[k] = v
        clean.append(clean_row)
    return clean


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    # POST - actual file upload logic
    file = request.files["file"]
    ext = file.filename.split(".")[-1].lower()

    session_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}.pkl")

    if ext == "csv":
        try:
            df = pd.read_csv(file, encoding="utf-8", sep=None, engine="python")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1", sep=None, engine="python")
    else:
        return jsonify({"error": "Unsupported file type. Please upload a CSV."}), 400

    df.to_pickle(filepath)  # keep real NaNs in the pickle for actual training
    session["session_id"] = session_id

    raw_preview = df.head(5).to_dict(orient="records")
    clean_preview = sanitize_records(raw_preview)

    return jsonify({
        "columns": df.columns.tolist(),
        "preview": clean_preview
    })


@app.route("/set_task", methods=["POST"])
def set_task():
    task_type = request.json.get("task_type")
    session["task_type"] = task_type
    return jsonify({"status": "ok"})


@app.route("/set_target", methods=["POST"])
def set_target():
    target = request.json.get("target")
    session["target"] = target

    session_id = session["session_id"]
    df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, f"{session_id}.pkl"))
    remaining_cols = [c for c in df.columns if c != target]

    return jsonify({"available_features": remaining_cols})


@app.route("/set_features", methods=["POST"])
def set_features():
    mode = request.json.get("mode")
    session_id = session["session_id"]
    df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, f"{session_id}.pkl"))

    if mode == "all":
        features = [c for c in df.columns if c != session["target"]]
    else:
        features = request.json.get("selected_features")

    session["features"] = features
    return jsonify({"status": "ok", "features": features})


@app.route("/get_models", methods=["GET"])
def get_models():
    task_type = session["task_type"]
    models = MODEL_REGISTRY[task_type]
    response = {name: info["params"] for name, info in models.items()}
    return jsonify(response)

@app.route("/train", methods=["POST"])
def train_model():
    data = request.json
    model_name = data.get("model_name")
    hyperparameters = data.get("hyperparameters", {})

    
    print("Model selected:", model_name)
    print("Hyperparameters:", hyperparameters)

    return jsonify({"status": "received"})

@app.route("/training")
def training():
    return render_template("training.html")


if __name__ == "__main__":
    app.run(debug=True)