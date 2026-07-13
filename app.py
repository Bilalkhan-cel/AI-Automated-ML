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
from data_cleaning import Data_cleaning, sanitize_records
from model_registry import MODELS

app = Flask(__name__)
app.secret_key = "your-secret-key"

UPLOAD_FOLDER = "temp_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_REGISTRY = MODELS


@app.route("/")
def index():
    return render_template("index.html")


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

    return jsonify({"columns": df.columns.tolist(), "preview": clean_preview})


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

    df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, f"{session.get("session_id")}.pkl"))
    target = session.get("target")
    test_siz = data.get("test_size", 0.2)
    features = session.get("features")
    features = list(features)
    x_train, x_test, y_train, y_test, pre_processor = Data_cleaning(
        daf=df, target=target, feature=features, test_size=test_siz
    )
    
    os.makedirs("clean_data", exist_ok=True)

    

    file_path = os.path.join(
    "clean_data",
    f"{session['session_id']}.pkl"
)

    joblib.dump(
    {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": pre_processor
    },
    file_path
    )

    session["training_data"] = {
        "Task_Type": session.get("task_type"),
        "Target_Column": session.get("target"),
        "Features": session.get("features"),
        "Model_Name": data.get("model_name"),
        "Hyperparameters": data.get("hyperparameters", {}),
        "Test_Size": data.get("test_size", 0.2),
        "Session_id": session.get("session_id"),
        "X_Train_Shape": x_train.shape,
        "X_Test_Shape": x_test.shape
    }

    # session['training_data'].task

    return jsonify({"status": "ok"})


@app.route("/training")
def training():
    config = session.get("training_data", {})
    file_path = os.path.join(
    "clean_data",
    f"{session['session_id']}.pkl"
)

    data = joblib.load(file_path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    x_test = data["x_test"]
    preprocessor = data["preprocessor"]

    return render_template("training.html", config=config)


if __name__ == "__main__":
    app.run(debug=True)
