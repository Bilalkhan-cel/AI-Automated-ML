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
from model_traning import Train_model
from dotenv import load_dotenv
load_dotenv()
from visualizations import plot_confusion_matrix, plot_actual_vs_predicted, plot_residuals, plot_feature_importance

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_KEY") or "dev-secret-key-change-in-production"

UPLOAD_FOLDER = "temp_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_REGISTRY = MODELS

def validate_task_target(task_type, y):
    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            return "This target column has text/category values, which doesn't work well for Regression. You might want Classification instead."
        return None
    else:
        if pd.api.types.is_numeric_dtype(y):
            is_continuous = pd.api.types.is_float_dtype(y) and not (y.dropna() % 1 == 0).all()
            if is_continuous or y.nunique() > 20:
                return "This target column looks continuous (many unique or decimal values), which doesn't work well for Classification. You might want Regression instead."
        return None


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

    task_type = session.get("task_type")
    warning = validate_task_target(task_type, df[target])

    return jsonify({"available_features": remaining_cols, "warning": warning})


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
    dele=os.path.join(UPLOAD_FOLDER, f"{session.get("session_id")}.pkl") # deleting the temp data file 
    try:
      os.remove(dele)
    except Exception as e:
        return f"Erorr deleting temp file {e}"
    

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
    task = session["training_data"]["Task_Type"]
    Model_name = session["training_data"]["Model_Name"]

    model, metrics, y_pred = Train_model(
        model_name=Model_name, task_type=task,
        X_TRAIN=x_train, Y_TRAIN=y_train, X_TEST=x_test, Y_TEST=y_test
    )

    plots = {}
    if task == "classification":
        plots["confusion_matrix"] = plot_confusion_matrix(y_test, y_pred)
        plots["actual_vs_predicted"] = plot_actual_vs_predicted(y_test, y_pred)
    else:
        plots["actual_vs_predicted"] = plot_actual_vs_predicted(y_test, y_pred)
        plots["residuals"] = plot_residuals(y_test, y_pred)

    try:
        feature_names = preprocessor.get_feature_names_out()
        fi_plot = plot_feature_importance(model, feature_names)
        if fi_plot:
            plots["feature_importance"] = fi_plot
    except Exception:
        pass

    try:
        os.remove(file_path)
    except Exception as e:
        return f"Erorr deleting clean data  file {e}"

    return render_template("training.html", config=config, metrics=metrics, plots=plots)


if __name__ == "__main__":
    app.run(debug=True)
