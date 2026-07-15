# 🤖 AutoML Studio

**Upload your data. Pick a target. Train a model. Download it. That's it.**

AutoML Studio is an open-source, end-to-end automated machine learning web app built with **Flask**. It takes raw, messy data (CSV, Pickle, and more), automatically cleans and preprocesses it, and lets you train and download production-ready ML models — no coding required.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/flask-web%20app-black)
![License](https://img.shields.io/badge/license-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange)

---

## 📖 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Supported Models](#-supported-models)
- [Getting Started](#-getting-started)
- [Usage Guide](#-usage-guide)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## ✨ Features

- 📂 **Flexible data upload** — supports CSV, Pickle (`.pkl`), and other common tabular formats
- 🧹 **Automated data cleaning** — handles missing values, duplicates, and inconsistent formatting
- 🔢 **Automatic encoding** — categorical variables are encoded automatically (label/one-hot)
- 🕳️ **Null imputation** — smart strategies for filling missing numeric and categorical data
- 🎯 **User-defined target & features** — choose your own target column and feature set
- 🧠 **Task selection** — supports both **Regression** and **Classification** problems
- ⚙️ **Multiple model choices** — pick from a wide range of pre-integrated ML algorithms
- 📊 **Training summary & metrics** — instant performance reports after training
- 📈 **Auto-generated visualizations** — understand your data and model results visually
- 💾 **Downloadable model bundle** — get a single `model.pkl` containing **both the trained model and the fitted preprocessing pipeline**, ready to use in your own projects
- 🌐 **No-code interface** — built with Flask so anyone can use it from the browser

---

## ⚙️ How It Works

```
 ┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
 │  Upload Data │ --> │  Auto Clean   │ --> │  Select Target  │ --> │  Choose Model  │
 │ (CSV/Pickle) │     │ & Preprocess  │     │  & Features     │     │ (Reg/Class)    │
 └─────────────┘     └──────────────┘     └────────────────┘     └───────┬───────┘
                                                                          │
                                                                          v
 ┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────────┐
 │ Download model.pkl│ <-- │ View Metrics &     │ <-- │  Train Model              │
 │ (model + preproc.) │     │ Visualizations     │     │                           │
 └─────────────────┘     └───────────────────┘     └──────────────────────────┘
```

1. **Upload** — Drop in your dataset (CSV, Pickle, etc.)
2. **Clean & Preprocess** — Nulls are imputed, categorical features are encoded, and data is standardized automatically
3. **Configure** — Choose your target column, feature columns, and task type (regression or classification)
4. **Train** — Select a model from the supported list and kick off training
5. **Evaluate** — Review metrics (accuracy, F1, RMSE, R², etc.) and auto-generated charts
6. **Download** — Grab your `model.pkl`, which bundles the trained model **and** the exact preprocessing pipeline used, so it works out-of-the-box on new raw data

---

## 🎬 Demo

><img width="1907" height="953" alt="image" src="https://github.com/user-attachments/assets/8aa5bea0-ed9e-404b-a2cf-f448deebe11c" />

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| Frontend | HTML, CSS, JavaScript |
| ML / Data Processing | scikit-learn, pandas, NumPy |
| Visualization | Matplotlib / Seaborn / Plotly  |
| Model Serialization | Pickle (`.pkl`) |

*(Feel free to edit this table to exactly match the libraries your project uses.)*

---

## 🧩 Supported Models

**Classification**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting / XGBoost
- Naive Bayes

**Regression**
- Linear Regression
- Ridge / Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- Gradient Boosting / XGBoost Regressor



---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/automl-studio.git
cd automl-studio

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py
```

The app will be available at **`http://127.0.0.1:5000`** by default.


## 📘 Usage Guide

1. Launch the app and open it in your browser.
2. Upload your dataset (`.csv` or `.pkl`).
3. Review the auto-cleaned data preview.
4. Select your **target column** and the **feature columns** to use.
5. Choose the task type: **Regression** or **Classification**.
6. Pick a model from the dropdown list.
7. Click **Train** and wait for the training summary.
8. Review the metrics and visualizations shown on screen.
9. Click **Download Model** to get your `model.pkl`.

### Using the downloaded model in your own code

```python
import pickle

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

preprocessor = bundle["preprocessor"]
model = bundle["model"]

# Preprocess new raw data the same way it was trained
X_new_processed = preprocessor.transform(X_new)

# Predict
predictions = model.predict(X_new_processed)
```

*(Update the dictionary keys above if your bundle structure differs.)*

---

## 🗺️ Roadmap

- [ ] Add support for more file formats (Excel, JSON, Parquet)
- [ ] Hyperparameter tuning (GridSearch / Optuna integration)
- [ ] Model comparison / leaderboard view
- [ ] Cross-validation support
- [ ] Docker support for one-command deployment
- [ ] Authentication & saved user sessions
- [ ] Export training report as PDF

Have an idea? [Open an issue](../../issues) or start a discussion!

---

## 🤝 Contributing

Contributions are what make open source amazing — **all contributions are welcome**, whether it's fixing a bug, adding a new model, improving preprocessing, or polishing the UI.

1. **Fork** the repository
2. **Create a branch** for your feature or fix
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit
   ```bash
   git commit -m "Add: short description of your change"
   ```
4. **Push** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** describing what you changed and why

### Contribution Ideas
- 🐛 Bug fixes
- 🧠 New models or preprocessing strategies
- 📊 New visualization types
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- ✅ Tests

Please make sure your code is clean and, where possible, add comments/docstrings so others can follow along. Consider opening an issue first for larger changes so we can discuss the approach.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. You're free to use, modify, and distribute this project, including for commercial purposes.

---

## 🙏 Acknowledgements

- Built with [Flask](https://flask.palletsprojects.com/)
- Powered by [scikit-learn](https://scikit-learn.org/) and [pandas](https://pandas.pydata.org/)
- Thanks to everyone who contributes ideas, code, and feedback!

---

### ⭐ If you find this project useful, consider giving it a star — it helps others discover it too!
