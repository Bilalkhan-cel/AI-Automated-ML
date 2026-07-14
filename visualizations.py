import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    n_classes = cm.shape[0]
    size = max(6, n_classes * 0.6)
    plt.figure(figsize=(size, size * 0.8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9}
    )

    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("Actual", fontsize=11)
    plt.title("Confusion Matrix", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    return save_plot()


def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    return save_plot()


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    return save_plot()
def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_).flatten()
    else:
        return None

    top_n = 15
    indices = importance.argsort()[::-1][:top_n]
    top_importance = [importance[i] for i in indices]
    top_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(6, 5))
    sns.barplot(x=top_importance, y=top_names)
    plt.xlabel("Importance")
    plt.title("Top 15 Features")
    return save_plot()