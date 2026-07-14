from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier

MODELS = {
    "regression": {
        "Linear Regression": {"class": LinearRegression, "params": {}},
        "Ridge": {"class": Ridge, "params": {"alpha": 1.0}},
        "Lasso": {"class": Lasso, "params": {"alpha": 1.0}},
        "Elastic Net": {"class": ElasticNet, "params": {"alpha": 1.0, "l1_ratio": 0.5}},
        "Random Forest": {"class": RandomForestRegressor, "params": {"n_estimators": 100, "max_depth": None}},
        "Extra Trees": {"class": ExtraTreesRegressor, "params": {"n_estimators": 100, "max_depth": None}},
        "Gradient Boosting": {"class": GradientBoostingRegressor, "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}},
        "Adaboost": {"class": AdaBoostRegressor, "params": {"n_estimators": 50, "learning_rate": 1.0}},
        "Decision Tree": {"class": DecisionTreeRegressor, "params": {"max_depth": None}},
        "SVR": {"class": SVR, "params": {"C": 1.0, "kernel": "rbf"}},
        "KNeighborsRegressor": {"class": KNeighborsRegressor, "params": {"n_neighbors": 5}},
        "MLPRegressor": {"class": MLPRegressor, "params": {"hidden_layer_sizes": (100,), "max_iter": 200}},
    },
    "classification": {
        "Logistic Regression": {"class": LogisticRegression, "params": {"C": 1.0, "max_iter": 100}},
        "Random Forest": {"class": RandomForestClassifier, "params": {"n_estimators": 100, "max_depth": None}},
        "Extra Trees": {"class": ExtraTreesClassifier, "params": {"n_estimators": 100, "max_depth": None}},
        "Gradient Boosting": {"class": GradientBoostingClassifier, "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}},
        "Adaboost": {"class": AdaBoostClassifier, "params": {"n_estimators": 50, "learning_rate": 1.0}},
        "Decision Tree": {"class": DecisionTreeClassifier, "params": {"max_depth": None}},
        "SVC": {"class": SVC, "params": {"C": 1.0, "kernel": "rbf"}},
        "KNeighborsClassifier": {"class": KNeighborsClassifier, "params": {"n_neighbors": 5}},
        "GaussianNB": {"class": GaussianNB, "params": {}},
        "MLPClassifier": {"class": MLPClassifier, "params": {"hidden_layer_sizes": (100,), "max_iter": 200}},
    },
}



from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

MODELS["regression"]["XGboost"] = {
    "class": XGBRegressor,
    "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
}
MODELS["classification"]["XGboost"] = {
    "class": XGBClassifier,
    "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
 }