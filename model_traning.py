

from model_registry import MODELS



from sklearn.metrics import *

def evaluate_model(y_pred, y_test, task_type):
    

    if task_type == "regression":
        return {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
        }

    else:
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1": f1_score(y_test, y_pred, average="weighted"),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred)
        }
        
        
        
def Train_model(model_name,task_type,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST):
    models=MODELS[task_type]
    if model_name in models.keys():
        mod=models[model_name]["class"]
        prams=models[model_name]["params"]
        
    model=mod(**prams)
    
    model.fit(X_TRAIN,Y_TRAIN)
    pred=model.predict(X_TEST)
    
    results=evaluate_model(y_pred=pred,y_test=Y_TEST,task_type=task_type)
    
    
    
    return model,results,pred
    
    
    
        
        
        
        
        

    
    