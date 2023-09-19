import pandas as pd
import numpy as np 
from prediction_model.config import config  
from prediction_model.processing.data_handling import load_dataset,save_pipeline
import prediction_model.processing.preprocessing as pp 
import prediction_model.pipeline as pipe 
import sys
import mlflow
import mlflow.sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def model_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, thresholds1 = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    # plot auc 
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    
    # Save plot
    plt.savefig("plots/ROC_curve.png")
    
    # Close plot
    plt.close()

    return(accuracy, f1, auc)

def mlflow_logs(model, X, y, name):
    
     with mlflow.start_run(run_name = name) as run:
        
        # Run id
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
         
        # Make predictions        
        pred = model.predict(X)
    
        # Generate performance metrics
        (accuracy, f1, auc) = model_metrics(y, pred)

        # Logging best parameters 
       # mlflow.log_params(model.get_params())

        # Logging model metric 
        #mlflow.log_metric("Best Accuracy on trainig score", model.score)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)

        # Logging artifacts and model
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)
        
        mlflow.end_run()

def perform_training():
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map({'N':0,'Y':1})
    X_train,X_test,y_train,y_test = train_test_split(train_data[config.FEATURES], train_y,random_state=6,test_size=0.3)
    mlflow.set_experiment("Loan_prediction")
    pipe.classification_pipeline.fit(X_train,y_train)
    mlflow_logs(pipe.classification_pipeline,X_test,y_test,"LogisticRegression")
    pipe.classification_pipeline_dt.fit(X_train,y_train)
    mlflow_logs(pipe.classification_pipeline_dt,X_test,y_test,"DecisionTree")
    pipe.classification_pipeline_rf.fit(X_train,y_train)
    mlflow_logs(pipe.classification_pipeline_rf,X_test,y_test,"RandomForest")
    # y_pred = pipe.classification_pipeline.predict(X_test)
    # acc,f1,auc = model_metrics(y_test,y_pred)
    # mlflow.log_metric("accuracy",acc)
    # mlflow.log_metric("f1-score",f1)
    # mlflow.log_metric("auc",auc)
    # model = pipe.classification_pipeline['LogisticClassifier']
    # mlflow.log_params(model.get_params())
    # mlflow.sklearn.log_model(pipe.classification_pipeline, "model")
    # mlflow.log_artifact("plots/ROC_curve.png")
    save_pipeline(pipe.classification_pipeline)


if __name__=='__main__':
    perform_training()