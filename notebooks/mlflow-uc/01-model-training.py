# Databricks notebook source
dbutils.widgets.text("model_name", "uc_model_example")
dbutils.widgets.text("run_name", "xgboost_models")
dbutils.widgets.text("experiment_name", "/Users/alex.miller@databricks.com/alex_m_uc_example")
dbutils.widgets.text("roll_back", "False")
dbutils.widgets.text("roll_back_version", "1")

model_name = dbutils.widgets.get("model_name")
run_name = dbutils.widgets.get("run_name")
experiment_name = dbutils.widgets.get("experiment_name")
roll_back = dbutils.widgets.get("roll_back")
roll_back_version = dbutils.widgets.get("roll_back_version")

print(f"Model name: {model_name}")
print(f"Run name: {run_name}")
print(f"Experiment name: {experiment_name}")

import json

if roll_back.lower() == "true":
  message = f"Roll-back model training pipeline. Exiting notebook!"
  dbutils.notebook.exit(json.dumps({f"message": message}))

# COMMAND ----------

catalog = "alex_m"
db = "uc_models"
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {db}")
spark.sql(f"USE SCHEMA {db}")

# COMMAND ----------

# MAGIC %md ### Load data and save to UC

# COMMAND ----------

from sklearn import datasets

data = datasets.load_breast_cancer(as_frame=True)
data_df = data.data
data_df['target'] = data.target
data_df.columns = [col.replace(" ", "_") for col in data_df.columns]

# COMMAND ----------

display(spark.createDataFrame(data_df))

# COMMAND ----------

# spark.createDataFrame(data_df).write.format("delta").saveAsTable("breast_cancer_data")

# COMMAND ----------

# MAGIC %md ### Preprocess data

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import numpy as np

RANDOM_SEED = 123 

data_df = spark.table("breast_cancer_data").toPandas()

# Splitting the dataset into training/validation and holdout sets
train_val, test = train_test_split(
    data_df, 
    test_size=0.1,
    shuffle=True, 
    random_state=RANDOM_SEED
)

# Creating X, y for training/validation set
X_train_val = train_val.drop(columns='target')
y_train_val = train_val.target

# Creating X, y for test set
X_test = test.drop(columns='target')
y_test = test.target

# Splitting training/testing set to create training set and validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, 
    y_train_val,
    stratify=y_train_val,
    shuffle=True, 
    random_state=RANDOM_SEED
)

# # Preprocessing data
# power = PowerTransformer(method='yeo-johnson', standardize=True)
# X_train = power.fit_transform(X_train)
# X_val = power.transform(X_val)
# X_test = power.transform(X_test)

# COMMAND ----------

# %sql
# DROP TABLE IF EXISTS validation_set

# COMMAND ----------

import datetime
import pandas as pd

# write out validation set to test later
# validation_set = pd.DataFrame(X_val.copy())
validation_set = X_val.copy()
validation_set["target"] = y_val
# validation_set.columns = data_df.columns
validation_set["date"] = datetime.date.today()
spark.createDataFrame(validation_set).write.format("delta").mode("overwrite").saveAsTable("validation_set")

# COMMAND ----------

# MAGIC %md ### Hyperopt

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime
import cloudpickle
from math import exp
import sklearn
from sklearn import datasets
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from hyperopt import (
    fmin, 
    hp, 
    tpe, 
    rand, 
    SparkTrials, 
    Trials, 
    STATUS_OK
)
from hyperopt.pyll.base import scope

RANDOM_SEED = 123

# Setting search space for xgboost model
search_space = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
    'subsample': hp.uniform('subsample', .5, 1.0),
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
    'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
    'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
    'gamma': hp.loguniform('gamma', -10, 10),
    'use_label_encoder': False,
    'verbosity': 0,
    'random_state': RANDOM_SEED
}

# experiment_name = "/Users/alex.miller@databricks.com/alex_m_uc_example"

try:
    EXPERIMENT_ID = mlflow.create_experiment(experiment_name)
except:
    EXPERIMENT_ID = dict(mlflow.get_experiment_by_name(experiment_name))['experiment_id']

def train_model(params):
    """
    Creates a hyperopt training model funciton that sweeps through params in a nested run
    Args:
        params: hyperparameters selected from the search space
    Returns:
        hyperopt status and the loss metric value
    """
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
    # This sometimes doesn't log everything you may want so I usually log my own metrics and params just in case
    mlflow.xgboost.autolog()

    # 
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
        # Training xgboost classifier
        model = xgb.XGBClassifier(**params)
        model = model.fit(X_train, y_train)

        # Predicting values for training and validation data, and getting prediction probabilities
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Evaluating model metrics for training set predictions and validation set predictions
        # Creating training and validation metrics dictionaries to make logging in mlflow easier
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'aucroc']
        # Training evaluation metrics
        train_accuracy = accuracy_score(y_train, y_train_pred).round(3)
        train_precision = precision_score(y_train, y_train_pred).round(3)
        train_recall = recall_score(y_train, y_train_pred).round(3)
        train_f1 = f1_score(y_train, y_train_pred).round(3)
        train_aucroc = roc_auc_score(y_train, y_train_pred_proba).round(3)
        training_metrics = {
            'Accuracy': train_accuracy, 
            'Precision': train_precision, 
            'Recall': train_recall, 
            'F1': train_f1, 
            'AUCROC': train_aucroc
        }
        training_metrics_values = list(training_metrics.values())

        # Validation evaluation metrics
        val_accuracy = accuracy_score(y_val, y_val_pred).round(3)
        val_precision = precision_score(y_val, y_val_pred).round(3)
        val_recall = recall_score(y_val, y_val_pred).round(3)
        val_f1 = f1_score(y_val, y_val_pred).round(3)
        val_aucroc = roc_auc_score(y_val, y_val_pred_proba).round(3)
        validation_metrics = {
            'Accuracy': val_accuracy, 
            'Precision': val_precision, 
            'Recall': val_recall, 
            'F1': val_f1, 
            'AUCROC': val_aucroc
        }
        validation_metrics_values = list(validation_metrics.values())

        conda_env =  _mlflow_conda_env(
          additional_conda_deps=None,
          additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), 
                               "scikit-learn=={}".format(sklearn.__version__), 
                               "xgboost=={}".format(xgb.__version__)],
          additional_conda_channels=None,
      )
        
        # Logging model signature, class, and name
        signature = infer_signature(X_train, y_val_pred)
        mlflow.xgboost.log_model(model, 'model', signature=signature, input_example=X_train.iloc[0:1], conda_env=conda_env)
        mlflow.set_tag('estimator_name', model.__class__.__name__)
        mlflow.set_tag('estimator_class', model.__class__)

        # Logging each metric
        for name, metric in list(zip(metric_names, training_metrics_values)):
            mlflow.log_metric(f'training_{name}', metric)
        for name, metric in list(zip(metric_names, validation_metrics_values)):
            mlflow.log_metric(f'validation_{name}', metric)

        # Set the loss to -1*validation auc roc so fmin maximizes the it
        return {'status': STATUS_OK, 'loss': -1*validation_metrics['AUCROC']}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep.
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials()

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .

with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
    xgboost_best_params = fmin(
        fn=train_model, 
        space=search_space, 
        algo=tpe.suggest,
        trials=spark_trials,
        max_evals=10
    )

mlflow.end_run()

# COMMAND ----------

# MAGIC %md ### Log experiments:
# MAGIC - Simple example, use hyperopt for more advanced solution
# MAGIC - Don't register model yet, just track experiments/run ids

# COMMAND ----------

# import mlflow
# import mlflow.pyfunc
# import mlflow.sklearn
# import numpy as np
# import sklearn
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, f1_score
# from mlflow.models.signature import infer_signature
# from mlflow.utils.environment import _mlflow_conda_env
# import cloudpickle
# import time
# import datetime

# uc_model_name = f"{catalog}.{db}.{model_name}"
# experiment_name = "/Users/alex.miller@databricks.com/alex_m_uc_example"

# mlflow.set_registry_uri("databricks-uc")
# mlflow.set_experiment(experiment_name)
# mlflow.sklearn.autolog()
# # mlflow.log_metric to record metrics like accuracy.
# with mlflow.start_run(run_name=model_name) as run:

#   if model_type == "rf":
#     model = train_rf(X_train, y_train)
#     conda_env =  _mlflow_conda_env(
#         additional_conda_deps=None,
#         additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
#         additional_conda_channels=None,
#     )
#   elif model_type == "xgb":
#     model = train_xgb(X_train, y_train)
#     conda_env =  _mlflow_conda_env(
#         additional_conda_deps=None,
#         additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__), "xgboost=={}".format(xgb.__version__)],
#         additional_conda_channels=None,
#     )
 
#   predictions_test = model.predict(X_test)
#   test_f1_score = f1_score(y_test, predictions_test, average="weighted")

#   input_df = X_train.iloc[0:1]
  
#   mlflow.log_metric('f1_score', test_f1_score)
#   # Log the model with a signature that defines the schema of the model's inputs and outputs. 
#   # When the model is deployed, this signature will be used to validate inputs.
#   signature = infer_signature(X_test, predictions_test)

#   mlflow.sklearn.log_model(
#     sk_model=model,
#     artifact_path="model",
#     conda_env=conda_env,
#     signature=signature,
#     input_example=input_df
# )
#   run_id = run.info.run_id

  # model_version = mlflow.register_model(
  #     f"runs:/{run_id}/model",
  #     uc_model_name
  # )
  # mlflow.set_tag(key="model_version", value=model_version.version)
#   mlflow.set_tag(key="experiment_id", value=run.info.experiment_id)
#   mlflow.set_tag(key="run_id", value=run_id)
#   mlflow.set_tag(key="timestamp", value=datetime.datetime.fromtimestamp(run.info.start_time/1000.0))
#   mlflow.set_tag(key="train_date", value=datetime.date.today())
#   mlflow.set_tag(key="feature_table", value=f"/dbfs/databricks-datasets/wine-quality")
#   mlflow.set_tag(key="description", value="Simple model showing how to use MLflow with UC")
#   mlflow.set_tag(key="validation_set", value=f"{catalog}.{db}.validation_set")

# mlflow.end_run()

# COMMAND ----------

# val_predictions2 = model.predict(X_val)
# f1_score(y_val, val_predictions2)

# COMMAND ----------

# MAGIC %md ### Query mlflow API and load best model

# COMMAND ----------

# mlflow.search_runs(experiment_ids=EXPERIMENT_ID, filter_string=f'tags.mlflow.runName = "{run_name}" and status = "FINISHED"', order_by=['metrics.validation_aucroc DESC'])

# COMMAND ----------

# Querying mlflow api instead of using web UI. Sorting by validation aucroc and then getting top run for best run.
runs_df = mlflow.search_runs(experiment_ids=EXPERIMENT_ID, order_by=['metrics.validation_aucroc DESC'])
best_run = runs_df.iloc[0]
best_run_id = best_run['run_id']
best_artifact_uri = best_run['artifact_uri']
# Loading model from best run
best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/model')

# Predicting and evaluating best model on holdout set
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred).round(3)
test_precision = precision_score(y_test, y_test_pred).round(3)
test_recall = recall_score(y_test, y_test_pred).round(3)
test_f1 = f1_score(y_test, y_test_pred).round(3)
test_aucroc = roc_auc_score(y_test, y_test_pred_proba).round(3)

print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing Precision: {test_precision}')
print(f'Testing Recall: {test_recall}')
print(f'Testing F1: {test_f1}')
print(f'Testing AUCROC: {test_aucroc}')

# COMMAND ----------

# MAGIC %md ### Register Best Model

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
uc_model_name = f"{catalog}.{db}.{model_name}"
mlflow.set_registry_uri("databricks-uc")

model_version = mlflow.register_model(
    f"runs:/{best_run_id}/model",
    uc_model_name
)
client.set_tag(run_id=best_run_id, key="model_version", value=model_version.version)
client.set_tag(run_id=best_run_id, key="model_version", value=model_version.version)
client.set_tag(run_id=best_run_id, key="experiment_id", value=run.info.experiment_id)
client.set_tag(run_id=best_run_id, key="run_id", value=best_run_id)
client.set_tag(run_id=best_run_id, key="timestamp", value=datetime.datetime.fromtimestamp(run.info.start_time/1000.0))
client.set_tag(run_id=best_run_id, key="train_date", value=datetime.date.today())
client.set_tag(run_id=best_run_id, key="description", value="Simple model showing how to use MLflow with UC")
client.set_tag(run_id=best_run_id, key="data_set", value=f"{catalog}.{db}.breast_cancer_data")
client.set_tag(run_id=best_run_id, key="validation_set", value=f"{catalog}.{db}.validation_set")
