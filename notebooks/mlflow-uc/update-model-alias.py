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

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException
import json
from pyspark.sql import functions as F
import os
import datetime

# COMMAND ----------

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

catalog = "alex_m"
db = "uc_models"
uc_model_name = f"{catalog}.{db}.{model_name}"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {db}")

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

model_version_to_evaluate = get_latest_model_version(uc_model_name)
model_version = client.get_model_version(uc_model_name, model_version_to_evaluate)
run_info = client.get_run(run_id=model_version.run_id)

model_uri = f"models:/{uc_model_name}/{model_version_to_evaluate}"
print(f"Model version to evaluate: {model_version_to_evaluate}")

# COMMAND ----------

# MAGIC %md ### Compare "champion" vs. "challenger" ("prod" vs. "dev") based on validation set
# MAGIC - Download models and compare performance
# MAGIC - Performance in this case will be based on validation set
# MAGIC - Validation set would need to have updated data (to test drift impact of champion model)

# COMMAND ----------

try:
  champion = client.get_model_version_by_alias(name=uc_model_name, alias="champion")
except RestException:
  print(f"Champion alias does not exist. Transition model to 'champion' alias for production use")
  client.set_registered_model_alias(name=uc_model_name, alias="champion", version=model_version_to_evaluate)
  message = f"{uc_model_name} has been promoted to 'champion' alias and will be deployed to model serving"
  dbutils.notebook.exit(json.dumps({f"message": message}))

try:
  challenger = client.get_model_version_by_alias(name=uc_model_name, alias="challenger")
except RestException:
  print(f"Challenger alias does not exist. No need to compare models")
  message = f"{uc_model_name} challenger alias does not exist. No need to compare models. The 'champion' alias and will be deployed to model serving"
  dbutils.notebook.exit(json.dumps({f"message": message}))

# COMMAND ----------

def predict(model, spark_df):

  # loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
  input_column_names = model.metadata.get_input_schema().input_names()

  return (spark_df.withColumn("predictions", model(*input_column_names)))

# COMMAND ----------

from pyspark.sql import functions as F
from sklearn.metrics import f1_score

champion_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{uc_model_name}@champion")
challenger_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{uc_model_name}@challenger")

# load validation set
validation_set = spark.table("validation_set")

champion_results = predict(champion_model, validation_set) \
  .withColumn("predictions", F.col("predictions")[0]) \
  .toPandas()
challenger_results = predict(challenger_model, validation_set) \
  .withColumn("predictions", F.col("predictions")[0]) \
  .toPandas()

# COMMAND ----------

champion_f1_score = f1_score(champion_results['target'], champion_results['predictions'])
print(champion_f1_score)

challenger_f1_score = f1_score(challenger_results['target'], challenger_results['predictions'])
print(challenger_f1_score)

# COMMAND ----------

# MAGIC %md ### Simple logic to show how to automate model selection but can be much more rigorous:
# MAGIC - Example: challenger needs to be x% higher due or within x% of accuracy

# COMMAND ----------

if challenger_f1_score > champion_f1_score:
  print(f"Promoting challenger to champion alias.")
  client.set_registered_model_alias(name=uc_model_name, alias="champion", version=challenger.version)
else:
  print(f"Challenger did not perform better. Champion will remain deployed in production")
