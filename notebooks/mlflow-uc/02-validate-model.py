# Databricks notebook source
# %pip install mlflow==2.9.0
# dbutils.library.restartPython()

# COMMAND ----------

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

description = "This is the updated description for my model."
model_version_to_evaluate = get_latest_model_version(uc_model_name)
client.update_model_version(name=uc_model_name, version=model_version_to_evaluate, description=description)
model_version = client.get_model_version(uc_model_name, model_version_to_evaluate)
run_info = client.get_run(run_id=model_version.run_id)

model_uri = f"models:/{uc_model_name}/{model_version_to_evaluate}"
print(f"Model version to evaluate: {model_version_to_evaluate}")

# COMMAND ----------

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# Select the feature table cols by model input schema
input_column_names = loaded_model.metadata.get_input_schema().input_names()

# load validation set
validation_set = spark.table("validation_set")

str_model_version_to_evaluate = str(model_version_to_evaluate)

# Predict on a Spark DataFrame
try:
    display(validation_set.withColumn('predictions', loaded_model(*input_column_names)))
    client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="predicts", 
        value=1  # convert boolean to string
    )
except Exception: 
    print("Unable to predict on features.")
    client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="predicts", 
        value=0  # convert boolean to string
    )
    pass

if not loaded_model.metadata.signature:
    print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
    # Update UC tag to note missing signature
    client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="has_signature",
        value=0
    )
else:
    # Update UC tag to note existence of signature
    client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="has_signature",
        value=1
    )

if not model_version.description:
    # Update UC tag to note lack of description
    client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="has_description",
        value=0
    )
    print("Did you forget to add a description?")
elif not len(model_version.description) > 2:
    # Update UC tag to note description is too basic
    client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="has_description",
        value=0
    )
    print("Your description is too basic, sorry.  Please resubmit with more detail (40 char min).")
else:
    # Update UC tag to note presence and sufficiency of description
    client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="has_description",
        value=1
    )

import os

# Create local directory 
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = client.download_artifacts(run_info.info.run_id, "", local_dir)

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
  client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="has_artifacts",
        value=0
    )
  print("There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more.")
else:
  client.set_model_version_tag(
        name=uc_model_name,
        version=str_model_version_to_evaluate,
        key="has_artifacts",
        value=1
    )
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

results = client.get_model_version(uc_model_name, model_version_to_evaluate)
results.tags

# COMMAND ----------

# MAGIC %md ### Trigger alert to Slack, email, etc.

# COMMAND ----------

# import urllib 
# import json 
# import requests, json

# def send_notification(message):
#   try:
#     slack_webhook = dbutils.secrets.get("rk_webhooks", "slack")
#     body = {'text': message}
#     response = requests.post(
#       slack_webhook, data=json.dumps(body),
#       headers={'Content-Type': 'application/json'})
#     if response.status_code != 200:
#       raise ValueError(
#           'Request to slack returned an error %s, the response is:\n%s'
#           % (response.status_code, response.text)
#       )
#   except:
#     print("slack isn't properly setup in this workspace.")
#     pass
#   displayHTML(f"""<div style="border-radius: 10px; background-color: #adeaff; padding: 10px; width: 400px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 3px">
#         <div style="padding-bottom: 5px"><img style="width:20px; margin-bottom: -3px" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/media/resources/images/bell.png"/> <strong>Churn Model update</strong></div>
#         {message}
#         </div>""")    

# COMMAND ----------

if '0' in results or 'fail' in results:
    print("Rejecting model alias to challenger...")
    # send email, slack channel, etc.
else:
    print(f"Transitioning {uc_model_name} version {model_version_to_evaluate} to challenger")
    client.set_registered_model_alias(name=uc_model_name, alias="challenger", version=model_version_to_evaluate)
