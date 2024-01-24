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

if roll_back.lower() == "false":
  message = f"Roll-back is {roll_back}. Exiting notebook!"
  dbutils.notebook.exit(json.dumps({f"message": message}))

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException
import json
import os
import datetime

catalog = "alex_m"
db = "uc_models"
uc_model_name = f"{catalog}.{db}.{model_name}"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {db}")

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

model_version = client.get_model_version(name=uc_model_name, version=roll_back_version)
if "champion" in model_version.aliases:
  print(f"Model is already champion")
else:
  print(f"Setting {uc_model_name} version {roll_back_version} to champion")
  client.set_registered_model_alias(name=uc_model_name, alias="champion", version=roll_back_version)
if "challenger" in model_version.aliases:
  print(f"Removing challenger alias from champion model")
  client.delete_registered_model_alias(name=uc_model_name, alias="challenger")
