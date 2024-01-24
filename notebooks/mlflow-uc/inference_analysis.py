# Databricks notebook source
# MAGIC %pip install databricks-genai-inference databricks-sdk==0.12.0 mlflow==2.9.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("model_name", "uc_model_example")
dbutils.widgets.text("run_name", "xgboost_models")
dbutils.widgets.text("experiment_name", "/Users/alex.miller@databricks.com/alex_m_uc_example")

model_name = dbutils.widgets.get("model_name")
run_name = dbutils.widgets.get("run_name")
experiment_name = dbutils.widgets.get("experiment_name")
# threshold = float(dbutils.widgets.get("threshold"))

print(f"Model name: {model_name}")
print(f"Run name: {run_name}")
print(f"Experiment name: {experiment_name}")

# COMMAND ----------

catalog = "alex_m"
db = "uc_models"
uc_model_name = f"{catalog}.{db}.{model_name}"
endpoint_name = "uc_mlops_example"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {db}")

# COMMAND ----------

dbutils.widgets.text("checkpoint_location", f'dbfs:/Volumes/{catalog}/{db}/checkpoints/payload_metrics', label = "Checkpoint Location")
checkpoint_location = dbutils.widgets.get("checkpoint_location")
print(checkpoint_location)

# COMMAND ----------

import requests
from typing import Dict
from pyspark.sql import functions as F


def get_endpoint_status(endpoint_name: str) -> Dict:
    # Fetch the PAT token to send in the API request
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}", json={"name": endpoint_name}, headers=headers).json()

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response.get("config", {}) or not response["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. \n"
                        f"Received response: {response} from endpoint.\n"
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response

response = get_endpoint_status(endpoint_name=endpoint_name)

auto_capture_config = response["config"]["auto_capture_config"]
catalog = auto_capture_config["catalog_name"]
schema = auto_capture_config["schema_name"]
# These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
payload_table_name = f"`{catalog}`.`{schema}`.`{payload_table_name}`"
print(f"Endpoint {endpoint_name} configured to log payload in table {payload_table_name}")

processed_table_name = f"{auto_capture_config['table_name_prefix']}_processed"
processed_table_name = f"`{catalog}`.`{schema}`.`{processed_table_name}`"
print(f"Processed requests with text evaluation metrics will be saved to: {processed_table_name}")

payloads = spark.table(payload_table_name) \
    # .filter(F.col("date")=="2024-01-03")
    # .where('status_code == 200').limit(10)
display(payloads)

# COMMAND ----------

# MAGIC %md ### Analyze Time-Series API Performance

# COMMAND ----------

display(payloads \
  .withColumn("timestamp", F.to_utc_timestamp(F.from_unixtime(F.col("timestamp_ms") / 1000, "yyyy-MM-dd HH:mm:ss"), "UTC"))
  .withColumn("model_name", F.col("request_metadata.model_name"))
  .withColumn("model_version", F.col("request_metadata.model_version"))
  .withColumn("model_name_version", F.concat("model_name", "model_version")))
