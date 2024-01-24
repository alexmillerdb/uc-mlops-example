# Databricks notebook source
# MAGIC %pip install databricks-genai-inference databricks-sdk==0.12.0 mlflow==2.9.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Model Serving Endpoint class

# COMMAND ----------

import urllib
import json
import mlflow
import requests
import time

class EndpointApiClient:
    def __init__(self):
        self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def create_inference_endpoint(self, endpoint_name, served_models):
        data = {"name": endpoint_name, "config": {"served_models": served_models}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}")
      
      
    def inference_endpoint_exists(self, endpoint_name):
      ep = self.get_inference_endpoint(endpoint_name)
      if 'error_code' in ep and ep['error_code'] == 'RESOURCE_DOES_NOT_EXIST':
          return False
      if 'error_code' in ep and ep['error_code'] != 'RESOURCE_DOES_NOT_EXIST':
          raise Exception(f"enpoint exists ? {ep}")
      return True

    def create_enpoint_if_not_exists(self, enpoint_name, model_name, model_version, workload_size, scale_to_zero_enabled=True, wait_start=True):
      if not self.inference_endpoint_exists(enpoint_name):
        models = [{
              "model_name": model_name,
              "model_version": model_version,
              "workload_size": workload_size,
              "scale_to_zero_enabled": scale_to_zero_enabled,
        }]
        self.create_inference_endpoint(enpoint_name, models)
        if wait_start:
          self.wait_endpoint_start(enpoint_name)
      
      
    def list_inference_endpoints(self):
        return self._get("api/2.0/serving-endpoints")

    def update_model_endpoint(self, endpoint_name, conf):
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", conf)

    def delete_inference_endpoint(self, endpoint_name):
        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    def wait_endpoint_start(self, endpoint_name):
      i = 0
      while self.get_inference_endpoint(endpoint_name)['state']['config_update'] == "IN_PROGRESS" and i < 500:
        time.sleep(5)
        i += 1
        print("waiting for endpoint to start")
      
    # Making predictions

    def query_inference_endpoint(self, endpoint_name, data):
        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri, data = {}):
        return requests.get(f"{self.base_url}/{uri}", params=data, headers=self.headers).json()

    def _post(self, uri, data = {}):
        return requests.post(f"{self.base_url}/{uri}", json=data, headers=self.headers).json()

    def _put(self, uri, data = {}):
        return requests.put(f"{self.base_url}/{uri}", json=data, headers=self.headers).json()

    def _delete(self, uri, data = {}):
        return requests.delete(f"{self.base_url}/{uri}", json=data, headers=self.headers).json()


serving_client = EndpointApiClient()

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

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException
import os

# COMMAND ----------

dbutils.widgets.text("model_name", "uc_model_example")
dbutils.widgets.text("run_name", "xgboost_models")
dbutils.widgets.text("experiment_name", "/Users/alex.miller@databricks.com/alex_m_uc_example")
dbutils.widgets.text("config_file", "model_serving_config1.json")
# experiment_name = "/Users/alex.miller@databricks.com/alex_m_uc_example"

model_name = dbutils.widgets.get("model_name")
run_name = dbutils.widgets.get("run_name")
experiment_name = dbutils.widgets.get("experiment_name")
config_file = dbutils.widgets.get("config_file")
# threshold = float(dbutils.widgets.get("threshold"))

print(f"Model name: {model_name}")
print(f"Run name: {run_name}")
print(f"Experiment name: {experiment_name}")
print(f"Model serving config file: {config_file}")

# COMMAND ----------

# config = {
#   "model_name": "uc_model_example",
#   "run_name": "xgboost_models",
#   "experiment_name": "/Users/alex.miller@databricks.com/alex_m_uc_example",
#   "catalog": "alex_m",
#   "db": "uc_models",
#   "endpoint_name": "uc_mlops_example",
#   "roll_back": False,
#   "roll_back_version": 2,
#   "challenger_model_exist": True
# }

# model_name = config["model_name"]
# catalog = config["catalog"]
# db = schema = config["db"]
# endpoint_name = config["endpoint_name"]
# uc_model_name = f"{catalog}.{db}.{model_name}"

# COMMAND ----------

catalog = "alex_m"
db = "uc_models"
uc_model_name = f"{catalog}.{db}.{model_name}"
endpoint_name = "uc_mlops_example"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {db}")

# Set registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()
champion_model = client.get_model_version_by_alias(name=uc_model_name, alias="champion")
champion_model_version = champion_model.version
try:
  challenger_model = client.get_model_version_by_alias(name=uc_model_name, alias="challenger")
  challenger_model_version = challenger_model.version
except RestException:
  print(f"Challenger alias does not exist. No need to compare models")
  challenger_model = False

# COMMAND ----------

import json

# Step 1: Load the YAML file
with open(f"../configs/{config_file}", 'r') as cfg_file:
    serving_config = json.load(cfg_file)

assert (type(serving_config) == dict), print("Config is not in dict format")

# COMMAND ----------

if not serving_client.inference_endpoint_exists(endpoint_name):
  print(f"Creating new model serving endpoint: {endpoint_name}")
  serving_client.create_inference_endpoint(endpoint_name, serving_config)
else:
  print(f"Updating model serving endpoint {endpoint_name} with {config_file}")
  serving_client.update_model_endpoint(endpoint_name, serving_config)

serving_client.wait_endpoint_start(endpoint_name)

# COMMAND ----------

# # start the endpoint using REST API
# serving_client.create_enpoint_if_not_exists(endpoint_name, 
#                                             model_name=uc_model_name, 
#                                             model_version=champion_model_version, 
#                                             workload_size="Small", 
#                                             scale_to_zero_enabled=True, 
#                                             wait_start=True)

# serving_client.wait_endpoint_start(endpoint_name)

# COMMAND ----------

# MAGIC %md ### Integration Testing

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput

df_split = DataframeSplitInput(columns=["mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness","mean_compactness","mean_concavity",
                                        "mean_concave_points","mean_symmetry","mean_fractal_dimension","radius_error","texture_error","perimeter_error",
                                        "area_error","smoothness_error","compactness_error","concavity_error","concave_points_error","symmetry_error",
                                        "fractal_dimension_error","worst_radius","worst_texture","worst_perimeter","worst_area","worst_smoothness",
                                        "worst_compactness","worst_concavity","worst_concave_points","worst_symmetry","worst_fractal_dimension"],
                               data=[[10.49,18.61,66.86,334.3,0.1068,0.06678,0.02297,0.0178,0.1482,0.066,0.1485,1.563,1.035,10.08,0.008875,0.009362,
                                      0.01808,0.009199,0.01791,0.003317,11.06,24.54,70.76,375.4,0.1413,0.1044,0.08423,0.06528,0.2213,0.07842
                                      ]])
w = WorkspaceClient()
w.serving_endpoints.query(endpoint_name, dataframe_split=df_split)

# COMMAND ----------

champion_loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{uc_model_name}@champion")
input_column_names = champion_loaded_model.metadata.get_input_schema().input_names()

data = spark.table("breast_cancer_data").select(*input_column_names).toPandas()
display(data)

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

os.environ["DATABRICKS_HOST"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)

os.environ["ENDPOINT_URL"] = os.path.join(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get(),
    "serving-endpoints",
    endpoint_name,
    "invocations",
)


def get_response_data(columns: list, data: list):
    
    return {
        "columns": columns,
        "data": [data],
    }


def score_model(dataset: dict):
    
    url = os.environ.get("ENDPOINT_URL")
    headers = {
        "Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        "Content-Type": "application/json",
    }
    ds_dict = {"dataframe_split": dataset}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()

# COMMAND ----------

test = get_response_data(columns=data.columns.tolist(), data=data.iloc[0].tolist())
score_model(test)

# COMMAND ----------

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import re
import os

url = os.environ.get("ENDPOINT_URL")

# Define the asynchronous function to make API calls
async def score_async(session, url, columns, d, semaphore):
    async with semaphore:  # Acquire a spot in the semaphore
        headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
        dataset = get_response_data(columns=columns, data=d)
        data_json = {'dataframe_split': dataset}
        
        async with session.post(url=url, json=data_json, headers=headers) as response:
            return await response.json()

async def main(url, data, max_concurrent_tasks):
    columns = data.columns.tolist()
    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Control concurrency
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(2000):  # Adjust the range as needed
            rand_int = np.random.randint(0, 500)
            d = data.iloc[rand_int].tolist()
            task = asyncio.create_task(score_async(session, url, columns, d, semaphore))
            tasks.append(task)
        
        raw_responses = await asyncio.gather(*tasks)
        results = []
        for resp in raw_responses:
            try:
                results.append(resp)
            except Exception as e:
                continue

        return results

max_concurrent_tasks = 20  # Set this to control concurrency
results = await main(url, data, max_concurrent_tasks)

df2 = pd.DataFrame(results)
df2.head()

# COMMAND ----------

# conf = {
#     "served_models": [
#         {
#             "name": "champion",
#             "model_name": "alex_m.uc_models.uc_model_example",
#             "model_version": "4",
#             "workload_size": "Small",
#             "scale_to_zero_enabled": True,
#             "workload_type": "CPU"
#         }
#     ],
#     "traffic_config": {
#         "routes": [
#             {
#                 "served_model_name": "champion",
#                 "traffic_percentage": 100,
#                 "served_entity_name": "champion",
#             }
#         ]
#     },
#     "auto_capture_config": {
#         "catalog_name": "alex_m",
#         "schema_name": "uc_models",
#         "table_name_prefix": "uc_mlops_example",
#         "enabled": True,
#     },
# }

# COMMAND ----------

# serving_client.update_model_endpoint(endpoint_name, serving_config)

# COMMAND ----------

# MAGIC %md ### A/B testing of models

# COMMAND ----------

# conf = {
#     "served_models": [
#         {
#             "name": "champion",
#             "model_name": "alex_m.uc_models.uc_model_example",
#             "model_version": "4",
#             "workload_size": "Small",
#             "scale_to_zero_enabled": True,
#         },
#         {
#             "name": "challenger",
#             "model_name": "alex_m.uc_models.uc_model_example",
#             "model_version": 3,
#             "workload_size": "Small",
#             "scale_to_zero_enabled": True,
#         },
#     ],
#     "traffic_config": {
#         "routes": [
#             {"served_model_name": "champion", "traffic_percentage": 80},
#             {"served_model_name": "challenger", "traffic_percentage": 20},
#         ]
#     },
# }
