# This GitHub Repo walks through example of how to deploy end-2-end model from feature engineering, model training, validation, comparison, and deployment
MLOps examples using Unity Catalog (UC) based on MLOps reference architecture.

How to navigate this repo:
- Notebooks: contains all the notebooks used to run the workflow
  - mlflow-uc contains the notebooks
  - config contains the model serving configurations used to deploy model serving endpoint
- Resources: contains the resource configurations for Databricks Asset Bundles (DABs)
  - Provisions job cluster, permissions, defines configurations, and creates workflow
- databricks.yml file is used for DABs which has to be present in the root path (more details here: https://docs.databricks.com/en/dev-tools/bundles/settings.html#overview)

To get started, run through each notebook which is numbered based on order of operations within the workflow. If you have Databricks CLI installed and configured (details here: https://docs.databricks.com/en/dev-tools/cli/authentication.html), update the databricks.yml file (instructions below). Only test in dev, this Repo is not setup for production use cases and purpose is to demo simple example. 

targets:
  dev:
    default: true
    workspace:
      host: https://<your Databricks Dev workspace>

  prod:
    workspace:
      host: https://<your Databricks Prod workspace>
      root_path: /Shared/.bundle/prod/${bundle.name}
