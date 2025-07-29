# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks上のMLFlowで`/plamo-embedding-1b`モデルを管理する
# MAGIC
# MAGIC この例では、[pfnet/plamo-embedding-1b](https://huggingface.co/pfnet/plamo-embedding-1b)を  MLFLow にロギングし、Unity Catalog でモデルを管理し、モデルサービングエンドポイントを作成する方法を示します。
# MAGIC
# MAGIC このノートブックの環境
# MAGIC - ランタイム: 16.3 ML Runtime
# MAGIC - インスタンス: AWS の `i3.xlarge` または Azure の `Standard_D4DS_v5` 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをMLFlowに記録する

# COMMAND ----------

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# You can download models from the Hugging Face Hub 🤗 as follows:
tokenizer = AutoTokenizer.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)
model = AutoModel.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # README通りに試す

# COMMAND ----------


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

query = "PLaMo-Embedding-1Bとは何ですか？"
documents = [
    "PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。",
    "最近は随分と暖かくなりましたね。"
]

with torch.inference_mode():
    # For embedding query texts in information retrieval, please use the `encode_query` method.
    # You also need to pass the `tokenizer`.
    query_embedding = model.encode_query(query, tokenizer)
    # For other texts/sentences, please use the `encode_document` method.
    # Also, for applications other than information retrieval, please use the `encode_document` method.
    document_embeddings = model.encode_document(documents, tokenizer)


similarities = F.cosine_similarity(query_embedding, document_embeddings)
print(similarities)

# COMMAND ----------

# カタログ、スキーマ、モデル名の定義
model_uc_catalog = "kohei_arai" #TODO: Change
model_uc_schema = "demo"
uc_model_name = "plamo_embedding_1b"

# COMMAND ----------

import mlflow
from transformers import pipeline
import numpy as np

transformers_model = {"model": model, "tokenizer": tokenizer}
task = "llm/v1/embeddings"  # correct task for embeddings

# Set the registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Create a pipeline
embedding_pipeline = pipeline(task="feature-extraction", model=model, tokenizer=tokenizer)

# Register the model
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=embedding_pipeline,
        task=task,
        artifact_path="model",
        registered_model_name=f"{model_uc_catalog}.{model_uc_schema}.{uc_model_name}"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルを Unity Catalog に登録する
# MAGIC  デフォルトでは、MLflowはDatabricksワークスペースのモデルレジストリにモデルを登録します。代わりにUnity Catalogにモデルを登録するには、[ドキュメント](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)に従い、レジストリサーバーをDatabricks Unity Catalogに設定します。
# MAGIC
# MAGIC  Unity Catalogにモデルを登録するには、ワークスペースでUnity Catalogが有効になっている必要があるなど、[いくつかの要件](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#requirements)があります。
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unityカタログからモデルを読み込む

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

# Define the registered model name and alias
registered_name = "kohei_arai.models.plamo_embedding_1b"
alias = "latest_alias"
version = 1

# Set the alias for the desired model version (e.g., version 1)
client = MlflowClient()
client.set_registered_model_alias(registered_name, alias, version)

# Load the model using the alias
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@{alias}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルサービングエンドポイントの作成
# MAGIC モデルが登録されたら、APIを使用してDatabricks GPU Model Serving Endpointを作成し、`bge-large-en`モデルをサービングしていきます。
# MAGIC
# MAGIC 以下のデプロイにはGPUモデルサービングが必要です。GPU モデルサービングの詳細については、Databricks チームにお問い合わせいただくか、サインアップ [こちら](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit) してください。

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# サービングエンドポイントの作成または更新
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType
import mlflow
from mlflow import MlflowClient

# Define the registered model name and alias
registered_name = f"{model_uc_catalog}.{model_uc_schema}.{uc_model_name}"
alias = "latest_alias"
version = 1

# Set the alias for the desired model version (e.g., version 1)
client = MlflowClient()
client.set_registered_model_alias(registered_name, alias, version)

# Load the model using the alias
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@{alias}")
serving_endpoint_name = endpoint_name

# Get the model version using the alias directly
model_uri = f"models:/{registered_name}@{alias}"
model_name = registered_name

# No need to get the latest model version using MLflow client
# latest_model_version = client.get_latest_versions(model_name, stages=["None"])[0].version

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=version,  # Use the version directly
            workload_type=ServedModelInputWorkloadType.GPU_SMALL,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{databricks_url}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint is None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config, timeout=timedelta(minutes=60))
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name, timeout=timedelta(minutes=60))
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC モデルサービングエンドポイントの準備ができたら、同じワークスペースで実行されているMLflow Deployments SDKで簡単にクエリできます。

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

embeddings_response = client.predict(
    endpoint=endpoint_name,
    inputs={
        "inputs": ["おはようございます"]
    }
)
embeddings_response['predictions']
