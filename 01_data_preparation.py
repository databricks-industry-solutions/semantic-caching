# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/semantic-caching).

# COMMAND ----------

# MAGIC %md
# MAGIC #Set up Vector Search for RAG
# MAGIC
# MAGIC Our AI chatbot utilizes a retriever-augmented generation (RAG) approach. Before implementing semantic caching, we’ll first set up the vector database that supports this RAG architecture. For this, we’ll use [Databricks Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster configuration
# MAGIC We recommend using a cluster with the following specifications to run this solution accelerator:
# MAGIC - Unity Catalog enabled cluster 
# MAGIC - Databricks Runtime 15.4 LTS ML or above
# MAGIC - Single-node cluster: e.g. `m6id.2xlarge` on AWS or `Standard_D8ds_v4` on Azure Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC We install the required packages from the `requirements.txt` file into the current session.

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC `config.py` is a key file that holds all the essential parameters for the application. Open the file and define the values for the parameters according to your specific setup, such as the embedding/generation model endpoint, catalog, schema, vector search endpoint, and more. The following cell will load these parameters into the `config` variable.

# COMMAND ----------

# DBTITLE 1,Load parameters
from config import Config
config = Config()

# COMMAND ----------

# MAGIC %md
# MAGIC In the next cell, we run the `99_init` notebook, which sets up the logging policy and downloads the chunked Databricks product documentation (if it doesn't already exist) into the specified tables under the catalog and schema you defined in config.py.

# COMMAND ----------

# DBTITLE 1,Run init notebok
# MAGIC %run ./99_init $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search endpoint
# MAGIC
# MAGIC We create a Vector Search endpoint using custom functions defined in the `utils.py` script.

# COMMAND ----------

import utils
from databricks.vector_search.client import VectorSearchClient

# Instantiate the Vector Search Client
vsc = VectorSearchClient(disable_notice=True)

# Check if the endpoint exists, if not create it
if not utils.vs_endpoint_exists(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME):
    utils.create_or_wait_for_endpoint(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME)

print(f"Endpoint named {config.VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search index
# MAGIC Create a Vector Search index from the chunks of documents loaded in the previous cell. We use custom functions defined in the `utils.py` script.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Check if the index exists, if not create it
if not utils.index_exists(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME):
  
  print(f"Creating index {config.VS_INDEX_FULLNAME} on endpoint {config.VECTOR_SEARCH_ENDPOINT_NAME}...")
  
  # Create a delta sync index
  vsc.create_delta_sync_index(
    endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=config.VS_INDEX_FULLNAME,
    source_table_name=config.SOURCE_TABLE_FULLNAME,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column='content', # The column containing our text
    embedding_model_endpoint_name=config.EMBEDDING_MODEL_SERVING_ENDPOINT_NAME, #The embedding endpoint used to create the embeddings
  )
  
  # Let's wait for the index to be ready and all our embeddings to be created and indexed
  utils.wait_for_index_to_be_ready(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME)
else:
  # Trigger a sync to update our vs content with the new data saved in the table
  utils.wait_for_index_to_be_ready(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME)
  vsc.get_index(config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME).sync()

print(f"index {config.VS_INDEX_FULLNAME} on table {config.SOURCE_TABLE_FULLNAME} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query Vector Search index
# MAGIC
# MAGIC Let's see if we can run a similarity search against the index.

# COMMAND ----------

# Let's search for the chunks that are most relevant to the query "What is Model Serving?"
results = vsc.get_index(
  config.VECTOR_SEARCH_ENDPOINT_NAME, 
  config.VS_INDEX_FULLNAME
  ).similarity_search(
  query_text="What is Model Serving?",
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC We have successfully set up the vector database for our RAG chatbot. In the next `02_rag_chatbot` notebook, we will build a standard RAG chatbot without semantic caching, which will serve as a benchmark. Later, in the `03_rag_chatbot_with_cache` notebook, we will introduce semantic caching and compare its performance.

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
