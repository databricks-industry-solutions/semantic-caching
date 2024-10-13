# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/semantic-caching).

# COMMAND ----------

# MAGIC %md
# MAGIC #Create and deploy a standard RAG chain
# MAGIC
# MAGIC In this notebook, we will build a standard RAG chatbot without semantic caching to serve as a benchmark. We will utilize the [Databricks Mosaic AI Agent Framework](https://www.databricks.com/product/machine-learning/retrieval-augmented-generation), which enables rapid prototyping of the initial application. In the following cells, we will define a chain, log and register it using MLflow and Unity Catalog, and finally deploy it behind a [Databricks Mosaic AI Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html) endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster configuration
# MAGIC We recommend using a cluster with the following specifications to run this solution accelerator:
# MAGIC - Unity Catalog enabled cluster 
# MAGIC - Databricks Runtime 15.4 LTS ML or above
# MAGIC - Single-node cluster: e.g. `m6id.2xlarge` on AWS or `Standard_D8ds_v4` on Azure Databricks.

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load parameters
from config import Config
config = Config()

# COMMAND ----------

# DBTITLE 1,Run init notebok
# MAGIC %run ./99_init $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we define environment variables `HOST` and `TOKEN` for our Model Serving endpoint to authenticate against our Vector Search index. 

# COMMAND ----------

# DBTITLE 1,Define environmental variables
import os

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['DATABRICKS_HOST'] = HOST
os.environ['DATABRICKS_TOKEN'] = TOKEN

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and register a chain to MLflow 
# MAGIC
# MAGIC The next cell defines a standard RAG chain using Langchain. When executed, it will write the content to the `chain/chain.py` file, which will then be used to log the chain in MLflow.

# COMMAND ----------

# MAGIC %%writefile chain/chain.py
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_core.prompts import ChatPromptTemplate
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from operator import itemgetter
# MAGIC from config import Config
# MAGIC import mlflow
# MAGIC import os
# MAGIC
# MAGIC # Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC # load parameters
# MAGIC config = Config()
# MAGIC
# MAGIC # Connect to the Vector Search Index
# MAGIC vs_index = VectorSearchClient(
# MAGIC     workspace_url=os.environ['DATABRICKS_HOST'],
# MAGIC     personal_access_token=os.environ['DATABRICKS_TOKEN'],
# MAGIC     disable_notice=True,
# MAGIC     ).get_index(
# MAGIC     endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME,
# MAGIC     index_name=config.VS_INDEX_FULLNAME,
# MAGIC )
# MAGIC
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     vs_index,
# MAGIC     text_column="content",
# MAGIC     columns=["id", "content", "url"],
# MAGIC ).as_retriever(search_kwargs={"k": 3}) # Number of search results that the retriever returns
# MAGIC # Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
# MAGIC mlflow.models.set_retriever_schema(primary_key="id", text_column="content", doc_uri="url")
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
# MAGIC def format_context(docs):
# MAGIC     chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC # Prompt template to be used to prompt the LLM
# MAGIC prompt = ChatPromptTemplate.from_messages(
# MAGIC     [
# MAGIC         ("system", f"{config.LLM_PROMPT_TEMPLATE}"),
# MAGIC         ("user", "{question}"),
# MAGIC     ]
# MAGIC )
# MAGIC
# MAGIC # Our foundation model answering the final prompt
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=config.LLM_MODEL_SERVING_ENDPOINT_NAME,
# MAGIC     extra_params={"temperature": 0.01, "max_tokens": 500}
# MAGIC )
# MAGIC
# MAGIC # Return the string contents of the most recent messages: [{...}] from the user to be used as input question
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "context": itemgetter("messages")
# MAGIC         | RunnableLambda(extract_user_query_string)
# MAGIC         | vector_search_as_retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)

# COMMAND ----------

# MAGIC %md
# MAGIC In this cell, we log the chain to MLflow. Note that we are passing `config.py` as a dependency, allowing the chain to load the necessary parameters when deployed to another compute environment or to a Model Serving endpoint. MLflow returns a trace of the inference that shows the detail breakdown of the latency and the input/output from each step in the chain.

# COMMAND ----------

# Log the model to MLflow
config_file_path = "config.py"

# Create a config file to be used by the chain
with mlflow.start_run(run_name=f"rag_chatbot"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain/chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
        artifact_path="chain",  # Required by MLflow
        input_example=config.INPUT_EXAMPLE,  # MLflow will execute the chain before logging & capture it's output schema.
        code_paths = [config_file_path], # Include the config file in the model
    )

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(config.INPUT_EXAMPLE)

# COMMAND ----------

# MAGIC %md
# MAGIC If we are happy with the logged chain, we will go ahead and register the chain in Unity Catalog.

# COMMAND ----------

# Register to UC
uc_registered_model_info = mlflow.register_model(
  model_uri=logged_chain_info.model_uri, 
  name=config.MODEL_FULLNAME
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the chain to a Model Serving endpoint
# MAGIC
# MAGIC We deploy the chaing using custom functions defined in the `utils.py` script.

# COMMAND ----------

import utils
utils.deploy_model_serving_endpoint(
  spark, 
  config.MODEL_FULLNAME,
  config.CATALOG,
  config.LOGGING_SCHEMA,
  config.ENDPOINT_NAME,
  HOST,
  TOKEN,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Wait until the endpoint is ready. This may take some time (~15 minutes), so grab a coffee!

# COMMAND ----------

utils.wait_for_model_serving_endpoint_to_be_ready(config.ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC Once the endpoint is up and running, let's send a request and see how it responds.

# COMMAND ----------

import utils
data = {
    "inputs": {
        "messages": [
            {
                "content": "What is Model Serving?",
                "role": "user"
            }
        ]
    }
}
# Now, call the function with the correctly formatted data
utils.send_request_to_endpoint(
    config.ENDPOINT_NAME, 
    data,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we built a standard RAG chatbot without semantic caching to serve. We will use this chain to benchmark against the chain with semantic caching, which we will build in the next `03_rag_chatbot_with_cache` notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
