# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions).

# COMMAND ----------

# MAGIC %md
# MAGIC #Create and deploy a RAG chain with semantic caching
# MAGIC
# MAGIC In this notebook, we will build a RAG chatbot with semantic caching. To do this, we first need to create and warm up our cache. We’ll use [Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) for semantic caching, taking advantage of its high-performance similarity search. In the following cells, we will create and warm the cache, build a chain with a semantic caching layer, log and register it using MLflow and Unity Catalog, and finally deploy it behind a [Databricks Mosaic AI Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html) endpoint.

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
# MAGIC Here, we define environmental variables `HOST` and `TOKEN` for our Model Serving endpoint to authenticate against our Vector Search index. 

# COMMAND ----------

# DBTITLE 1,Set environmental variables
import os

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['HOST'] = HOST
os.environ['TOKEN'] = TOKEN

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and warm a cache 

# COMMAND ----------

# MAGIC %md
# MAGIC We instantiate a Vector Search client to interact with a Vector Search endpoint to create a cache. This will be an additional Vector Search Index, which, if the cache is hit, can immediately route the question to the answer in the cache.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from cache import Cache

# Create a Vector Search Client
vsc = VectorSearchClient(
    workspace_url=HOST,
    personal_access_token=TOKEN,
    disable_notice=True,
    )

# Initialize the cache
semantic_cache = Cache(vsc, config)

# COMMAND ----------

# MAGIC %md
# MAGIC We first delete the cache if it already exists.

# COMMAND ----------

semantic_cache.clear_cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC We then create a cache.

# COMMAND ----------

semantic_cache.create_cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC We finally load the cache with predefined Q&A pairs: i.e., `/data/synthetic_qa.txt`. This synthetic dataset contains a set of questions that have already been answered.

# COMMAND ----------

semantic_cache.warm_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and register a chain to MLflow 
# MAGIC
# MAGIC The next cell defines our RAG chain with semantic cache using Langchain. When executed, it will write the content to the `chain/chain_cache.py` file, which will then be used to log the chain in MLflow.

# COMMAND ----------

# MAGIC %%writefile chain/chain_cache.py
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import ChatPromptTemplate
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from operator import itemgetter
# MAGIC from datetime import datetime
# MAGIC from uuid import uuid4
# MAGIC import os
# MAGIC import mlflow
# MAGIC from cache import Cache
# MAGIC from config import Config
# MAGIC
# MAGIC
# MAGIC # Set up logging
# MAGIC import logging
# MAGIC logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# MAGIC logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
# MAGIC logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
# MAGIC
# MAGIC # Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC # Get configuration
# MAGIC config = Config()
# MAGIC
# MAGIC # Connect to Vector Search
# MAGIC vsc = VectorSearchClient(
# MAGIC     workspace_url=os.environ['HOST'],
# MAGIC     personal_access_token=os.environ['TOKEN'],
# MAGIC     disable_notice=True,
# MAGIC )
# MAGIC
# MAGIC # Get the Vector Search index
# MAGIC vs_index = vsc.get_index(
# MAGIC     index_name=config.VS_INDEX_FULLNAME,
# MAGIC     endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME,
# MAGIC     )
# MAGIC
# MAGIC # Instantiate a Cache object
# MAGIC semantic_cache = Cache(vsc, config)
# MAGIC
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     vs_index,
# MAGIC     text_column="content",
# MAGIC     columns=["id", "content", "url"],
# MAGIC ).as_retriever(search_kwargs={"k": 3}) # Number of search results that the retriever returns
# MAGIC
# MAGIC # Method to retrieve the context from the Vector Search index
# MAGIC def retrieve_context(qa):
# MAGIC     return vector_search_as_retriever.invoke(qa["question"])
# MAGIC
# MAGIC # Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
# MAGIC mlflow.models.set_retriever_schema(primary_key="id", text_column="content", doc_uri="url")
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
# MAGIC def format_context(docs):
# MAGIC     chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC # Create a prompt template for response generation
# MAGIC prompt = ChatPromptTemplate.from_messages(
# MAGIC     [
# MAGIC         ("system", f"{config.LLM_PROMPT_TEMPLATE}"),
# MAGIC         ("user", "{question}"),
# MAGIC     ]
# MAGIC )
# MAGIC
# MAGIC # Define our foundation model answering the final prompt
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=config.LLM_MODEL_SERVING_ENDPOINT_NAME,
# MAGIC     extra_params={"temperature": 0.01, "max_tokens": 500}
# MAGIC )
# MAGIC
# MAGIC # Call the foundation model
# MAGIC def call_model(prompt):
# MAGIC     response = model.invoke(prompt)
# MAGIC     semantic_cache.store_in_cache(
# MAGIC         question = prompt.dict()['messages'][1]['content'], 
# MAGIC         answer = response.content
# MAGIC     )
# MAGIC     return response
# MAGIC
# MAGIC # Return the string contents of the most recent messages: [{...}] from the user to be used as input question
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC # Router to determine which subsequent step to be executed
# MAGIC def router(qa):
# MAGIC     if qa["answer"] == "":
# MAGIC         return rag_chain
# MAGIC     else:
# MAGIC         return (qa["answer"])
# MAGIC
# MAGIC # RAG chain
# MAGIC rag_chain = (
# MAGIC     {
# MAGIC         "question": lambda x: x["question"],
# MAGIC         "context": RunnablePassthrough()
# MAGIC         | RunnableLambda(retrieve_context)
# MAGIC         | RunnableLambda(format_context),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | RunnableLambda(call_model)
# MAGIC )
# MAGIC
# MAGIC # Full chain with cache
# MAGIC full_chain = (
# MAGIC     itemgetter("messages")
# MAGIC     | RunnableLambda(extract_user_query_string)
# MAGIC     | RunnableLambda(semantic_cache.get_from_cache)
# MAGIC     | RunnableLambda(router)
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=full_chain)

# COMMAND ----------

# MAGIC %md
# MAGIC In this cell, we log the chain to MLflow. Note that this time we are passing `cache.py` and `utils.py` along with `config.py` as dependencies, allowing the chain to also load custom classes and functions needed to another compute environment or to a Model Serving endpoint. MLflow returns a trace of the inference that shows the detail breakdown of the latency and the input/output from each step in the chain.

# COMMAND ----------

# Log the model to MLflow
config_file_path = "config.py"
cache_file_path = "cache.py"
utils_file_path = "utils.py"

with mlflow.start_run(run_name=f"rag_chatbot"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain/chain_cache.py'),  # Chain code file e.g., /path/to/the/chain.py 
        artifact_path="chain",  # Required by MLflow
        input_example=config.INPUT_EXAMPLE,  # MLflow will execute the chain before logging & capture it's output schema.
        code_paths = [cache_file_path, config_file_path, utils_file_path],
    )

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(config.INPUT_EXAMPLE)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's ask a question to the chain that we know a similar question has not been asked before therefore doesn't exist in the cache. We see in the trace that the entire chain is indeed executed.

# COMMAND ----------

chain.invoke({'messages': [{'content': "How does Databricks' feature Genie automate feature engineering for machine learning models?", 'role': 'user'}]})

# COMMAND ----------

# MAGIC %md
# MAGIC If we reformulate the question without changing the meaning, we get the response from the cache, as the question has been upserted into the cache. We see this in the trace and the execution time is less than half.

# COMMAND ----------

chain.invoke({'messages': [{'content': "What is the role of Databricks' feature Genie in automating feature engineering for machine learning models?", 'role': 'user'}]})

# COMMAND ----------

# MAGIC %md
# MAGIC Where to set the similarity threshold -`0.01` in this demo defined in `config.py`- is arguably the most important design decision you need to make for your solution. A threshold that is too high will reduce the hit rate and undermine the effect of semantic caching, but a threshold too low could generate many false positives. There is a fine balance you would need to strike. To make an informed decision, refer to the exploratory data analysis performed in the `00_introduction` notebook.
# MAGIC
# MAGIC If we are happy with the chain, we will go ahead and register the chain in Unity Catalog.

# COMMAND ----------

# Register to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=config.MODEL_FULLNAME_CACHE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the chain to a Model Serving endpoint
# MAGIC
# MAGIC We deploy the chain using custom functions defined in the `utils.py` script.

# COMMAND ----------

import utils
utils.deploy_model_serving_endpoint(
  spark, 
  config.MODEL_FULLNAME_CACHE,
  config.CATALOG_CACHE,
  config.LOGGING_SCHEMA_CACHE,
  config.ENDPOINT_NAME_CACHE,
  HOST,
  TOKEN,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Wait until the endpoint is ready. This may take some time (~15 minutes), so grab a coffee!

# COMMAND ----------

utils.wait_for_model_serving_endpoint_to_be_ready(config.ENDPOINT_NAME_CACHE)

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
utils.send_request_to_endpoint(config.ENDPOINT_NAME_CACHE, data)

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we built a RAG chatbot with semantic caching. In the next `04_evaluate` notebook, we will compare the two chains we built. 

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
