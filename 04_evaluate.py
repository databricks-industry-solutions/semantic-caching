# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions).

# COMMAND ----------

# MAGIC %md
# MAGIC #Evaluate the RAG chains with and without caching
# MAGIC
# MAGIC In the previous notebooks, we created and deployed RAG chains with and without semantic caching. Both are now up and running, ready to handle requests. In this notebook, we will conduct a benchmarking exercise to evaluate the latency reduction achieved by the cached chain and assess the trade-off in response quality.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster configuration
# MAGIC We recommend using a cluster with the following specifications to run this solution accelerator:
# MAGIC - Unity Catalog enabled cluster 
# MAGIC - Databricks Runtime 15.4 LTS ML or above
# MAGIC - Single-node cluster: e.g. `m6id.2xlarge` on AWS or `Standard_D8ds_v4` on Azure Databricks.

# COMMAND ----------

# DBTITLE 1,Load parameters
from config import Config
config = Config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation
# MAGIC
# MAGIC For the benchmarking exercise, we will use a hundred synthesized questions stored in `data/synthetic_questions_100.csv`. To create these, we first generated ten questions related to Databricks Machine Learning product features using [dbrx-instruct](https://e2-demo-field-eng.cloud.databricks.com/editor/notebooks/1284968239746639?o=1444828305810485#command/1284968239757668). We then expanded these by reformulating each of the ten questions slightly, without changing their meaning, generating ten variations of each. This resulted in a hundred questions in total. For this process, we used [Meta Llama 3.1 70B Instruct](https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html#meta-llama-31-70b-instruct).
# MAGIC
# MAGIC We read this dataset in and save it into a delta table.

# COMMAND ----------

import pandas as pd
df = pd.read_csv('data/synthetic_questions_100.csv') # this is a small sample of 100 questions
df = spark.createDataFrame(df) # convert to a Spark DataFrame
df.write.mode('overwrite').saveAsTable(f'{config.CATALOG}.{config.SCHEMA}.synthetic_questions_100') # save to a table

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will format the questions so that we can apply the chain directly later. We store the formatted dataset in another delta table.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {config.CATALOG}.{config.SCHEMA}.synthetic_questions_100_formatted AS
SELECT STRUCT(ARRAY(STRUCT(question AS content, "user" AS role)) AS messages) AS question, base as base
FROM {config.CATALOG}.{config.SCHEMA}.synthetic_questions_100;
""")

df = spark.table(f'{config.CATALOG}.{config.SCHEMA}.synthetic_questions_100_formatted')
display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Test standard rag chain endpoint
# MAGIC
# MAGIC Now that we have our test dataset, we are going to go ahead and test the standard RAG chain endpoint. We will use [ai_query](https://docs.databricks.com/en/sql/language-manual/functions/ai_query.html) to apply the chain to the formatted table. We write the result out to another delta table.

# COMMAND ----------

# DBTITLE 1,Load testing standard RAG chain
spark.sql(f"""
CREATE OR REPLACE TABLE {config.CATALOG}.{config.SCHEMA}.standard_rag_chain_results AS
SELECT question, ai_query(
  'standard_rag_chatbot',
  question,
  returnType => 'STRUCT<choices:ARRAY<STRING>>'
  ) AS prediction, base
FROM {config.CATALOG}.{config.SCHEMA}.synthetic_questions_100_formatted;
""")

standard_rag_chain_results = spark.table(f'{config.CATALOG}.{config.SCHEMA}.standard_rag_chain_results')
display(standard_rag_chain_results)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Test rag chain with cache endpoint
# MAGIC
# MAGIC We are now going to test the RAG chain with cache endpoint.

# COMMAND ----------

# DBTITLE 1,Load testing RAG chain with cache
spark.sql(f"""
CREATE OR REPLACE TABLE {config.CATALOG}.{config.SCHEMA}.rag_chain_with_cache_results AS
SELECT question, ai_query(
    'rag_chatbot_with_cache',
    question,
    returnType => 'STRUCT<choices:ARRAY<STRING>>'
  ) AS prediction, base
FROM {config.CATALOG}.{config.SCHEMA}.synthetic_questions_100_formatted;
""")

rag_chain_with_cache_results = spark.table(f'{config.CATALOG}.{config.SCHEMA}.rag_chain_with_cache_results')
display(rag_chain_with_cache_results)

# COMMAND ----------

# MAGIC %md
# MAGIC Just by looking at the execution time, we notice that the chain with cache ran more thatn 2x faster than the the chain without.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate results using MLflow
# MAGIC
# MAGIC We will begin by evaluating the quality of the responses from both endpoints. Since the 100 questions were derived from the original 10 through reformulation (without changing their meaning), we can use the answers to the original questions as the ground truth for evaluating the responses to the 100 variations.

# COMMAND ----------

# DBTITLE 1,Reading in the original 10 questions and answers
import json
synthetic_qa = []
with open('data/synthetic_qa.txt', 'r') as file:
    for line in file:
        synthetic_qa.append(json.loads(line))

display(synthetic_qa)

# COMMAND ----------

# MAGIC %md
# MAGIC We construct an evaluation dataset for the standard RAG chain and the chain with the cache. The `prediction` colume stores the responses from the chain.

# COMMAND ----------

evaluation_standard = spark.table(f'{config.CATALOG}.{config.SCHEMA}.standard_rag_chain_results').toPandas()
evaluation_cache = spark.table(f'{config.CATALOG}.{config.SCHEMA}.rag_chain_with_cache_results').toPandas()

evaluation_standard["question"] = evaluation_standard["question"].apply(lambda x: x["messages"][0]["content"])
evaluation_standard["prediction"] = evaluation_standard["prediction"].apply(lambda x: json.loads(x["choices"][0])["message"]["content"])

evaluation_cache["question"] = evaluation_cache["question"].apply(lambda x: x["messages"][0]["content"])
evaluation_cache["prediction"] = evaluation_cache["prediction"].apply(lambda x: json.loads(x["choices"][0])["message"]["content"])

labels = pd.DataFrame(synthetic_qa).drop(["question"], axis=1)

evaluation_standard = evaluation_standard.merge(labels, on='base')
evaluation_cache = evaluation_cache.merge(labels, on='base')

# COMMAND ----------

evaluation_standard

# COMMAND ----------

evaluation_cache

# COMMAND ----------

# MAGIC %md
# MAGIC To assess the quality of the responses, we will use [`mlflow.evaluate`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate).

# COMMAND ----------

import mlflow
from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")
judge_model = "endpoints:/databricks-meta-llama-3-1-70b-instruct" # this is the model endpont you want to use as a judge

# Run evaluation for the standard chain
with mlflow.start_run(run_name="evaluation_standard"):
    standard_results = mlflow.evaluate(        
        data=evaluation_standard,
        targets="answer",
        predictions="prediction",
        model_type="question-answering",
        extra_metrics=[
          mlflow.metrics.genai.answer_similarity(model=judge_model), 
          mlflow.metrics.genai.answer_correctness(model=judge_model),
          mlflow.metrics.genai.answer_relevance(model=judge_model),
          ],
        evaluator_config={
            'col_mapping': {'inputs': 'question'}
        }
    )

# Run evaluation for the chain with cache
with mlflow.start_run(run_name="evaluation_cache"):
    cache_results = mlflow.evaluate(        
        data=evaluation_cache,
        targets="answer",
        predictions="prediction",
        model_type="question-answering",
        extra_metrics=[
          mlflow.metrics.genai.answer_similarity(model=judge_model), 
          mlflow.metrics.genai.answer_correctness(model=judge_model),
          mlflow.metrics.genai.answer_relevance(model=judge_model),
          ],
        evaluator_config={
            'col_mapping': {'inputs': 'question'}
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Let's print out the aggregated statistics of the quality metrics. 

# COMMAND ----------

print(f"See aggregated evaluation results below: \n{standard_results.metrics}")

# COMMAND ----------

print(f"See aggregated evaluation results below: \n{cache_results.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC The evaluation results show that the standard RAG chain performed slightly better on metrics like `answer_correctness/v1/mean` (scoring `4.82` vs. `4.69`) and `answer_relevance/v1/mean` (scoring `4.91` vs. `4.7`). These minor drops in performance are expected when responses are retrieved from the cache. The key takeaway is to assess whether these differences are acceptable given the cost and latency reductions provided by the caching solution. Ultimately, the decision should be based on how these trade-offs impact the business value of your use case.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the Inference tables
# MAGIC
# MAGIC Each request and response that hits the endpoint can be logged to an [inference table](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html) along with its [trace](https://docs.databricks.com/en/mlflow/mlflow-tracing.html#use-mlflow-tracing-in-production). These tables are particularly useful for debugging and auditing. We will query the inference tables for both endpoints to gain insights into performance optimization.

# COMMAND ----------

# You can just query the inference table 
standard_log = spark.read.table(f"{config.CATALOG}.{config.LOGGING_SCHEMA}.standard_rag_chatbot_payload").toPandas()
display(standard_log)

# COMMAND ----------

cache_log = spark.read.table(f"{config.CATALOG_CACHE}.{config.LOGGING_SCHEMA_CACHE}.rag_chatbot_with_cache_payload").toPandas()
display(cache_log)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's calculate the mean execution time per query. We see a significant drop in the chain with cache, which is direclty translatable for cost reduction.

# COMMAND ----------

print(f"standard rag chain mean execution time: {round(standard_log['execution_time_ms'].mean()/1000, 4)} seconds")
print(f"rag chain with cache mean execution time: {round(cache_log['execution_time_ms'].mean()/1000, 4)} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC One of the important KPIs for a cachin solution is the hit rate. We can retrieve this information from the traces stored in the inferenc table.

# COMMAND ----------

import json
import numpy as np

cache_trace = np.array(
    cache_log["response"].apply(lambda x: 1 if len(json.loads(x)["databricks_output"]["trace"]["data"]["spans"]) == 6 else 0)
)
print(f"Number of times the query hit the cache: {cache_trace.sum()}/100")

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we conducted a benchmarking exercise to compare the solutions with and without semantic caching. For this specific dataset, we observed a significant reduction in both cost and latency, though with a slight trade-off in quality. It’s important to emphasize that every use case should carefully assess the impact of these gains and losses on business objectives before making a final decision.

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
