# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/semantic-caching).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Semantic Cache Solution Accelerator
# MAGIC
# MAGIC Generative AI models are revolutionizing industries, with techniques like Retrieval Augmented Generation (RAG) and Compound AI systems leading the charge. These models empower organizations by enhancing capabilities such as information retrieval, decision-making, and content generation. However, the implementation of these systems is often accompanied by significant costs, especially in terms of computational resources. Despite these challenges, the rapid advancement of AI platforms and the development of more efficient algorithms are enabling businesses to optimize costs and scale AI-driven solutions more effectively.
# MAGIC
# MAGIC Semantic cache is a technique that is adopted by many enterprises to reduce the computational load of AI-driven systems. As generative AI models handle increasingly complex queries, there is often semantic overlap between different queries, such as users asking variations of the same question. Without semantic caching, these systems would need to repeatedly perform resource-intensive computations, leading to inefficiencies. By storing the previously processed queries and responses, semantic caching allows AI models to retrieve relevant information without recalculating, thereby reducing latency, lowering server load, and conserving computational resources. This becomes especially important as AI applications scale, ensuring cost-effectiveness and maintaining high performance, particularly in natural language processing, where nuanced query variations are frequent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How Semantic Caching works
# MAGIC
# MAGIC Semantic caching leverages a vector database to store and retrieve answers based on the meaning or semantics of a question rather than just its keywords. In this system, each question is embedded as a vector, and the cached answers are stored. When a new query is submitted, the system searches the database for similar vectors, returning a cached response when a match is found. When a suitable match is not found, the system proceeds to execute the standard pipeline to generate a response, and in turn persists the new question and answer pair in the database. 
# MAGIC
# MAGIC ![Semantic Caching Architecture](image/architecture.png)
# MAGIC
# MAGIC This technique is particularly effective for handling high-volume, repetitive queries such as those often found in customer FAQs, where users frequently ask the same or similar questions. Some of the key business benefits of semantic cache are:
# MAGIC
# MAGIC - **Reduce Costs**: With fewer computationally expensive model calls, businesses will see significant cost savings. The system bypasses the need to generate new answers for questions that have already been asked, leading to reduced usage of cloud resources and lower operational costs.
# MAGIC - **Faster Response Time**: Customer satisfaction is closely tied to how quickly they receive answers. With semantic caching, chatbots can instantly retrieve answers from the cache, dramatically reducing the time it takes to respond to queries.
# MAGIC - **Scalability**: As businesses scale, so do the number of customer inquiries. Caching frequently asked questions ensures the chatbot can handle increased volumes without a corresponding increase in costs or latency.
# MAGIC
# MAGIC Some use cases we see in the market that are especially suitable for semantic caching include:
# MAGIC
# MAGIC - **FAQs**: Questions that customers frequently ask—such as product details, order statuses, or return policies—are prime candidates for caching. Businesses can quickly address these repetitive queries without taxing the system.
# MAGIC - **Support Tickets**: For companies that manage large-scale customer support, semantic caching can be implemented to address commonly recurring issues.
# MAGIC - **Internal Knowledge Bases**: Employees often ask the same internal queries, and caching these answers can improve productivity by providing instant access to stored knowledge.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Semantic Cache on Databricks Mosaic AI
# MAGIC
# MAGIC Databricks provides an optimal platform for building AI agents with semantic caching capabilities. With Databricks Mosaic AI, users have access to all necessary components such as a vector database, agent development framework, agent serving, and an agent evaluation framework on a unified, highly governed platform. This ensures that key assets, including data, vector indexes, models, agents, and endpoints, are centrally managed under robust governance.
# MAGIC
# MAGIC Mosaic AI also offers an open architecture, allowing users to experiment with various models for embeddings and generation. Leveraging the Mosaic AI Agent Framework and Evaluation tools, users can rapidly iterate on applications until they meet production-level standards. Once deployed, KPIs like hit ratios and latency can be monitored using MLflow traces, which are automatically logged in Inference Tables for easy tracking.
# MAGIC
# MAGIC If you're looking to implement semantic caching for your AI system on Databricks, we're excited to introduce the Semantic Cache Solution Accelerator. This accelerator is designed to help you get started quickly and efficiently, providing a streamlined path to implementation.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Case Study
# MAGIC
# MAGIC Imagine you're operating an AI chatbot on your products' public documentation page. This chatbot answers visitors' questions about your products using a retriever-augmented generation (RAG) architecture. After reviewing the submitted user questions and responses, you notice a large number of redundant queries—phrased differently but carrying the same meaning. You're getting feedback from the users that the chatbot's response time is too long and also facing pressure from management to reduce the operational costs of the chatbot. 
# MAGIC
# MAGIC In the following notebooks, we'll explore how semantic caching can significantly lower both total cost and latency, with only a minor trade-off in response quality.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC The dataset we use for this solution accelerator was synthesized and is stored inside `./data/`. We first generated a list of 10 questions related to Databricks Machine Learning product features using [dbrx-instruct](https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html#dbrx-instruct). We then bootstrapped these questions to generate 100 questions. We reformulate each of the 10 questions slightly differently without changing the meaning. We used [Meta Llama 3.1 70B Instruct](https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html#meta-llama-31-70b-instruct) for this. 
# MAGIC
# MAGIC
# MAGIC The goal of this exploratory data analysis is to identify the optimal similarity score threshold that separates semantically similar questions from non-similar ones. This threshold should maximize the cache hit rate while minimizing false positives. In other words, we want to ensure that the cache is only hit when a question is similar enough to an existing question in our vector database, but also want to avoid running our LLM chain if the question could have been answered from the cache instead.
# MAGIC
# MAGIC A synthesized dataset is helpful as it provides ground truth labels, meaning that the questions bootstrapped from the same original question belong to the same semantic class. This is captured in the colume `base` in the `data/synthetic_questions_100.csv dataset`. This dataset allows for accurate validation of the threshold's performance in separating similar and non-similar questions, which we we see in the following.
# MAGIC
# MAGIC Let's first load in the configuration parameters (find more information about `Config` in the next notebook).

# COMMAND ----------

# DBTITLE 1,Load parameters
from config import Config
config = Config()

# COMMAND ----------

# MAGIC %md
# MAGIC We will read in the dataset as a pandas DataFrame and apply an embedding model to the questions.

# COMMAND ----------

import pandas as pd
import mlflow.deployments
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import StringType

qa_data = pd.read_csv('data/synthetic_questions_100.csv')[['base', 'question']]

deploy_client = mlflow.deployments.get_deploy_client("databricks")
def get_embedding(question):
    response = deploy_client.predict(endpoint=config.EMBEDDING_MODEL_SERVING_ENDPOINT_NAME, inputs={"input": question})
    return response.data[0]["embedding"]

# Apply an embedding model to the 'question' column and create a new column 'embedding'
qa_data["embedding"] = qa_data["question"].apply(lambda x: get_embedding(x))

display(qa_data)

# COMMAND ----------

# MAGIC %md
# MAGIC We will perform a cross join between all the questions to calculate the similarity score for every possible pair of combinations, which will result in 10,000 rows. 

# COMMAND ----------

df = qa_data.merge(qa_data, how='cross')

# COMMAND ----------

# MAGIC %md
# MAGIC [Databricks Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) uses L2 distance as a similarity score:
# MAGIC
# MAGIC $$\frac{1}{(1 + dist(q,x)^2)}$$
# MAGIC
# MAGIC where dist is the Euclidean distance between the query q and the index entry x, defined as:
# MAGIC
# MAGIC $$dist(q,x) = \sqrt{(q_1-x_1)^2 + (q_2-x_2)^2 + \ldots + (q_d-x_d)^2}.$$
# MAGIC
# MAGIC We will calculate this metric for each combination of questions. The `similar` column shown below indicates whether both questions in the pair belong to the same semantic class (based on the base question used for synthetic data generation). Please note that due to the nature of the Euclidean distance formula, the resulting distance values may appear relatively low, especially when data points are closely clustered in a high-dimensional space.

# COMMAND ----------

import numpy as np

def get_similarity_score(embedding_x, embedding_y):
    l_norm = np.linalg.norm(np.array(embedding_x) - np.array(embedding_y))
    score = 1.0/(1.0 + l_norm*l_norm)
    return score

# Apply an embedding model to the 'question' column and create a new column 'embedding'
df["score"] = df.apply(lambda x: get_similarity_score(x["embedding_x"], x["embedding_y"]), axis=1)
df = df.loc[df["score"] != 1] # Exclude the self-similar combinations
df ["similar"] = df.apply(lambda x: True if x["base_x"] == x["base_y"] else False, axis=1)
df = df[["similar", "score"]]

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at the summary statistics and the distribution of the similar and non-similar pairs, where similar pairs are part of the "True" class, and dissimilar pairs are part of the "False" class. 

# COMMAND ----------

df.groupby('similar').describe().T

# COMMAND ----------

df.groupby('similar')['score'].plot(
  kind='hist', 
  bins=50, 
  alpha=0.65, 
  density=True, 
  figsize=(10, 6), 
  grid=True, 
  legend=True,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC The analysis shows that the similar and non-similar questions synthesized for this demo exhibit distinct distributions. However, there is a notable overlap between the two distributions, presenting a critical decision point for the solution:
# MAGIC -  If we prioritize the hit rate (i.e., more queries are sent to the cache) and set a low similarity threshold (e.g., 0.005), we can achieve a recall of over 0.75, but this will come at the expense of precision. 
# MAGIC - On the other hand, setting a higher threshold (e.g., 0.015) to prioritize precision will limit recall to around 0.25. This trade-off must be carefully evaluated by the team in collaboration with business stakeholders and keeping in mind the cost vs. quality trade-off.
# MAGIC
# MAGIC In the following notebook, we will set the threshold to 0.01 as a balanced starting point.

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
