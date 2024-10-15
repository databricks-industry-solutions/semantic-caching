# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/semantic-caching).

# COMMAND ----------

# MAGIC %md
# MAGIC #Cache eviction
# MAGIC
# MAGIC This notebook walks you through some of the eviction strategies you can employ to your semantic cache. 

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load parameters
from config import Config
config = Config()

# COMMAND ----------

# DBTITLE 1,Set environmental variables
import os

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['DATABRICKS_HOST'] = HOST
os.environ['DATABRICKS_TOKEN'] = TOKEN

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning up the cache
# MAGIC
# MAGIC We instantiate a Vector Search client to interact with a Vector Search endpoint.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from cache import Cache

vsc = VectorSearchClient(
    workspace_url=HOST,
    personal_access_token=TOKEN,
    disable_notice=True,
    )

semantic_cache = Cache(vsc, config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## FIFO (First-In-First-Out) Strategy
# MAGIC
# MAGIC **FIFO** (First-In-First-Out) removes the oldest cached items first. In a **semantic caching** context for **LLM responses**, it is useful when:
# MAGIC **Static or frequently changing queries**: If queries or questions tend to change frequently over time, older answers might become irrelevant quickly.
# MAGIC - **Use Case**: Effective in scenarios where users query frequently changing topics (e.g., breaking news or real-time.)
# MAGIC
# MAGIC #### Pros:
# MAGIC - Simple to implement.
# MAGIC - Removes outdated or stale responses automatically.
# MAGIC
# MAGIC #### Cons:
# MAGIC - Does not account for query popularity. Frequently asked questions might be evicted even if they are still relevant.
# MAGIC - Not ideal for handling frequently recurring queries, as important cached answers could be removed.
# MAGIC

# COMMAND ----------

semantic_cache.evict(strategy='FIFO', max_documents=4, batch_size=4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LRU (Least Recently Used) Strategy
# MAGIC
# MAGIC **LRU** (Least Recently Used) evicts items that haven't been accessed recently. This strategy works well in **semantic caching** for **LLM responses** when:
# MAGIC - **Popular or recurring questions**: Frequently asked questions (FAQs) remain in the cache while infrequent or one-off queries are evicted.
# MAGIC - **Use Case**: Best suited for systems handling recurring queries, such as customer support, FAQ systems, or educational queries where the same questions are asked repeatedly.
# MAGIC
# MAGIC #### Pros:
# MAGIC - Ensures that frequently accessed answers stay in the cache.
# MAGIC - Minimizes re-computation for common queries.
# MAGIC
# MAGIC #### Cons:
# MAGIC - Higher overhead compared to FIFO, as it tracks access patterns.
# MAGIC - May retain less relevant but frequently accessed responses, while important but less commonly asked answers could be evicted.
# MAGIC

# COMMAND ----------

semantic_cache.evict(strategy='LRU', max_documents=49)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Limitations:**
# MAGIC
# MAGIC - **Sequential Batch Eviction:** Both FIFO and LRU rely on batch eviction that involves querying and removing documents iteratively. This sequential process could slow down as the number of documents increases.
# MAGIC - **Full Cache Query:** The current implementation of __evict_fifo_ and __evict_lru_ fetches a batch of documents for each iteration, which requires a similarity search query each time. This may introduce latency for larger caches.
# MAGIC - **Single-threaded Eviction:** The eviction process operates in a single thread, and as the number of documents grows, the time taken to query and delete entries will increase.
# MAGIC
# MAGIC **Potential Improvements:**
# MAGIC
# MAGIC - **Bulk Deletion:**
# MAGIC    - Instead of deleting documents in small batches (based on batch_size), consider implementing bulk deletion by gathering all the documents to be evicted in a single query and deleting them all at once.
# MAGIC - **Parallelism/Concurrency:**
# MAGIC    - Use parallel or multi-threaded processing to speed up both the similarity search and deletion processes using Spark.
# MAGIC    - Implementing asynchronous operations can allow multiple batches to be processed concurrently, reducing overall eviction time.
# MAGIC - **Optimize Batch Size:**
# MAGIC    - Fine-tune the batch_size dynamically based on the current system load or cache size. Larger batches may reduce the number of queries but may also consume more memory, so optimization here is key.
# MAGIC - **Index Partitioning:**
# MAGIC    - If possible, partition the index based on time (for FIFO) or access time (for LRU). This would allow the search and eviction process to be more efficient, as it would target a specific partition instead of querying the entire cache.
# MAGIC - **Cache Usage Statistics:**
# MAGIC    - Integrate a system to track the real-time size of the cache and update indexed_row_count without querying the entire cache each time. This would reduce the number of times you need to check the total cache size during eviction.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
