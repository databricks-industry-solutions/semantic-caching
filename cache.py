import json
import utils
import mlflow
import logging
from uuid import uuid4
from datetime import datetime
from databricks.vector_search.client import VectorSearchClient

class Cache:
    def __init__(self, vsc, config):
        mlflow.set_tracking_uri("databricks")
        self.vsc = vsc
        self.config = config
        
    def create_cache(self):
        # Create or wait for the endpoint
        utils.create_or_wait_for_endpoint(self.vsc, self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE)
        logging.info(f"Vector search endpoint '{self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE}' is ready")

        # Create or update the main index
        utils.create_or_update_direct_index(
            self.vsc, 
            self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE, 
            self.config.VS_INDEX_FULLNAME_CACHE, 
            self.config.VECTOR_SEARCH_INDEX_SCHEMA_CACHE,
            self.config.VECTOR_SEARCH_INDEX_CONFIG_CACHE,
        )
        logging.info(f"Main index '{self.config.VS_INDEX_FULLNAME_CACHE}' created/updated and is ready")
        logging.info("Environment setup completed successfully")

    @staticmethod
    def load_data(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def get_embedding(self, text):
        from mlflow.deployments import get_deploy_client
        client = get_deploy_client("databricks")
        response = client.predict(
        endpoint=self.config.EMBEDDING_MODEL_SERVING_ENDPOINT_NAME,
        inputs={"input": [text]})
        return response.data[0]['embedding']

    def warm_cache(self, batch_size=100):
        vs_index_cache = self.vsc.get_index(
            index_name=self.config.VS_INDEX_FULLNAME_CACHE, 
            endpoint_name=self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
            )
        # Load dataset
        data = Cache.load_data(self.config.CACHE_WARMING_FILE_PATH)
        logging.info(f"Loaded {len(data)} documents from {self.config.CACHE_WARMING_FILE_PATH}")
        documents = []
        for idx, item in enumerate(data):
            if 'question' in item and 'answer' in item:
                embedding = self.get_embedding(item['question']) 
                doc = {
                    "id": str(idx),
                    "creator": "system",
                    "question": item["question"],
                    "answer": item["answer"],
                    "access_level": 0,
                    "created_at": datetime.now().isoformat(),
                    "text_vector": embedding
                }
                documents.append(doc)
            
            # Upsert when batch size is reached
            if len(documents) >= batch_size:
                try:
                    vs_index_cache.upsert(documents)
                    print(f"Successfully upserted batch of {len(documents)} documents.")
                except Exception as e:
                    print(f"Error upserting batch: {str(e)}")
                documents = []  # Clear the batch

        # Upsert any remaining documents
        if documents:
            try:
                vs_index_cache.upsert(documents)
                print(f"Successfully upserted final batch of {len(documents)} documents.")
            except Exception as e:
                print(f"Error upserting final batch: {str(e)}")

        logging.info("Index details:")
        logging.info(f"  Type: {type(vs_index_cache)}")
        logging.info(f"  Name: {vs_index_cache.name}")
        logging.info(f"  Endpoint name: {vs_index_cache.endpoint_name}")
        logging.info(f"Finished loading documents into the index.")
        logging.info("Cache warming completed successfully")

    # Get response from cache 
    def get_from_cache(self, question, creator="user", access_level=0):   
        vs_index_cache = self.vsc.get_index(
            index_name=self.config.VS_INDEX_FULLNAME_CACHE, 
            endpoint_name=self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
            ) 
        # Check if the question exists in the cache
        qa = {"question": question, "answer": ""}
        results = vs_index_cache.similarity_search(
            query_vector=self.get_embedding(question),
            columns=["id", "question", "answer"],
            num_results=1
        )
        if results and results['result']['row_count'] > 0:
            score = results['result']['data_array'][0][3]  # Get the score
            logging.info(f"Score: {score}")
            try:
                if float(score) >= self.config.SIMILARITY_THRESHOLD: 
                    # Cache hit
                    qa["answer"] = results['result']['data_array'][0][2]
                    record_id = results['result']['data_array'][0][0]  # Assuming 'id' is the first column
                    logging.info("Cache hit: True")
                else:
                    logging.info("Cache hit: False")
            except ValueError:
                logging.info(f"Warning: Invalid score value: {score}")
        return qa

    # Store response to the cache 
    def store_in_cache(self, question, answer, creator="user", access_level=0):
        vs_index_cache = self.vsc.get_index(
            index_name=self.config.VS_INDEX_FULLNAME_CACHE, 
            endpoint_name=self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
            )
        document = {
            "id": str(uuid4()),
            "creator": creator,
            "question": question,
            "answer": answer,
            "access_level": access_level,
            "created_at": datetime.now().isoformat(),
            "text_vector": self.get_embedding(question),
        }
        vs_index_cache.upsert([document])

    def evict(self, strategy='FIFO', max_documents=1000, batch_size=100):
        total_docs = self.get_indexed_row_count()
        
        if total_docs <= max_documents:
            logging.info(f"Cache size ({total_docs}) is within limit ({max_documents}). No eviction needed.")
            return
    
        docs_to_remove = total_docs - max_documents
        logging.info(f"Evicting {docs_to_remove} documents from cache using {strategy} strategy...")
            
        index = self.vsc.get_index(
            index_name=self.config.VS_INDEX_FULLNAME_CACHE,
            endpoint_name=self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE
        )

        if strategy == 'FIFO':
            self._evict_fifo(index, docs_to_remove, batch_size)
        elif strategy == 'LRU':
            self._evict_lru(index, docs_to_remove, batch_size)
        else:
            raise ValueError(f"Unknown eviction strategy: {strategy}")
        
        logging.info("Cache eviction completed.")

    def _evict_fifo(self, index, docs_to_remove, batch_size):
        while docs_to_remove > 0:
            results = index.similarity_search(
                query_vector=[0] * self.config.EMBEDDING_DIMENSION,
                columns=["id", "created_at"],
                num_results=min(docs_to_remove, batch_size),
            )
            
            if not results or results['result']['row_count'] == 0:
                break
            
            ids_to_remove = [row[0] for row in results['result']['data_array']]
            index.delete(ids_to_remove)
            
            docs_to_remove -= len(ids_to_remove)
            logging.info(f"Removed {len(ids_to_remove)} documents from cache (FIFO).")

    def _evict_lru(self, index, docs_to_remove, batch_size):
        while docs_to_remove > 0:
            results = index.similarity_search(
                query_vector=[0] * self.config.EMBEDDING_DIMENSION,
                columns=["id", "last_accessed"],
                num_results=min(docs_to_remove, batch_size),
            )
            
            if not results or results['result']['row_count'] == 0:
                break
            
            ids_to_remove = [row[0] for row in results['result']['data_array']]
            index.delete(ids_to_remove)
            
            docs_to_remove -= len(ids_to_remove)
            logging.info(f"Removed {len(ids_to_remove)} documents from cache (LRU).")

    def get_indexed_row_count(self):
        index = self.vsc.get_index(
            index_name=self.config.VS_INDEX_FULLNAME_CACHE,
            endpoint_name=self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
        )
        description = index.describe()
        return description.get('status', {}).get('indexed_row_count', 0)
        
    def clear_cache(self):
        logging.info(f"Cleaning cache on endpoint {self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE}...")
        if utils.index_exists(self.vsc, self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE, self.config.VS_INDEX_FULLNAME_CACHE):
            try:
                self.vsc.delete_index(self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE, self.config.VS_INDEX_FULLNAME_CACHE)
                logging.info(f"Cache index {self.config.VS_INDEX_FULLNAME_CACHE} deleted successfully")
            except Exception as e:
                logging.error(f"Error deleting cache index {self.config.VS_INDEX_FULLNAME_CACHE}: {str(e)}")
        else:
            logging.info(f"Cache index {self.config.VS_INDEX_FULLNAME_CACHE} does not exist")

