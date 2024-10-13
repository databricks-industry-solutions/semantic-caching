from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
import yaml
import time
from typing import Iterator
from mlflow import MlflowClient
mlflow.set_registry_uri('databricks-uc')


########################################################################
###### Functions for setting up vector search index for RAG and cache
########################################################################
def vs_endpoint_exists(vsc, vs_endpoint_name):
    '''Check if a vector search endpoint exists'''
    try:
        return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
    except Exception as e:
        #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
            return True
        else:
            raise e


def create_or_wait_for_endpoint(vsc, vs_endpoint_name):
    '''Create a vector search endpoint if it doesn't exist. If it does exist, wait for it to be ready'''
    if not vs_endpoint_exists(vsc, vs_endpoint_name):
        vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")
    wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)


def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  '''Wait for a vector search endpoint to be ready'''
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")


def delete_endpoint(vsc, vs_endpoint_name):
  '''Delete a vector search endpoint'''
  print(f"Deleting endpoint {vs_endpoint_name}...")
  try:
    vsc.delete_endpoint(vs_endpoint_name)
    print(f"Endpoint {vs_endpoint_name} deleted successfully")
  except Exception as e:
    print(f"Error deleting endpoint {vs_endpoint_name}: {str(e)}")


def index_exists(vsc, vs_endpont_name, vs_index_name):
  '''Check if a vector search index exists'''
  try:
    vsc.get_index(vs_endpont_name, vs_index_name).describe()
    return True
  except Exception as e:
    if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
      print(f'Unexpected error describing the index. This could be a permission issue.')
      raise e
  return False


def wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname):
  '''Wait for a vector search index to be ready'''
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, vs_index_fullname).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{vs_index_fullname}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(vs_index_fullname, vs_endpoint_name)}")


def create_or_update_direct_index(vsc, vs_endpoint_name, vs_index_fullname, vector_search_index_schema, vector_search_index_config):
    '''Create a direct access vector search index if it doesn't exist. If it does exist, update it.'''
    try:
        vsc.create_direct_access_index(
            endpoint_name=vs_endpoint_name,
            index_name=vs_index_fullname,
            schema=vector_search_index_schema,
            **vector_search_index_config
        )
    except Exception as e:
        if 'RESOURCE_ALREADY_EXISTS' not in str(e):
            print(f'Unexpected error...')
            raise e
    wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)
    print(f"index {vs_index_fullname} is ready")


#######################################################################
###### Functions for deploying a chain in Model Serving
#######################################################################
def get_latest_model_version(model_name):
    '''Get the latest model version for a given model name'''
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version
  

def deploy_model_serving_endpoint(
  spark, 
  model_full_name, 
  catalog, 
  logging_schema, 
  endpoint_name, 
  host,
  token,
  ):
    '''Deploy a model serving endpoint'''
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    _config = {
        "served_models": [{
            "model_name": model_full_name,
            "model_version": get_latest_model_version(model_full_name),
            "workload_type": "CPU",
            "workload_size": "Small",
            "scale_to_zero_enabled": "true",
            "environment_vars": {
                    "DATABRICKS_HOST": host,
                    "DATABRICKS_TOKEN": token,
                    "ENABLE_MLFLOW_TRACING": "true",
                }
            }],
        "auto_capture_config": {
            "catalog_name": catalog,
            "schema_name": logging_schema,
            "table_name_prefix": endpoint_name,
            }
        }
    try:
        r = client.get_endpoint(endpoint_name)
        endpoint = client.update_endpoint(
            endpoint="chat",
            config=_config,
            )
    except:
        # Make sure to the schema for the inference table exists
        _ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{logging_schema}")
        # Make sure to drop the inference table it exists
        _ = spark.sql(f"DROP TABLE IF EXISTS {catalog}.{logging_schema}.`{endpoint_name}_payload`")

        endpoint = client.create_endpoint(
            name = endpoint_name,
            config = _config,
            )


def wait_for_model_serving_endpoint_to_be_ready(endpoint_name):
    '''Wait for a model serving endpoint to be ready'''
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
    import time

    # Wait for it to be ready
    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(endpoint_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {endpoint_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
          print('endpoint ready.')
          return
        else:
          break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")


def send_request_to_endpoint(endpoint_name, data):
    '''Send a request to a model serving endpoint'''
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    response = client.predict(endpoint=endpoint_name, inputs=data)
    return response


def delete_model_serving_endpoint(endpoint_name):
    '''Delete a model serving endpoint'''
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    r = client.delete_endpoint(endpoint_name)

  