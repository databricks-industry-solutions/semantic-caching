from databricks.vector_search.client import VectorSearchClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from config import Config
import mlflow
import os

# Enable MLflow Tracing
mlflow.langchain.autolog()

# load parameters
config = Config()

# Connect to the Vector Search Index
vs_index = VectorSearchClient(
    workspace_url=os.environ['DATABRICKS_HOST'],
    personal_access_token=os.environ['DATABRICKS_TOKEN'],
    disable_notice=True,
    ).get_index(
    endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=config.VS_INDEX_FULLNAME,
)

# Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="content",
    columns=["id", "content", "url"],
).as_retriever(search_kwargs={"k": 3}) # Number of search results that the retriever returns
# Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
mlflow.models.set_retriever_schema(primary_key="id", text_column="content", doc_uri="url")

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

# Prompt template to be used to prompt the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"{config.LLM_PROMPT_TEMPLATE}"),
        ("user", "{question}"),
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=config.LLM_MODEL_SERVING_ENDPOINT_NAME,
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)
