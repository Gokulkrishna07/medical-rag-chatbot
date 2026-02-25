import time
import mlflow
from mlflow_config import *

from langchain_ollama import OllamaLLM
from langchain.vectorstores import Chroma

# Start MLflow run
with mlflow.start_run():

    model_name = "llama3"
    top_k = 3

    mlflow.log_param("llm_model", model_name)
    mlflow.log_param("top_k", top_k)

    llm = OllamaLLM(model=model_name)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )

    query = input("Ask Question: ")

    start = time.time()

    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    response = llm.invoke(
        f"Context:{context}\nQuestion:{query}"
    )

    latency = time.time() - start

    mlflow.log_metric("latency", latency)
    mlflow.log_text(query, "query.txt")
    mlflow.log_text(response, "response.txt")

    print(response)