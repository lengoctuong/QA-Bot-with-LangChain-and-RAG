# from ibm_watsonx_ai.foundation_models import ModelInference
# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
# from ibm_watsonx_ai import Credentials
# from langchain_ibm import WatsonxLLM, WatsonxEmbeddings

from azureml.core.authentication import ServicePrincipalAuthentication
from langchain_openai import AzureOpenAIEmbeddings, AzureOpenAI, AzureChatOpenAI

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback
from langchain.chains import RetrievalQA

from fastapi import FastAPI, Response
from pydantic import BaseModel

import gradio as gr
import os

# You can use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM
# os.environ["HUGGINGFACEHUB_API_TOKEN"]
# def get_hg_llm():
#     llm = HuggingFaceEndpoint(
#         repo_id="HuggingFaceH4/zephyr-7b-beta",
#         task="text2text-generation",
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     )

#     return llm

# LLM
def get_hg_llm(task="text2text-generation"):
    model_name = "google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text2text_pipeline = pipeline(task, model=model, tokenizer=tokenizer, max_length=2048, device=0)
    llm = HuggingFacePipeline(pipeline=text2text_pipeline)

    return llm

# LLM
def get_az_llm():
    llm = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('OPENAI_API_VERSION'),
        azure_deployment="gpt-35-turbo-instruct",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        max_tokens=20
    )

    return llm

## Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Vector db
def vector_database(chunks):
    embedding_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
        azure_endpoint=os.getenv("AZURE_OPENAI_EMB_ENDPOINT"), # If not provided, will read env variable AZURE_OPENAI_ENDPOINT
        api_key=os.getenv("AZURE_OPENAI_EMB_API_KEY"), # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
        openai_api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION"), # If not provided, will read env variable AZURE_OPENAI_API_VERSION
    )

    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

# Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# QA Chain
def retriever_qa(file, query):
    llm = get_hg_llm()
    retriever_obj = retriever(file)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj, return_source_documents=False,)
    response = qa.invoke(query)

    return response['result']

if __name__ == 'main':
    llm = get_hg_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj, return_source_documents=False,)
    app = FastAPI()

    class Body(BaseModel):
        text: str

    @app.get('/qabot')
    def predict(body: Body):
        response = qa.invoke(body.text)
        return response