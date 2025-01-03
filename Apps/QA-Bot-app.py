__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

from fastapi import FastAPI, Response, File, UploadFile, Form
from pydantic import BaseModel

import gradio as gr
import os
import shutil

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
    model_name = "google/flan-t5-large"
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
        max_tokens=128
    )

    return llm

## Document loader
def document_loader(filename):
    loader = PyPDFLoader(filename)
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
    global vectordb

    embedding_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
        azure_endpoint=os.getenv("AZURE_OPENAI_EMB_ENDPOINT"), # If not provided, will read env variable AZURE_OPENAI_ENDPOINT
        api_key=os.getenv("AZURE_OPENAI_EMB_API_KEY"), # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
        openai_api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION"), # If not provided, will read env variable AZURE_OPENAI_API_VERSION
    )

    if vectordb != None:
        vectordb._collection.delete(vectordb._collection.get()['ids'])
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

# Retriever
def retriever(filename):
    splits = document_loader(filename)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# QA Chain
def retriever_qa(filename, query):
    llm = get_hg_llm()
    retriever_obj = retriever(filename)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj, return_source_documents=False,)
    response = qa.invoke(query)

    return response['result']

vectordb = None
llm = get_hg_llm()
cur_filename = ""
retriever_obj = None
qa = None
app = FastAPI()

@app.get('/')
def root():
    return Response('Chat Bot answering any questions from documents you upload')

@app.post('/qabot')
def predict(question: str = Form(...), file: UploadFile = File(...)):
    global llm, cur_filename, retriever_obj, qa

    # Copy to repo
    upload_folder = "/workspaces/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    new_filename = os.path.join(upload_folder, file.filename)

    with open(new_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Use file uploaded to create retriever object
    if retriever_obj == None or cur_filename != new_filename:
        cur_filename = new_filename
        retriever_obj = retriever(new_filename)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj, return_source_documents=False,)

    response = qa.invoke(question)
    response['document'] = cur_filename
    return response

@app.get('/database')
def get_database():
    if vectordb == None:
        return {'message': 'No database created'}
    return vectordb._collection.get()

# uvicorn --host 0.0.0.0 --port 7890 QA-Bot-app:app
# uvicorn --host localhost --port 7890 QA-Bot-app:app