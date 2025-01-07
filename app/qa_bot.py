__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
import tiktoken

from fastapi import HTTPException
from langchain_openai import AzureOpenAIEmbeddings, AzureOpenAI
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

class LLMService:
    '''
    provider in ['AzureOpenAI', 'HuggingFace']
    model_name: e.g 'gpt-35-turbo-instruct', 'google/flan-t5-large',...
    '''

    def __init__(self, provider='HuggingFace', model_name='google/flan-t5-large', max_gen_tokens=512, task='text2text-generation', local=True):
        self.provider = provider
        self.model_name = model_name
        self.max_gen_tokens = max_gen_tokens

        try:
            if provider == 'AzureOpenAI':
                self.model = AzureOpenAI(
                    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                    api_version=os.getenv('OPENAI_API_VERSION'),
                    azure_deployment=model_name,
                    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                    max_tokens=max_gen_tokens
                )
            # elif provider == 'HuggingFace':
            #     if local:
            #         tokenizer = AutoTokenizer.from_pretrained(model_name)
            #         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            #         text2text_pipeline = pipeline(task, model=model, tokenizer=tokenizer)
            #         self.model = HuggingFacePipeline(pipeline=text2text_pipeline)
            #     else:
            #         self.model = HuggingFaceEndpoint(repo_id=model_name, task=task, max_new_tokens=max_gen_tokens)
            else:
                raise HTTPException(status_code=500, detail='Provider got invalid.')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class EmbeddingService:
    def __init__(self, model_name='text-embedding-3-large'):
        self.model_name = model_name

        try:
            self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_for_model(model_name).name)
            self.model = AzureOpenAIEmbeddings(
                model=model_name,
                azure_endpoint=os.getenv('AZURE_OPENAI_EMB_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_EMB_API_KEY'),
                openai_api_version=os.getenv('AZURE_OPENAI_EMB_API_VERSION')
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class QA_Bot:
    def __init__(self, llm_service=LLMService('AzureOpenAI', 'gpt-35-turbo-instruct'), embedding_service=EmbeddingService()):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.file_path = None
        self.db_tokens = []
        self.vector_db = None
        self.retrieval_chain = None

    def update_db(self, file_path, chunk_size=1000, chunk_overlap=50, length_function=len):
        self.file_path = file_path
        try:
            # Document loader
            loader = PyPDFLoader(file_path)
            loaded_document = loader.load()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        chunks = text_splitter.split_documents(loaded_document)
        self.db_tokens = [len(self.embedding_service.tokenizer.encode(chunk.page_content)) for chunk in chunks]

        # Vector db
        if self.vector_db != None:
            self.vector_db._collection.delete(self.vector_db._collection.get()['ids'])
        self.vector_db = Chroma.from_documents(chunks, self.embedding_service.model)

        # Retrieval chain
        retriever = self.vector_db.as_retriever()
        self.retrieval_chain = RetrievalQA.from_chain_type(llm=self.llm_service.model, chain_type='stuff', retriever=retriever, return_source_documents=False)

    def __call__(self, query):
        response = self.retrieval_chain.invoke(query)
        response['document'] = self.file_path
        return response