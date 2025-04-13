from fastapi import HTTPException
from .qa_bot import LLMService, EmbeddingService
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

class RAG_Bot:
    def __init__(self, llm_service=LLMService('AzureOpenAI', 'DeepSeek-R1-houab'), embedding_service=EmbeddingService()):
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