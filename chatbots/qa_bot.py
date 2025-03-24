__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os, tiktoken, re, ast, hashlib

from typing import Any, Annotated, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from IPython.display import Image, display
from uuid import uuid4
from fastapi import HTTPException

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents.agent_toolkits import create_retriever_tool

from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class LLMService:
    '''
    provider in ['AzureOpenAI', 'Google', 'HuggingFace']
    model_name: e.g 'gpt-35-turbo', 'DeepSeek-R1-houab', gemini-2.0-flash, 'google/flan-t5-large',...
    '''

    def __init__(self, provider='AzureOpenAI', model_name='DeepSeek-R1-houab', max_gen_tokens=512, task='text2text-generation', local=True):
        self.provider = provider
        self.model_name = model_name
        self.max_gen_tokens = max_gen_tokens

        try:
            if provider == 'AzureOpenAI':
                if model_name == "gpt-35-turbo-16k":
                    self.model = AzureChatOpenAI(api_version="2024-08-01-preview", azure_deployment=model_name, max_tokens=max_gen_tokens)
                elif model_name == "DeepSeek-R1-houab":
                    self.model = AzureAIChatCompletionsModel(model_name=model_name, max_tokens=max_gen_tokens)
                else:
                    raise HTTPException(status_code=500, detail='Model name is not supported.')
            elif provider == 'Google':
                self.model = ChatGoogleGenerativeAI(model=model_name, max_tokens=max_gen_tokens)
            # elif provider == 'HuggingFace':
            #     if local:
            #         tokenizer = AutoTokenizer.from_pretrained(model_name)
            #         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            #         text2text_pipeline = pipeline(task, model=model, tokenizer=tokenizer)
            #         self.model = HuggingFacePipeline(pipeline=text2text_pipeline)
            #     else:
            #         self.model = HuggingFaceEndpoint(repo_id=model_name, task=task, max_new_tokens=max_gen_tokens)
            else:
                raise HTTPException(status_code=500, detail='Provider is not supported.')
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
                openai_api_version="2023-05-15"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

class SQL_Bot:
    def __init__(self, db=None, num_try=2):
        if db == None:
            self.db = SQLDatabase.from_uri("sqlite:///Chinook.db")
            self.dialect = self.db.dialect
        else:
            self.db = db

        self.num_try = num_try

        # Setup tools
        toolkit = SQLDatabaseToolkit(db=self.db, llm=AzureChatOpenAI(
            azure_deployment="gpt-35-turbo-16k",
            openai_api_version="2024-08-01-preview",
            temperature=0))
        tools = toolkit.get_tools()

        self.list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
        self.get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
        self.query_db_tool = tool("sql_db_query", self.query_db_func)

        description = "Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is valid proper nouns. Use the noun most similar to the search."
        self.retrieve_noun_tool = create_retriever_tool(self.create_proper_noun_retriever(), name="sql_db_retrieve_proper_nouns", description=description)

        # Chain that is for a model to choose the relevant tables based on the question and available tables
        schema_get_system = f"""Given an input question to {self.dialect} and a list of tables. You will choose tables that are relevant to the question, after that you will call the 'sql_db_schema' tool.
        Your output can only be a tool call of 'sql_db_schema'.
        """

        self.schema_get = ChatPromptTemplate.from_messages(
                [("system", schema_get_system), ("placeholder", "{messages}")]
            ) | AzureChatOpenAI(
                azure_deployment="gpt-35-turbo-16k",
                openai_api_version="2024-08-01-preview",
                temperature=0).bind_tools(
                [self.get_schema_tool], tool_choice="required"
            )

        # Chain that is for a model to make decision wheather to have a proper noun in the question and use search tool for finding the exact noun
        noun_retrieve_system = f"""Given an input question to {self.dialect}. You will check whether there are any proper nouns in the question:
        - If the question requires filtering on a proper noun like a Name, you must ALWAYS first look up the filter value using the 'sql_db_retrieve_proper_nouns' tool! Do not try to guess at the proper name - use this function to find similar ones.
        - In contrast, if the question does not require filtering on a proper noun, you will generate a syntactically correct {self.dialect} query for the question based schemas of relevant tables using the 'sql_db_query' tool.

        Your output can only be a tool call of 'sql_db_retrieve_proper_nouns' or 'sql_db_query'."""

        self.noun_retrieve = ChatPromptTemplate.from_messages(
                [("system", noun_retrieve_system), ("placeholder", "{messages}")]
            ) | AzureChatOpenAI(
                azure_deployment="gpt-35-turbo-16k",
                openai_api_version="2024-08-01-preview",
                temperature=0).bind_tools(
                [self.retrieve_noun_tool, self.query_db_tool], tool_choice="required"
            )

        # Chain that is for a model to generate a query based on the question and schemas
        query_gen_system = f"""Given an input question to {self.dialect}, and schemas of relevant tables. You will do one of the following jobs:
        - If the most recent message is database schemas, you should output a syntactically correct {self.dialect} query for the question.
        - If you are given a query error from the database, correct the wrong {self.dialect} query.
        - If you are given a result of the right query and have enough information to answer the input question, call the SubmitFinalAnswer to answer the question with the information you have for the user.

        Rules for a query:
        - Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most five results.
        - You can order the results by a relevant column to return the most interesting examples in the database.
        - Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        - If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
        - Answer the user after {self.num_try} times of try, the answer can be a true result from database running or an error.

        DO NOT:
        - NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.
        - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        """

        self.query_gen = ChatPromptTemplate.from_messages(
                [("system", query_gen_system), ("placeholder", "{messages}")]
            ) | AzureChatOpenAI(
                azure_deployment="gpt-35-turbo-16k",
                openai_api_version="2024-08-01-preview",
                temperature=0).bind_tools(
                [self.query_db_tool, self.SubmitFinalAnswer], tool_choice="required"
            )

        # Define a graph
        graph = StateGraph(State)
        graph.add_node("first_tool_call", self.first_tool_call)
        graph.add_node("list_tables_tool", create_tool_node_with_fallback([self.list_tables_tool]))
        graph.add_node("model_get_schema", self.model_get_schema)
        graph.add_node("get_schema_tool", create_tool_node_with_fallback([self.get_schema_tool]))
        graph.add_node("model_retrieve_noun", self.model_retrieve_noun)
        graph.add_node("retrieve_noun_tool", create_tool_node_with_fallback([self.retrieve_noun_tool]))
        graph.add_node("model_gen_query", self.model_gen_query)
        graph.add_node("query_db_tool", create_tool_node_with_fallback([self.query_db_tool]))
        graph.add_node("count_retry_call", self.count_retry_call)

        graph.add_edge(START, "first_tool_call")
        graph.add_edge("first_tool_call", "list_tables_tool")
        graph.add_edge("list_tables_tool", "model_get_schema")
        graph.add_edge("model_get_schema", "get_schema_tool")
        graph.add_edge("get_schema_tool", "model_retrieve_noun")
        graph.add_conditional_edges("model_retrieve_noun", self.have_proper_noun)
        graph.add_edge("retrieve_noun_tool", "model_gen_query")
        graph.add_conditional_edges("model_gen_query", self.should_continue)
        graph.add_edge("query_db_tool", "count_retry_call")
        graph.add_edge("count_retry_call", "model_gen_query")

        # Compile the graph into a runnable
        memory = MemorySaver()
        self.graph = graph.compile(checkpointer=memory)

    # Retriever to retrieve proper nouns for high-cardinality
    def create_proper_noun_retriever(self):
        def query_as_list(db, query):
            res = db.run(query)
            res = [el for sub in ast.literal_eval(res) for el in sub if el]
            res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
            return list(set(res))

        artists = query_as_list(self.db, "SELECT Name FROM Artist")
        albums = query_as_list(self.db, "SELECT Title FROM Album")
        genres = query_as_list(self.db, "SELECT Name FROM Genre")

        embeddings = AzureOpenAIEmbeddings(
            model='text-embedding-3-large',
            azure_endpoint=os.getenv("AZURE_OPENAI_EMB_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_EMB_API_KEY"),
            openai_api_version="2023-05-15"
        )

        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_texts(artists + albums + genres)
        return vector_store.as_retriever(search_kwargs={"k": 5})

    # Setup custom query tool
    def query_db_func(self, query: str) -> str:
        """
        Execute a SQL query against the database and get back the result.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again.
        """
        result = self.db.run_no_throw(query)
        if not result:
            return "Warning: Empty result. You should try to rewrite the query to get a non-empty result set."
        return result

    # Setup custom tool to represent the end state
    class SubmitFinalAnswer(BaseModel):
        """Submit the final answer to the user based on the query results."""
        final_answer: str = Field(..., description="The final answer to the user")

    def first_tool_call(self, state: State) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content="", tool_calls=[{
                    "name": "sql_db_list_tables", "args": {}, "id": str(uuid4())
                }])]}

    def model_get_schema(self, state: State):
        return {"messages": [self.schema_get.invoke({"messages": state["messages"]})]}
    
    def model_retrieve_noun(self, state: State) -> dict[str, list[AIMessage]]:
        return {"messages": [self.noun_retrieve.invoke({"messages": state["messages"]})]}

    def have_proper_noun(self, state: State) -> Literal["retrieve_noun_tool", "query_db_tool"]:
        messages = state["messages"]
        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            if last_message.tool_calls[0]["name"] == "sql_db_retrieve_proper_nouns":
                return "retrieve_noun_tool"
            if last_message.tool_calls[0]["name"] == "sql_db_query":
                return "query_db_tool"
            
    def model_gen_query(self, state: State):
        return {"messages": [self.query_gen.invoke({"messages": state["messages"]})]}
    
    # Define a conditional edge to decide whether to continue or end the workflow
    def should_continue(self, state: State) -> Literal[END, "query_db_tool"]:
        messages = state["messages"]
        last_message = messages[-1]
                
        # If there is a tool call, then we finish
        if getattr(last_message, "tool_calls", None):
            if last_message.tool_calls[0]["name"] == "SubmitFinalAnswer":
                return END
            if last_message.tool_calls[0]["name"] == "sql_db_query":
                return "query_db_tool"
        
    def count_retry_call(self, state: State):
        count = 0
        for mess in state["messages"]:
            if getattr(mess, "tool_calls", None):
                if mess.tool_calls[0]["name"] == "sql_db_query":
                    count += 1
        
        if count == 2:
            return {"messages": [HumanMessage(content=f"You have reached {count} times of try. Please call the SubmitFinalAnswer tool!")]}
        
    def show_graph(self):
        display(Image(self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)))