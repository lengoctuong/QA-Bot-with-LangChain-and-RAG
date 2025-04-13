import os
import shutil
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Response, Form, File, UploadFile
from chatbots.qa_bot import RAG_Bot, SQL_Bot

router = APIRouter()
rag_bot = RAG_Bot()
sql_bot = SQL_Bot()

@router.get('/')
def root():
    return Response('Chat Bot answering any questions from documents you upload')

@router.post('/query-docs')
def query_docs(query: str=Form(...), file: UploadFile=File(...)):
    try:
        if rag_bot.file_path == None or os.path.basename(rag_bot.file_path) != file.filename:
            upload_folder = '/tmp/qabot-app'
            os.makedirs(upload_folder, exist_ok=True)
            new_file_path = os.path.join(upload_folder, file.filename)

            with open(new_file_path, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)

            rag_bot.update_db(new_file_path)

        return rag_bot(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/query-dbs')
def query_dbs(query: str=Form(...)):
    try:
        for s in sql_bot.graph.stream({"messages": [("user", query)]}, {"configurable": {"thread_id": str(uuid4())}}):
            print(s)

        return s['model_gen_query']['messages'][0].tool_calls[0]['args']['final_answer']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/database')
def get_db_info():
    if rag_bot.vector_db == None:
        return {'metadatas': 'No database created'}

    return {'total_tokens': sum(rag_bot.db_tokens),
            'documents': rag_bot.vector_db._collection.get()['documents'],
            'metadatas': rag_bot.vector_db._collection.get()['metadatas'],}