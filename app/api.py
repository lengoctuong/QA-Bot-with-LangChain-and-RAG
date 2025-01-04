import os
import shutil
from fastapi import APIRouter, HTTPException, Response, Form, File, UploadFile
from qa_bot import QA_Bot

router = APIRouter()
qa_bot = QA_Bot()

@router.get('/')
def root():
    return Response('Chat Bot answering any questions from documents you upload')

@router.post('/query')
def query_llm(query: str=Form(...), file: UploadFile=File(...)):
    try:
        if qa_bot.file_path == None or os.path.basename(qa_bot.file_path) != file.filename:
            upload_folder = '/tmp/qabot-app'
            os.makedirs(upload_folder, exist_ok=True)
            new_file_path = os.path.join(upload_folder, file.filename)

            with open(new_file_path, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)

            qa_bot.update_db(new_file_path)

        return qa_bot(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/database')
def get_db_info():
    if qa_bot.vector_db == None:
        return {'metadatas': 'No database created'}

    return {'total_tokens': sum(qa_bot.db_tokens),
            'documents': qa_bot.vector_db._collection.get()['documents'],
            'metadatas': qa_bot.vector_db._collection.get()['metadatas'],}