import os
import shutil
import uvicorn
import gradio as gr
from fastapi import FastAPI
from app.qa_bot import RAG_Bot

rag_bot = RAG_Bot()

def query_docs(query, file):
    try:
        if rag_bot.file_path == None or os.path.basename(rag_bot.file_path) != file:
            rag_bot.update_db(file)

        return rag_bot(query)['result']
    except Exception as e:
        raise gr.Error(str(e))

def see_vdb():
    if rag_bot.vector_db == None:
        return {'metadatas': 'No database created'}

    return {'total_tokens': sum(rag_bot.db_tokens),
            'documents': rag_bot.vector_db._collection.get()['documents'],
            'metadatas': rag_bot.vector_db._collection.get()['metadatas'],}

# Create Gradio interface
with gr.Blocks() as page:
    gr.Markdown("## PDF Agent")
    gr.Interface(
        fn=query_docs,
        allow_flagging="never",
        inputs=[
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
            gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        ],
        outputs=gr.Textbox(label="Output"),
        title="PDF-Querying Chatbot",
        description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
    )

    gr.Markdown("## Vector DB")
    gr.Button("Vector Database").click(fn=see_vdb, outputs=gr.JSON(label="JSON Output"))
    

if __name__ == "__main__":
    page.launch()