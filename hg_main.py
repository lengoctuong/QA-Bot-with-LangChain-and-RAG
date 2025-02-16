import os
import shutil
import gradio as gr
from app.qa_bot import RAG_Bot

rag_bot = RAG_Bot()

def query_llm(query, file):
    try:
        if rag_bot.file_path == None or os.path.basename(rag_bot.file_path) != file:
            rag_bot.update_db(file)

        return rag_bot(query)
    except Exception as e:
        raise gr.Error(str(e))

# Create Gradio interface
rag_application = gr.Interface(
    fn=query_llm,
    allow_flagging="never",
    inputs=[
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
rag_application.launch()