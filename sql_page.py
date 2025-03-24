import os
import shutil
import uvicorn
import gradio as gr
from fastapi import FastAPI
from chatbots.qa_bot import SQL_Bot
from uuid import uuid4

sql_bot = SQL_Bot()

def query_dbs(query, file):
    try:
        for s in sql_bot.graph.stream({"messages": [("user", query)]}, {"configurable": {"thread_id": str(uuid4())}}):
            print(s)

        return s['model_gen_query']['messages'][0].tool_calls[0]['args']['final_answer']
    except Exception as e:
        raise gr.Error(str(e))

# Create Gradio interface
with gr.Blocks() as page:
    gr.Markdown("## SQL Agent")
    gr.Interface(
        fn=query_dbs,
        allow_flagging="never",
        inputs=[
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
            gr.File(label="Upload .sql File", file_count="single", file_types=['.sql'], type="filepath"),  # Drag and drop file upload
        ],
        outputs=gr.Textbox(label="Output"),
        title="SQL-Querying Chatbot",
        description="Upload a .sql file and ask any question. The chatbot will try to answer by querying databases."
    )

    gr.Markdown("## List of tables")
    gr.Markdown("## Table Schemas")
    gr.Markdown("## Diagram")

if __name__ == "__main__":
    page.launch()