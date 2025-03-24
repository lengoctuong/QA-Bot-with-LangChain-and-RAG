import os
import shutil
import uvicorn
import gradio as gr
import pdf_page, sql_page
from fastapi import FastAPI

app = FastAPI()

with gr.Blocks() as interface:
    gr.Markdown("# AI Agents for Question-Answer")

    with gr.Tab("PDF Agent"):
        pdf_page.page.render()

    with gr.Tab("SQL Agent"):
        sql_page.page.render()

app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    interface.launch()