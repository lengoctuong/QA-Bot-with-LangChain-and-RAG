import gradio as gr
import pdf_page, sql_page
from fastapi import FastAPI

# #### code improving
# - unit testing
# - makefile
# - deploying docker (dockerfile)
# - updating hg space

# #### general bot impoving
# - interact (memory)
# - app (api, interface (html, css, js, flask))
# - print info: gen tks
# - handle over access

# #### qa-bot
# - print info: max embed tks
# - print error: over max embed tks

# #### sql-bot
# - use deepseek-r1
# - loi question: alis in chain, Big Ones, describe playlist track, cau hoi khong ro rang
# - interact (agent check need to query db,...)
# - MySQL, databricks, snowflake
# - csv
# - query check

# #### extending
# - voice
# - website
# - multi-file rag

app = FastAPI()

with gr.Blocks() as interface:
    gr.Markdown("# AI Agents for Question-Answer")

    with gr.Tab("PDF Agent"):
        pdf_page.page.render()

    with gr.Tab("SQL Agent"):
        sql_page.page.render()

app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    interface.launch(debug=True)