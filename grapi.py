from fastapi import FastAPI
import gradio as gr

# Define a Gradio interface
def greet(name):
    return f"Hello, {name}!"

gradio_app = gr.Interface(fn=greet, inputs="text", outputs="text")

# Create a FastAPI application
app = FastAPI()

# Mount the Gradio app to the FastAPI app
gr.mount_gradio_app(app, gradio_app, path="/gradio")

# Run the FastAPI application with Uvicorn
# Command: uvicorn your_file_name:app --reload
