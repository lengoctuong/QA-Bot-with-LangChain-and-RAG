import gradio as gr
import requests

# Function to send the file and string to the FastAPI backend
def gradio_function(file, string_input):
    # Open the file for sending
    with open(file.name, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/process/",
            files={"file": f},
            data={"string_input": string_input}
        )
    return response.json()

# Gradio interface
gr_interface = gr.Interface(
    fn=gradio_function,
    inputs=[gr.File(label="Upload a File"), gr.Textbox(label="Enter a String")],
    outputs=gr.JSON(label="Output")
)

# Launch the Gradio app
gr_interface.launch()