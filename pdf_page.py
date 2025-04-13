import os
import gradio as gr
from uuid import uuid4
from chatbots.mmrag_bot import RAG_Bot

rag_bot = RAG_Bot()
chat_history = []
config = {}
last_response_refs = {'context': 'There is no use RAG.'}

def query_docs(query, file=None):
    global config, last_response_refs
    try:
        if file:
            if rag_bot.file_path == None or os.path.basename(rag_bot.file_path) != file:
                rag_bot.update_db(file)

        # Get response
        config = {"configurable": {"thread_id": str(uuid4())}}
        # for state in  rag_bot.graph.stream({'question': query, 'history': []}, config):
        #     pass
        response = rag_bot.graph.invoke({'question': query, 'history': [], 'last_response_refs': last_response_refs}, config)['history'][-1].content
        last_response_refs = rag_bot.graph.get_state(config).values['last_response_refs']

        # Append to chat history
        chat_history.append((query, response))

        return chat_history # Return updated chat history
    except Exception as e:
        raise gr.Error(str(e))

def clear_chat():
    """Clears the chat history but does NOT affect uploaded documents."""
    global chat_history
    chat_history = []
    return []

def see_vdb():
    if rag_bot.vector_db == None:
        return {'metadatas': 'No database created'}

    return {'total_tokens': sum(rag_bot.db_tokens),
            'documents': rag_bot.vector_db._collection.get()['documents'],
            'metadatas': rag_bot.vector_db._collection.get()['metadatas'],}

# Create Gradio interface
with gr.Blocks() as page:
    gr.Markdown("## PDF Chatbot: Ask Questions from PDFs")

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            chatbox = gr.Chatbot(label="Chat with PDF")
            user_input = gr.Textbox(label="Your Question", placeholder="Ask something about the document...")
            
            with gr.Row():
                send_button = gr.Button("Send")
                clear_button = gr.Button("Clear Chat")

            gr.Button("See References").click(fn=lambda: last_response_refs, outputs=gr.JSON(label="References"))
            gr.Button("See Memories").click(fn=lambda: rag_bot.memory.load_memory_variables({}), outputs=gr.JSON(label="Memories"))

    # Connect Buttons
    send_button.click(query_docs, inputs=[user_input, pdf_input], outputs=[chatbox])
    clear_button.click(clear_chat, outputs=[chatbox])
    
    # gr.Interface(
    #     fn=query_docs,
    #     allow_flagging="never",
    #     inputs=[
    #         gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
    #         gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
    #     ],
    #     outputs=gr.Textbox(label="Output"),
    #     title="PDF-Querying Chatbot",
    #     description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
    # )

    gr.Markdown("## Vector DB")
    gr.Button("Vector Database").click(fn=see_vdb, outputs=gr.JSON(label="JSON Output"))
    

if __name__ == "__main__":
    page.launch()