from flask import Flask, request, render_template
import json

# Web App Source
# https://github.com/ibm-developer-skills-network/LLM_application_chatbot

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/ragbot', methods=['POST'])
def handle_prompt_with_rag():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    return json.dumps({'response': 'Hello, World!'})

if __name__ == '__main__':
    app.run()