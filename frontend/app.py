from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sys
from pathlib import Path
import json
import os

# Add the backend directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agent.agent import Agent

app = Flask(__name__)
CORS(app)

# Initialize the agent
agent = Agent()

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    stream = data.get('stream', True)  # Default to streaming

    if not stream:
        # Use synchronous version for non-streaming requests
        response = agent.chat(user_message, stream=False)
        return jsonify({'response': response})
    
    def generate():
        # Use asynchronous version for streaming
        for text_chunk in agent.chat(user_message, stream=True):
            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive'})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    stream = data.get('stream', True)  # Default to streaming

    if not stream:
        # Use synchronous version for non-streaming requests
        response = agent.search(query, stream=False)
        return jsonify({'response': response})
    
    def generate():
        # Use asynchronous version for streaming
        for text_chunk in agent.search(query, stream=True):
            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive'})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
