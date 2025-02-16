from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from flask_cors import CORS
import json
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
logger.info(f"Project root: {project_root}")
sys.path.append(str(project_root))
logger.info(f"Python path: {sys.path}")

from backend.agent.agent import Agent

app = Flask(__name__)
CORS(app)

# Initialize the agent as a global variable
logger.info("Initializing Agent...")
agent = Agent()
logger.info("Agent initialized successfully")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    stream = data.get('stream', True)  # Default to streaming
    tags = data.get('tags', None) 

    if not stream:
        # Use non-streaming version and collect the full response
        response = ""
        for chunk in agent.chat(user_message, stream=True, tags=tags):
            response += chunk
        return jsonify({'response': response})
    
    def generate():
        # Use streaming version
        for text_chunk in agent.chat(user_message, stream=True, tags=tags):
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
    tags = data.get('tags', ['search'])  # Always include 'search' tag

    if not stream:
        # Use non-streaming version and collect the full response
        response = ""
        for chunk in agent.search(query, stream=True, tags=tags):
            response += chunk
        return jsonify({'response': response})
    
    def generate():
        # Use streaming version
        for text_chunk in agent.search(query, stream=True, tags=tags):
            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive'})

@app.route('/history', methods=['GET'])
def get_history():
    n_recent = request.args.get('n_recent', default=5, type=int)
    tags = request.args.get('tags', default=None, type=str)
    if tags:
        tags = tags.split(',')
    
    history = agent.get_conversation_history(n_recent=n_recent, tags=tags)
    return jsonify(history)

@app.route('/stats', methods=['GET'])
def get_stats():
    stats = agent.get_conversation_stats()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)
