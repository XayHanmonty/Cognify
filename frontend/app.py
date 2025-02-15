from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sys
from pathlib import Path
import json

# Add the backend directory to Python path
project_root = Path(__file__).parent.parent
backend_dir = project_root / "backend" / "testIntegration"
sys.path.append(str(backend_dir))

from testChat import get_chat_response, get_chat_response_sync

app = Flask(__name__)
CORS(app)

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
        response = get_chat_response_sync(user_message)
        return jsonify({'response': response})
    
    def generate():
        for text_chunk in get_chat_response(user_message):
            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive'})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
