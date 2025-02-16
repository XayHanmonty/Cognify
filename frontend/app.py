import sys
import logging
import asyncio
import json
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Python path: {sys.path}")

from backend.controllers.agentRouter import SimpleRouterAgent

app = Flask(__name__)
CORS(app)

# Initialize the agent router
logger.info("Initializing Agent Router...")
agent_router = SimpleRouterAgent()
logger.info("Agent Router initialized successfully")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_message():
    data = request.get_json()
    user_input = data.get('message', '')
    
    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(agent_router.process_message(user_input))
            yield f"data: {json.dumps(result)}\n\n"
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            loop.close()
    
    return Response(stream_with_context(generate()),
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                          'Connection': 'keep-alive'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)
