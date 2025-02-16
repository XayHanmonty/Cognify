# NeuroParallel.ai - Intelligent AI Assistant Platform

NeuroParallel.ai is an advanced AI assistant platform that combines powerful language models with specialized search capabilities. The platform integrates OpenAI's GPT-4 for general conversation and Perplexity AI for enhanced search functionality, providing users with a versatile tool for both chat and research purposes.

## Features

### Current Features
- **Dual-Mode Interface**
  - Chat Mode: Engage in natural conversations using GPT-4
  - Search Mode: Get detailed, citation-backed answers using Perplexity AI
- **Real-time Streaming**
  - Character-by-character response streaming
  - Typing indicator animation
  - Smooth scrolling and auto-resizing input
- **Modern UI/UX**
  - Clean, responsive design
  - Easy mode switching
  - Mobile-friendly interface

### Upcoming Features
- PDF RAG (Retrieval-Augmented Generation) System
- Task Decomposition and Orchestration
- Agent-based Problem Solving
- Result Aggregation

## Tech Stack

### Frontend
- HTML5/CSS3/JavaScript
- Flask Templates
- Server-Sent Events (SSE) for streaming
- Material Icons
- Inter Font Family

### Backend
- Python 3.9+
- Flask
- Flask-CORS
- OpenAI API
- Perplexity AI API

### Project Structure
```
NeuroParallel.ai/
├── frontend/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── main.js
│   └── app.py
├── backend/
│   ├── agent/
│   │   └── agent.py
│   └── agent_summarizer/
│       └── pdf_processor.py
└── .env
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NeuroParallel.ai.git
   cd NeuroParallel.ai
   ```

2. Set up your environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask server:
   ```bash
   python frontend/app.py
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

1. **Chat Mode**
   - Click the "Chat" button in the navbar
   - Type your message and press Enter or click Send
   - Receive streamed responses from GPT-4

2. **Search Mode**
   - Click the "Search" button in the navbar
   - Enter your search query
   - Get detailed, citation-backed responses from Perplexity AI

## Development Status

The project is actively under development. Current focus areas:
- Integration of PDF processing capabilities
- Implementation of agent-based task decomposition
- Enhancement of search functionality
- Addition of conversation memory and context

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
