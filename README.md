# NeuroParallel.ai - Intelligent AI Research Assistant

NeuroParallel.ai is an advanced AI research assistant platform that helps analyze and summarize research papers, generate code, and identify emerging trends in AI. The platform combines OpenAI's GPT-4 for sophisticated analysis with a powerful task decomposition system, providing users with comprehensive research insights and practical implementations.

## Features

### Current Features
- **Intelligent Task Decomposition**
  - Automatically breaks down complex research tasks
  - Handles multiple subtasks in parallel
  - Combines results coherently
- **Research Analysis**
  - Analyzes latest AI research papers
  - Identifies emerging trends and technologies
  - Provides citation-backed insights
- **Code Generation**
  - Creates implementation scripts
  - Handles data scraping and processing
  - Follows best practices and includes documentation
- **Modern UI/UX**
  - Real-time streaming responses
  - Syntax-highlighted code blocks
  - Clean, responsive design
  - Mobile-friendly interface

### Upcoming Features
- PDF RAG (Retrieval-Augmented Generation) System
- Direct paper downloading and processing
- Collaborative research sessions
- Custom research focus areas

## Tech Stack

### Frontend
- HTML5/CSS3/JavaScript
- Flask Templates
- Server-Sent Events (SSE) for streaming
- Syntax highlighting for code blocks
- Material Icons
- Inter Font Family

### Backend
- Python 3.9+
- Flask with async support
- ChromaDB for message storage
- OpenAI GPT-4
- Task decomposition system

### Project Structure
```
NeuroParallel.ai/
├── frontend/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── css/
│   │   │   ├── style.css
│   │   │   └── message-styles.css
│   │   └── js/
│   │       └── main.js
│   └── app.py
├── backend/
│   ├── agent/
│   │   └── agent.py
│   ├── controllers/
│   │   ├── agentController.py
│   │   └── agentRouter.py
│   └── database/
│       └── serverDB.py
└── .env
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/XayHanmonty/NeuroParallel.ai.git
   cd NeuroParallel.ai
   ```

2. Set up your environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   CHROMA_API_KEY=your_chroma_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask server:
   ```bash
   PYTHONPATH=/path/to/NeuroParallel.ai python3 frontend/app.py
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

1. **Research Analysis**
   - Enter your research question or topic
   - Receive a comprehensive analysis of recent papers
   - View emerging trends with citations

2. **Code Generation**
   - Request implementation of research concepts
   - Get fully documented code with best practices
   - Code is syntax-highlighted and easily copyable

## Recent Updates

### Backend Improvements
- Implemented task decomposition system
- Added ChromaDB integration for message storage
- Upgraded to GPT-4 for enhanced analysis
- Improved async handling in Flask routes

### Frontend Enhancements
- Added syntax highlighting for code blocks
- Improved message formatting and styling
- Enhanced UI for research results
- Better error handling and user feedback

### Code Quality
- Refactored for better maintainability
- Improved error handling and logging
- Added comprehensive documentation
- Enhanced type hints and comments

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
