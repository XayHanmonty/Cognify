from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
from backend.database.serverDB import MessageDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory (where .env is located)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
logger.info(f"Loading .env from: {env_path.absolute()}")
load_dotenv(env_path)

class Agent:
    def __init__(self):
        # Initialize OpenAI client for general chat
        self.openai_client = OpenAI()  # Uses OPENAI_API_KEY from env
        
        # Initialize Perplexity client for search
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in .env file")
        self.search_client = OpenAI(
            api_key=self.perplexity_api_key, 
            base_url="https://api.perplexity.ai"
        )
        
        # Initialize message database
        self.message_db = MessageDatabase()

    def chat(self, user_message, stream=True, tags=None):
        """
        Get a response from the chat model (GPT-4)
        Args:
            user_message (str): The user's input message
            stream (bool): Whether to stream the response
            tags (List[str], optional): Tags to categorize the conversation
        Returns:
            If stream=True: Generator yielding response chunks
            If stream=False: Complete response as a string
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]

        if stream:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                stream=True
            )
            # Collect the full response while streaming
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    yield chunk.choices[0].delta.content
            
            # Analyze sentiment (simple example - you could use a more sophisticated model)
            sentiment = "positive" if any(word in user_message.lower() for word in ["good", "great", "excellent", "amazing"]) else \
                       "negative" if any(word in user_message.lower() for word in ["bad", "terrible", "awful", "poor"]) else \
                       "neutral"
            
            # Store the complete conversation in the database with metadata
            self.message_db.add_message(
                user_message=user_message,
                agent_response=full_response,
                tags=tags,
                sentiment=sentiment
            )
        else:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            response = completion.choices[0].message.content
            
            # Analyze sentiment
            sentiment = "positive" if any(word in user_message.lower() for word in ["good", "great", "excellent", "amazing"]) else \
                       "negative" if any(word in user_message.lower() for word in ["bad", "terrible", "awful", "poor"]) else \
                       "neutral"
            
            # Store the conversation in the database with metadata
            self.message_db.add_message(
                user_message=user_message,
                agent_response=response,
                tags=tags,
                sentiment=sentiment
            )
            return response

    # def get_conversation_history(self, n_recent=5, tags=None):
    #     """
    #     Get recent conversation history
    #     Args:
    #         n_recent (int): Number of recent conversations to retrieve
    #         tags (List[str], optional): Filter by specific tags
    #     Returns:
    #         Dict containing conversation history
    #     """
    #     return self.message_db.get_all_messages(tags=tags)

    # def get_similar_conversations(self, query, n_results=2, tags=None):
    #     """
    #     Find similar past conversations
    #     Args:
    #         query (str): Query to find similar conversations
    #         n_results (int): Number of results to return
    #         tags (List[str], optional): Filter by specific tags
    #     Returns:
    #         Dict containing similar conversations
    #     """
    #     return self.message_db.query_messages(query, n_results=n_results, tags=tags)

    # def get_conversation_stats(self, time_range=None):
    #     """
    #     Get statistics about conversations
    #     Args:
    #         time_range (tuple, optional): Tuple of (start_time, end_time) to analyze
    #     Returns:
    #         Dict containing conversation statistics
    #     """
    #     return self.message_db.get_message_stats(time_range=time_range)

    def search(self, query, stream=True, tags=None):
        """
        Search using Perplexity AI
        Args:
            query (str): The search query
            stream (bool): Whether to stream the response
            tags (List[str], optional): Tags to categorize the search
        Returns:
            If stream=True: Generator yielding response chunks
            If stream=False: Complete response as a string
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent search assistant that retrieves and summarizes relevant information "
                    "based on user queries. Your goal is to provide concise, structured, and fact-based answers.\n\n"
                    "Guidelines:\n"
                    "1. Understand the user's intent and retrieve accurate, up-to-date information.\n"
                    "2. If the query asks for a summary, provide key takeaways in bullet points.\n"
                    "3. If the query is a direct question, provide a clear, well-structured response.\n"
                    "4. If numerical data or statistics are involved, ensure accuracy and cite sources.\n"
                    "5. End responses with a 'Sources' section listing source names (not URLs).\n\n"
                    "Response Format:\n"
                    "• **Summary** (if applicable)\n"
                    "• **Key Points** (bullet points for clarity)\n"
                    "• **Sources:**\n"
                    "  1. Source Name\n"
                    "  2. Source Name\n"
                    "  3. Source Name\n\n"
                    "Adjust responses based on query type while maintaining conciseness and clarity."
                ),
            },
            {"role": "user", "content": query},
        ]

        if stream:
            response_stream = self.search_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                stream=True
            )
            # Collect the full response while streaming
            full_response = ""
            for response in response_stream:
                if response.choices[0].delta.content:
                    chunk = response.choices[0].delta.content
                    full_response += chunk
                    yield chunk
            
            # Store the search query and response
            self.message_db.add_message(
                user_message=query,
                agent_response=full_response,
                tags=["search"] + (tags or []),
                sentiment="neutral"  
            )
        else:
            response = self.search_client.chat.completions.create(
                model="sonar-pro",
                messages=messages
            )
            response_text = response.choices[0].message.content
            
            self.message_db.add_message(
                user_message=query,
                agent_response=response_text,
                tags=["search"] + (tags or []),
                sentiment="neutral"
            )
            return response_text
    
    def summarize(self, query, stream=True):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI-powered summarization assistant that extracts key information from text. "
                    "Follow these guidelines:\n\n"
                    "1. Understand the main ideas and remove unnecessary details.\n"
                    "2. Provide structured summaries based on the requested style:\n"
                    "   - 'bullet': Use concise bullet points.\n"
                    "   - 'paragraph': Write a well-structured paragraph.\n"
                    "3. Adjust summary length as requested:\n"
                    "   - 'short': Focus on the core message (1-2 sentences).\n"
                    "   - 'medium': Cover key points concisely (3-5 sentences).\n"
                    "   - 'long': Provide an in-depth summary (detailed, 7+ sentences).\n"
                    "4. Ensure clarity, accuracy, and coherence.\n"
                ),
            },
            {"role": "user", "content": query},
        ]

        if stream:
            response_stream = self.search_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                stream=True
            )
            # Collect the full response while streaming
            full_response = ""
            for response in response_stream:
                if response.choices[0].delta.content:
                    chunk = response.choices[0].delta.content
                    full_response += chunk
                    yield chunk
            
            # Store the search query and response
            self.message_db.add_message(
                user_message=query,
                agent_response=full_response,
                tags=["summarization"],
                sentiment="neutral"  
            )
        else:
            response = self.search_client.chat.completions.create(
                model="sonar-pro",
                messages=messages
            )
            response_text = response.choices[0].message.content
            
            self.message_db.add_message(
                user_message=query,
                agent_response=response_text,
                tags=["summarization"],
                sentiment="neutral"
            )
            return response_text
   

    def generate_code(self, query, stream=True):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a specialized AI code-generation assistant that helps developers write, "
                    "refactor, and optimize code efficiently. Follow these guidelines:\n\n"
                    "1. Generate clean, modular, and well-documented code.\n"
                    "2. Use best practices and performance optimizations where applicable.\n"
                    "3. If applicable, provide a brief explanation before the code block.\n"
                    "4. Ensure correctness, avoiding unnecessary complexity.\n"
                    "5. Output code in markdown format for clarity.\n\n"
                    "Example:\n"
                    "```python\n"
                    "# This function sorts a list using quicksort\n"
                    "def quicksort(arr):\n"
                    "    if len(arr) <= 1:\n"
                    "        return arr\n"
                    "    pivot = arr[len(arr) // 2]\n"
                    "    left = [x for x in arr if x < pivot]\n"
                    "    middle = [x for x in arr if x == pivot]\n"
                    "    right = [x for x in arr if x > pivot]\n"
                    "    return quicksort(left) + middle + quicksort(right)\n"
                    "```\n\n"
                    "Provide the output accordingly based on the user's request."
                ),
            },
            {"role": "user", "content": query},
        ]

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=stream
        )

        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                    full_response += chunk_text
                    yield chunk_text
            
            # Store the complete response after streaming
            self.message_db.add_message(
                user_message=query,
                agent_response=full_response,
                tags=["code_generate"],
                sentiment="neutral"
            )
        else:
            response_text = response.choices[0].message.content
            
            self.message_db.add_message(
                user_message=query,
                agent_response=response_text,
                tags=["code_generate"],
                sentiment="neutral"
            )
            return response_text

# Example usage
# if __name__ == "__main__":
#     agent = Agent()
    
#     print("Testing chat:")
#     for chunk in agent.chat("What is artificial intelligence?"):
#         print(chunk, end='', flush=True)
#     print("\n\nTesting search:")
#     for chunk in agent.search("What are the latest developments in quantum computing?"):
#         print(chunk, end='', flush=True)
#     print()
