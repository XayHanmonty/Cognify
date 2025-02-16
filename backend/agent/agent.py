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
        self.chat_client = OpenAI()  # Uses OPENAI_API_KEY from env
        
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
            completion = self.chat_client.chat.completions.create(
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
            completion = self.chat_client.chat.completions.create(
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

    def get_conversation_history(self, n_recent=5, tags=None):
        """
        Get recent conversation history
        Args:
            n_recent (int): Number of recent conversations to retrieve
            tags (List[str], optional): Filter by specific tags
        Returns:
            Dict containing conversation history
        """
        return self.message_db.get_all_messages(tags=tags)

    def get_similar_conversations(self, query, n_results=2, tags=None):
        """
        Find similar past conversations
        Args:
            query (str): Query to find similar conversations
            n_results (int): Number of results to return
            tags (List[str], optional): Filter by specific tags
        Returns:
            Dict containing similar conversations
        """
        return self.message_db.query_messages(query, n_results=n_results, tags=tags)

    def get_conversation_stats(self, time_range=None):
        """
        Get statistics about conversations
        Args:
            time_range (tuple, optional): Tuple of (start_time, end_time) to analyze
        Returns:
            Dict containing conversation statistics
        """
        return self.message_db.get_message_stats(time_range=time_range)

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
                    "You are an artificial intelligence assistant that helps users "
                    "by providing detailed, accurate information. Format your response as follows:\n\n"
                    "1. Present information in a clean, consistent format with proper spacing\n"
                    "2. For weather forecasts:\n"
                    "   • Present each day on a new line\n"
                    "   • Format temperatures as: High: XX°F • Low: XX°F\n"
                    "   • Present conditions after temperatures\n"
                    "   • Use bullet points (•) to separate different pieces of information\n"
                    "3. Keep URLs hidden - use only source names in the Sources section\n"
                    "4. End with a 'Sources' section that lists source names (not URLs)\n\n"
                    "Example weather format:\n"
                    "Wednesday, February 14, 2024\n"
                    "High: 56°F • Low: 49°F • Conditions: Rain likely\n"
                    "Wind: Southwest, 10-20 mph with gusts up to 35 mph\n\n"
                    "Sources:\n"
                    "1. Weather.gov\n"
                    "2. National Weather Service"
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
                sentiment="neutral"  # Search queries typically don't have sentiment
            )
        else:
            response = self.search_client.chat.completions.create(
                model="sonar-pro",
                messages=messages
            )
            response_text = response.choices[0].message.content
            
            # Store the search query and response
            self.message_db.add_message(
                user_message=query,
                agent_response=response_text,
                tags=["search"] + (tags or []),
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
