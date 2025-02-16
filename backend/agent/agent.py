from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

# Get the project root directory (where .env is located)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
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

    def chat(self, user_message, stream=True):
        """
        Get a response from the chat model (GPT-4)
        Args:
            user_message (str): The user's input message
            stream (bool): Whether to stream the response
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
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            completion = self.chat_client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return completion.choices[0].message.content

    def search(self, query, stream=True):
        """
        Search using Perplexity AI
        Args:
            query (str): The search query
            stream (bool): Whether to stream the response
        Returns:
            If stream=True: Generator yielding response chunks
            If stream=False: Complete response as a string
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an artificial intelligence assistant that helps users "
                    "by providing detailed, accurate information from reliable sources."
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
            for response in response_stream:
                if response.choices[0].delta.content:
                    yield response.choices[0].delta.content
        else:
            response = self.search_client.chat.completions.create(
                model="sonar-pro",
                messages=messages
            )
            return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    agent = Agent()
    
    print("Testing chat:")
    for chunk in agent.chat("What is artificial intelligence?"):
        print(chunk, end='', flush=True)
    print("\n\nTesting search:")
    for chunk in agent.search("What are the latest developments in quantum computing?"):
        print(chunk, end='', flush=True)
    print()
