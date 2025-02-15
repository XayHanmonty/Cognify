from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Get the project root directory (where .env is located)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

client = OpenAI()  # The API key will be automatically read from environment variable OPENAI_API_KEY

def get_chat_response(user_message):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        stream=True 
    )
    
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def get_chat_response_sync(user_message):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )
    return completion.choices[0].message.content

##TEST##
if __name__ == "__main__":
    # Test the streaming function
    for text_chunk in get_chat_response("Hello!"):
        print(text_chunk, end='', flush=True)
