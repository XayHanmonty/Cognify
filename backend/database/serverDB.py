import chromadb
import os
from dotenv import load_dotenv
from pathlib import Path
from chromadb.utils import embedding_functions
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
logger.info(f"Loading .env from: {env_path.absolute()}")
load_dotenv(env_path)

class MessageDatabase:
    def __init__(self):
        logger.info("Initializing MessageDatabase...")
        
        # Get ChromaDB API key from environment
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        if not chroma_api_key:
            logger.error("CHROMA_API_KEY not found in environment variables")
            raise ValueError("CHROMA_API_KEY not found in .env file")
        logger.info("Found CHROMA_API_KEY in environment")
        
        # Initialize Chroma client with cloud configuration
        try:
            self.client = chromadb.HttpClient(
                ssl=True,
                host='api.trychroma.com',
                tenant='87cbae94-2e8a-47e1-b7d0-1c7d1e6631e4',
                database='TreeHacks 2024',
                headers={
                    'x-chroma-token': chroma_api_key
                }
            )
            logger.info("Connected to ChromaDB cloud instance")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            raise
        
        # Create OpenAI embedding function
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in .env file")
        logger.info("Found OPENAI_API_KEY in environment")
            
        try:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-ada-002"
            )
            logger.info("OpenAI embedding function initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {str(e)}")
            raise

    def create_user_message_collection(self, collection_name="user_messages"):
        """Create or get a collection for storing user messages"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Retrieved existing collection: {collection_name}")
        except Exception as e:
            logger.info(f"Collection {collection_name} not found, creating new one")
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "User messages and agent responses"}
            )
            logger.info(f"Created new collection: {collection_name}")
        return collection

    def add_message(self, user_message: str, agent_response: str, 
                   tags: Optional[List[str]] = None, 
                   sentiment: Optional[str] = None,
                   collection_name: str = "user_messages") -> str:
        """Add a user message and agent response to the collection"""
        logger.info(f"Adding message pair to collection: {collection_name}")
        
        collection = self.create_user_message_collection(collection_name)
        
        # Create a unique ID for the message pair
        message_id = f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Prepare minimal metadata to stay within quota limits
        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "sentiment": sentiment or "neutral"
        }
        
        try:
            # Add the message pair to the collection
            collection.add(
                documents=[f"User: {user_message}\nAgent: {agent_response}"],
                metadatas=[metadata],
                ids=[message_id]
            )
            logger.info(f"Successfully added message with ID: {message_id}")
            return message_id
        except Exception as e:
            logger.error(f"Error adding message to collection: {str(e)}")
            raise

    def query_messages(self, query_text: str, n_results: int = 2, 
                      tags: Optional[List[str]] = None,
                      time_range: Optional[tuple] = None,
                      collection_name: str = "user_messages") -> Dict:
        """
        Query the message collection with filters
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            tags: Optional list of tags to filter by
            time_range: Optional tuple of (start_time, end_time) as datetime objects
            collection_name: Name of the collection to query
            
        Returns:
            Dict containing query results
        """
        logger.info(f"Querying messages in collection: {collection_name}")
        logger.info(f"Query text: {query_text}")
        
        collection = self.create_user_message_collection(collection_name)
        
        # Build where clause for filtering
        where = {}
        if tags:
            where["tags"] = {"$contains": json.dumps(tags)}
        if time_range:
            start_time, end_time = time_range
            where["timestamp"] = {
                "$gte": start_time.isoformat(),
                "$lte": end_time.isoformat()
            }
        
        # Execute query with filters
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where if where else None
        )
        logger.info(f"Query results: {len(results)}")
        return results

    def get_all_messages(self, collection_name: str = "user_messages",
                        time_range: Optional[tuple] = None,
                        tags: Optional[List[str]] = None) -> Dict:
        """
        Get all messages from the collection with optional filters
        
        Args:
            collection_name: Name of the collection to query
            time_range: Optional tuple of (start_time, end_time) as datetime objects
            tags: Optional list of tags to filter by
            
        Returns:
            Dict containing all matching messages
        """
        logger.info(f"Retrieving all messages in collection: {collection_name}")
        
        collection = self.create_user_message_collection(collection_name)
        
        # Build where clause for filtering
        where = {}
        if tags:
            where["tags"] = {"$contains": json.dumps(tags)}
        if time_range:
            start_time, end_time = time_range
            where["timestamp"] = {
                "$gte": start_time.isoformat(),
                "$lte": end_time.isoformat()
            }
            
        results = collection.get(where=where if where else None)
        logger.info(f"Retrieved {len(results['ids'])} messages")
        return results

    def get_message_stats(self, time_range: Optional[tuple] = None,
                         collection_name: str = "user_messages") -> Dict:
        """
        Get statistics about the messages in the collection
        
        Args:
            time_range: Optional tuple of (start_time, end_time) as datetime objects
            collection_name: Name of the collection to analyze
            
        Returns:
            Dict containing statistics about the messages
        """
        logger.info(f"Calculating message statistics for collection: {collection_name}")
        
        messages = self.get_all_messages(collection_name, time_range)
        
        if not messages["ids"]:
            logger.info("No messages found")
            return {"total_messages": 0}
        
        # Calculate statistics
        stats = {
            "total_messages": len(messages["ids"]),
            "avg_word_count": sum(meta["word_count"] for meta in messages["metadatas"]) / len(messages["ids"]),
            "avg_char_count": sum(meta["char_count"] for meta in messages["metadatas"]) / len(messages["ids"]),
            "sentiment_distribution": {},
            "tags_distribution": {}
        }
        
        # Calculate sentiment and tag distributions
        for meta in messages["metadatas"]:
            # Sentiment distribution
            sentiment = meta["sentiment"]
            stats["sentiment_distribution"][sentiment] = stats["sentiment_distribution"].get(sentiment, 0) + 1
            
            # Tags distribution
            tags = json.loads(meta["tags"])
            for tag in tags:
                stats["tags_distribution"][tag] = stats["tags_distribution"].get(tag, 0) + 1
        
        logger.info(f"Calculated message statistics: {stats}")
        return stats

    def delete_messages(self, message_ids: List[str], 
                       collection_name: str = "user_messages") -> None:
        """
        Delete specific messages from the collection
        
        Args:
            message_ids: List of message IDs to delete
            collection_name: Name of the collection to delete from
        """
        logger.info(f"Deleting messages from collection: {collection_name}")
        logger.info(f"Message IDs: {message_ids}")
        
        collection = self.create_user_message_collection(collection_name)
        collection.delete(ids=message_ids)
        logger.info("Messages deleted")

    def delete_old_messages(self, days: int, 
                          collection_name: str = "user_messages") -> int:
        """
        Delete messages older than specified days
        
        Args:
            days: Number of days to keep messages for
            collection_name: Name of the collection to clean up
            
        Returns:
            Number of messages deleted
        """
        logger.info(f"Deleting old messages from collection: {collection_name}")
        logger.info(f"Days to keep: {days}")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        messages = self.get_all_messages(collection_name)
        
        if not messages["ids"]:
            logger.info("No messages found")
            return 0
        
        # Find messages older than cutoff date
        old_message_ids = [
            msg_id for msg_id, meta in zip(messages["ids"], messages["metadatas"])
            if datetime.fromisoformat(meta["timestamp"]) < cutoff_date
        ]
        
        if old_message_ids:
            self.delete_messages(old_message_ids, collection_name)
            logger.info(f"Deleted {len(old_message_ids)} old messages")
        
        return len(old_message_ids)