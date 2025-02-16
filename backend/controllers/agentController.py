from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import warnings
import os
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from backend.controllers.query_classifier import QueryClassifier
import uuid

# Load the spacy model for NLP tasks
# nlp = spacy.load("en_core_web_sm")

# Schema Definitions
class SubQuery(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{id(object())}")
    task_type: str = Field(
        description="Type of task to be executed",
        examples=["summarization", "research_idea_generation", "data_analysis", "code_generation"]
    )
    sub_query: str = Field(
        description="Atomic task for specialized agent execution"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific parameters for the task"
    )
    search_type: str = Field(
        default="closed",
        description="Search environment type: 'closed' (OpenAI) or 'web' (Perplexity)",
        examples=["closed", "web"]
    )

class AgentController:
    def __init__(self, model_name: str = "gpt-4-turbo-preview", mongo_uri: str = "mongodb://localhost:27017/"):
        """Initialize the agent controller."""
        self.model_name = model_name
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.llm = ChatOpenAI(model=model_name, api_key=api_key)
        self.parser = CommaSeparatedListOutputParser()
        
        # Initialize MongoDB connection
        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client.cognify
            self.tasks_collection = self.db.tasks
            self.subtasks_collection = self.db.subtasks
        except Exception as e:
            warnings.warn(f"MongoDB connection failed: {str(e)}")
            # Create fallback collections
            self.tasks_collection = {}
            self.subtasks_collection = {}
        
        # Initialize query classifier
        self.query_classifier = QueryClassifier(model_name=model_name)
        
        # Add search classification prompt
        self.search_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify if the task requires real-web search (Perplexity) or can be handled in closed environment (OpenAI). Use:
            - 'web' if needing current/live data, real-world updates, or external verification
            - 'closed' for theoretical, general knowledge, or code-related tasks"""),
            ("human", "Task: {task}\nClassification:")
        ])
        
        self.search_classifier_chain = self.search_classifier_prompt | self.llm | CommaSeparatedListOutputParser()

        # Configure the decomposition pipeline
        system_prompt = """
        You are Cognify's Task Decomposer. Analyze the query and break it into
        focused subtasks. Each subtask MUST be:
        1. Self-contained and independently executable
        2. Written as a complete, clear instruction
        3. Specific about what needs to be done and how
        4. NOT just keywords or fragments
        
        IMPORTANT LIMITATIONS:
        1. Maximum 5 subtasks total
        2. Maximum 1 subtask per agent type
        
        For each subtask, determine its type:
        - research_idea_generation: Generating novel research directions
        - data_analysis: Working with data and statistics
        - code_generation: Writing or modifying code
        - summarization: Condensing information
        
        Format each subtask as: <type>: <description>
        
        Example of GOOD subtasks:
        - research_idea_generation: Analyze recent papers in quantum computing to identify gaps in current quantum error correction methods
        - data_analysis: Calculate the correlation between model size and inference speed using the provided benchmark data
        - code_generation: Create a Python script that implements parallel processing for large-scale data preprocessing
        - summarization: Create a comprehensive report comparing different transformer architectures' performance on NLP tasks
        
        Example of BAD subtasks (too vague or incomplete):
        - research_idea_generation: quantum computing research
        - data_analysis: analyze data
        - code_generation: implement algorithm
        - summarization: transformer architectures
        
        Remember to:
        1. Stay within the limits of 5 total subtasks and 1 subtask per type
        2. Make each subtask clear, specific, and independently actionable
        3. Include necessary context within each subtask description
        4. Avoid fragmentary or ambiguous descriptions
        """
        
        self.decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Query: {query}\nSubtasks:")
        ])
        
        self.decomposition_chain = self.decomposition_prompt | self.llm | self.parser
        
        # Registry of available agent capabilities
        self.AGENT_REGISTRY = {
            "summarization": self._summarization_agent,
            "research_idea_generation": self._research_agent,
            "data_analysis": self._analysis_agent,
            "code_generation": self._code_agent
        }

    def recursive_decompose(self, task: str, depth: int = 0) -> List[SubQuery]:
        """Recursively decompose a task into subtasks."""
        if depth > 3:  # Prevent infinite recursion
            return [SubQuery(sub_query=task, task_type="summarization")]
            
        try:
            # Get raw subtasks
            raw_subtasks = self.decomposition_chain.invoke({"query": task})
            
            # Convert to SubQuery objects
            subtasks = []
            for raw_task in raw_subtasks:
                # Determine task type from content
                task_type = self._determine_task_type(raw_task)
                subtasks.append(SubQuery(
                    sub_query=raw_task,
                    task_type=task_type
                ))
            
            # Verify executability
            verified = []
            for st in subtasks:
                if self._agent_capability_check(st.task_type):
                    verified.append(st)
                else:
                    verified.extend(self.recursive_decompose(st.sub_query, depth+1))
                    
            return verified
        except Exception as e:
            warnings.warn(f"Error in task decomposition: {str(e)}")
            return [SubQuery(sub_query=task, task_type="summarization")]

    def parallel_decompose(self, query: str) -> List[SubQuery]:
        """Decompose tasks in parallel for better performance."""
        pre_split_queries = self._pre_split(query)
        
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.decomposition_chain.invoke, {"query": q})
                for q in pre_split_queries
            ]
            results = []
            for f in futures:
                try:
                    results.extend(f.result())
                except Exception as e:
                    warnings.warn(f"Error in parallel decomposition: {str(e)}")
                    
        # Convert to SubQuery objects
        subtasks = []
        for raw_task in results:
            # Determine task type from content
            task_type = self._determine_task_type(raw_task)
            subtasks.append(SubQuery(
                sub_query=raw_task,
                task_type=task_type
            ))
        
        return subtasks

    def _validate_decomposition(self, query: str, subtasks: List[Dict[str, str]]) -> bool:
        """
        Validate the decomposition results.
        
        Args:
            query (str): Original query
            subtasks (List[Dict[str, str]]): List of decomposed subtasks
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not subtasks:
            warnings.warn("No subtasks generated")
            return False
            
        # Validate task types
        for task in subtasks:
            if task["type"] not in self.AGENT_REGISTRY:
                warnings.warn(f"Invalid task type: {task['type']}")
                return False
                
        # Calculate query coverage
        query_terms = set(query.lower().split())
        covered_terms = set()
        for task in subtasks:
            task_terms = set(task["query"].lower().split())
            covered_terms.update(task_terms)
            
        # Remove common words that don't contribute to meaning
        common_words = {"a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "but"}
        query_terms = {term for term in query_terms if term not in common_words}

        if query_terms:
            coverage = len(covered_terms.intersection(query_terms)) / len(query_terms)
            if coverage < 0.7:  
                warnings.warn(f"Insufficient query coverage: {coverage:.2%}")
                return False
                
        return True

    def decompose_task(self, query: str) -> Dict:
        """
        Decompose a complex task into subtasks and store in MongoDB.
        
        Args:
            query (str): The input query to decompose
            
        Returns:
            Dict: Contains task_id and list of subtasks
        """
        try:
            # Create main task document
            task_doc = {
                "query": query,
                "status": "processing",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            task_id = self.tasks_collection.insert_one(task_doc).inserted_id
            
            # Get raw subtasks from LLM
            raw_subtasks = self.decomposition_chain.invoke({"query": query})
            
            # Process and validate each subtask
            subtasks = []
            agent_type_count = {
                "summarization": 0,
                "research_idea_generation": 0,
                "data_analysis": 0,
                "code_generation": 0
            }
            
            for task in raw_subtasks:
                # Break if we've reached 5 subtasks
                if len(subtasks) >= 5:
                    break
                    
                task = task.strip()
                if not task:
                    continue
                    
                # Extract task type and description
                if ":" in task:
                    task_type, description = task.split(":", 1)
                    task_type = task_type.strip()
                    description = description.strip()
                else:
                    description = task
                    task_type = self._determine_task_type(description)
                
                # Validate task type
                if task_type not in self.AGENT_REGISTRY:
                    task_type = self._determine_task_type(description)
                
                # Skip if this agent type has already handled a task
                if agent_type_count[task_type] >= 1:
                    continue
                
                # Validate subtask description
                if not self._validate_subtask_description(description):
                    warnings.warn(f"Invalid subtask description: {description}")
                    continue
                
                # Classify search environment
                search_type = self._classify_search_type(description)
                
                # Create subtask document
                subtask_doc = {
                    "id": f"{task_id}_{uuid.uuid4()}",  # Unique document ID
                    "text": description,                # The subquery text
                    "metadata": {
                        "parent_task_id": task_id,
                        "type": task_type,
                        "search_type": search_type,     # 'web' or 'closed'
                        "status": "pending",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
                
                # Insert into MongoDB
                subtask_id = self.subtasks_collection.insert_one(subtask_doc).inserted_id
                subtask_doc["_id"] = str(subtask_id)
                subtasks.append(subtask_doc)
                
                # Increment the count for this agent type
                agent_type_count[task_type] += 1
            
            # Update main task status
            self.tasks_collection.update_one(
                {"_id": task_id},
                {
                    "$set": {
                        "status": "decomposed",
                        "updated_at": datetime.utcnow(),
                        "subtask_count": len(subtasks)
                    }
                }
            )
            
            return {
                "task_id": str(task_id),
                "subtasks": subtasks
            }
            
        except Exception as e:
            if 'task_id' in locals():
                # Update task status to failed
                self.tasks_collection.update_one(
                    {"_id": task_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error": str(e),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            warnings.warn(f"Error in task decomposition: {str(e)}")
            
            # Fallback: treat the entire query as a single task
            task_type = self._determine_task_type(query)
            fallback_task = {
                "query": query,
                "type": task_type,
                "status": "pending",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            task_id = self.tasks_collection.insert_one(fallback_task).inserted_id
            fallback_task["_id"] = str(task_id)
            return {
                "task_id": str(task_id),
                "subtasks": [fallback_task]
            }
            
    def get_task_status(self, task_id: str) -> Dict:
        """
        Get the status of a task and its subtasks.
        
        Args:
            task_id (str): The ID of the task to check
            
        Returns:
            Dict: Task status and details
        """
        try:
            # Convert string ID to ObjectId
            task_id = ObjectId(task_id)
            
            # Get task details
            task = self.tasks_collection.find_one({"_id": task_id})
            if not task:
                raise ValueError(f"Task not found: {task_id}")
                
            # Get subtasks if they exist
            subtasks = list(self.subtasks_collection.find({"metadata.parent_task_id": task_id}))
            
            # Convert ObjectId to string for JSON serialization
            task["_id"] = str(task["_id"])
            for subtask in subtasks:
                if "_id" in subtask:
                    subtask["_id"] = str(subtask["_id"])
                if "metadata" in subtask and "parent_task_id" in subtask["metadata"]:
                    subtask["metadata"]["parent_task_id"] = str(subtask["metadata"]["parent_task_id"])
                    
            return {
                "task": task,
                "subtasks": subtasks
            }
            
        except Exception as e:
            warnings.warn(f"Error getting task status: {str(e)}")
            return None
            
    def update_task_status(self, task_id: str, status: str, result: Optional[Dict] = None) -> bool:
        """
        Update the status and result of a task.
        
        Args:
            task_id (str): The ID of the task to update
            status (str): New status
            result (Optional[Dict]): Task result data
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Convert string ID to ObjectId
            task_id = ObjectId(task_id)
            
            update_doc = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if result:
                update_doc["result"] = result
                
            # Update the task
            result = self.tasks_collection.update_one(
                {"_id": task_id},
                {"$set": update_doc}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            warnings.warn(f"Error updating task status: {str(e)}")
            return False
            
    def cleanup(self):
        """Close MongoDB connection when done."""
        if hasattr(self, 'client'):
            self.client.close()

    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms from text, ignoring common words."""
        # Common words to ignore
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words, convert to lowercase, and filter out stop words
        words = text.lower().split()
        return {word for word in words if word not in stop_words}

    def _agent_capability_check(self, task_type: str) -> bool:
        """Check if we have an agent capable of handling the task type."""
        return task_type in self.AGENT_REGISTRY

    def _pre_split(self, query: str) -> List[str]:
        """Split complex queries into manageable chunks."""
        # Implementation depends on your specific needs
        return [query]  # Placeholder implementation

    def _determine_task_type(self, task: str) -> str:
        """Determine the task type based on the content."""
        task = task.lower()
        
        # Check for explicit task type in the string
        if ":" in task:
            task_type = task.split(":")[0].strip()
            if task_type in self.AGENT_REGISTRY:
                return task_type
        
        # Keyword-based detection
        keywords = {
            "code_generation": {"code", "implement", "create", "generate", "build", "develop", "program", "script"},
            "research_idea_generation": {"research", "investigate", "explore", "analyze", "study", "define", "requirements"},
            "data_analysis": {"data", "analyze", "process", "validate", "test", "check"},
            "summarization": {"summarize", "summary", "condense", "brief"}
        }
        
        # Count keyword matches for each type
        scores = {task_type: 0 for task_type in keywords}
        words = set(task.split())
        
        for task_type, keyword_set in keywords.items():
            scores[task_type] = len(words.intersection(keyword_set))
        
        # Get the task type with highest keyword matches
        max_score = max(scores.values())
        if max_score > 0:
            for task_type, score in scores.items():
                if score == max_score:
                    return task_type
                    
        return "summarization"  # default type

    def _classify_search_type(self, task_description: str) -> str:
        """Determine appropriate search environment for a task."""
        return self.query_classifier.classify(task_description)

    def _validate_subtask_description(self, description: str) -> bool:
        """
        Validate that a subtask description is clear and self-contained.
        
        Args:
            description (str): The subtask description to validate
            
        Returns:
            bool: True if the description is valid, False otherwise
        """
        # Check minimum length (arbitrary but reasonable threshold)
        words = description.split()
        if len(words) < 8:
            return False
            
        # Check for common ambiguous words when used alone
        ambiguous_starts = ['analyze', 'process', 'handle', 'manage', 'do', 'make', 'perform']
        first_word = words[0].lower()
        if first_word in ambiguous_starts and len(words) < 12:
            return False
            
        # Check for action verbs (basic heuristic)
        action_verbs = ['analyze', 'create', 'develop', 'implement', 'identify', 
                       'research', 'study', 'investigate', 'evaluate', 'assess',
                       'compare', 'contrast', 'generate', 'build', 'design',
                       'write', 'calculate', 'measure', 'test', 'validate']
        
        has_action_verb = any(verb in description.lower() for verb in action_verbs)
        if not has_action_verb:
            return False
            
        return True

    # Agent implementation methods
    def _summarization_agent(self, task: SubQuery):
        """Handle summarization tasks."""
        raise NotImplementedError

    def _research_agent(self, task: SubQuery):
        """Handle research idea generation tasks."""
        raise NotImplementedError

    def _analysis_agent(self, task: SubQuery):
        """Handle data analysis tasks."""
        raise NotImplementedError

    def _code_agent(self, task: SubQuery):
        """Handle code generation tasks."""
        raise NotImplementedError

# Usage example:
if __name__ == "__main__":
    import sys
    
    # Check if API key is provided as argument
    if len(sys.argv) > 1:
        os.environ["OPENAI_API_KEY"] = sys.argv[1]
    
    try:
        controller = AgentController()
        query = "Analyze this dataset and generate research ideas about climate change"
        print(f"Processing query: {query}")
        tasks = controller.decompose_task(query)
        if tasks:
            print("\nValid decomposition. Tasks:")
            for task in tasks["subtasks"]:
                print(f"\n- Task ID: {task['_id']}")
                print(f"  Type: {task['metadata']['type']}")
                print(f"  Query: {task['text']}")
                print(f"  Status: {task['metadata']['status']}")
    except Exception as e:
        print(f"Error: {str(e)}")
