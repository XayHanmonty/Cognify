from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import warnings
import os
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

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

class AgentController:
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.llm = ChatOpenAI(model=model_name, api_key=api_key)
        self.parser = CommaSeparatedListOutputParser()
        
        # Configure the decomposition pipeline
        system_prompt = """
        You are Cognify's Task Decomposer. Analyze this query and break it into
        specialized sub-tasks for parallel agent processing. Use these task types:
        - summarization
        - research_idea_generation
        - data_analysis
        - code_generation
        
        Return the sub-tasks as a comma-separated list.
        Include domain-specific parameters where relevant.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        self.decomposition_chain = self.prompt | self.llm | self.parser
        
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

    def validate_decomposition(self, input_query: str, output_tasks: List[SubQuery]) -> bool:
        """Validate the decomposition results."""
        try:
            # Basic validation
            if not output_tasks:
                warnings.warn("No tasks generated")
                return False
                
            # Capability Validation
            for task in output_tasks:
                if not self._agent_capability_check(task.task_type):
                    warnings.warn(f"Unsupported task type: {task.task_type}")
                    return False
            
            # Semantic validation - check if key terms from input are represented
            key_terms = self._extract_key_terms(input_query)
            task_terms = set()
            for task in output_tasks:
                task_terms.update(self._extract_key_terms(task.sub_query))
            
            # Check if at least 50% of key terms are covered
            coverage = len(key_terms.intersection(task_terms)) / len(key_terms)
            if coverage < 0.5:
                warnings.warn(f"Insufficient query coverage: {coverage:.2%}")
                return False
            
            return True
        except Exception as e:
            warnings.warn(f"Validation error: {str(e)}")
            return False
            
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
        if "summarize" in task or "summary" in task:
            return "summarization"
        elif "research" in task or "analyze" in task:
            return "research_idea_generation"
        elif "data" in task or "analyze" in task:
            return "data_analysis"
        elif "code" in task or "implement" in task or "create" in task:
            return "code_generation"
        return "summarization"  # default type

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
        tasks = controller.recursive_decompose(query)
        if controller.validate_decomposition(query, tasks):
            print("\nValid decomposition. Tasks:")
            for task in tasks:
                print(f"\n- Task ID: {task.task_id}")
                print(f"  Type: {task.task_type}")
                print(f"  Query: {task.sub_query}")
                print(f"  Parameters: {task.parameters}")
    except Exception as e:
        print(f"Error: {str(e)}")
