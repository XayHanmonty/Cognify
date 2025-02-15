import os
from agentController import AgentController

def test_agent():
    # Example queries to test different capabilities
    test_queries = [
        "Analyze climate change data and suggest research directions",
        "Generate code for a simple web scraper",
        "Summarize the key findings from the latest AI research papers"
    ]
    
    try:
        # Initialize the agent controller
        controller = AgentController()
        
        # Test each query
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Testing query: {query}")
            print(f"{'='*50}")
            
            # Get task decomposition
            tasks = controller.recursive_decompose(query)
            
            # Validate and display results
            if controller.validate_decomposition(query, tasks):
                print("\nSuccessful decomposition! Tasks:")
                for task in tasks:
                    print(f"\n- Task ID: {task.task_id}")
                    print(f"  Type: {task.task_type}")
                    print(f"  Query: {task.sub_query}")
                    if task.parameters:
                        print(f"  Parameters: {task.parameters}")
            else:
                print("\nDecomposition validation failed!")
                
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Run the test
    test_agent()
