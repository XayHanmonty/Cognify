import os
import sys
from agentController import AgentController

def test_agent():
    # Initialize the agent with MongoDB connection
    agent = AgentController(mongo_uri="mongodb://localhost:27017/")
    
    try:
        # Test queries
        test_queries = [
            # "Given a set of logical conditions that form a paradoxical loop, such as 'The next statement is false. The previous statement is true,' how would you resolve the paradox while maintaining consistency with Gödel's incompleteness theorem and taking into account non-classical logics like paraconsistent logic?"
            "Analyze the lates AI research papers and suggest research directions",
            # "Generate code for a simple web scraper",
            # "Summarize the key findings from the latest AI research papers"
        ]
        
        # Test each query
        for query in test_queries:
            print("\n" + "="*50)
            print(f"Testing query: {query}")
            print("="*50)
            
            try:
                # Decompose the task
                task_result = agent.decompose_task(query)
                
                if task_result and 'task_id' in task_result:
                    main_task_id = task_result['task_id']
                    subtasks = task_result['subtasks']
                    
                    # Get the task status
                    task_status = agent.get_task_status(main_task_id)
                    
                    if task_status:
                        print("\nTask Status:")
                        print(f"Main Task: {task_status['task']['status']}")
                        print("\nSubtasks:")
                        for subtask in task_status['subtasks']:
                            print(f"- Task ID: {subtask['id']}")
                            print(f"  Type: {subtask['metadata']['type']}")
                            print(f"  Text: {subtask['text']}")
                            print(f"  Search Type: {subtask['metadata'].get('search_type', 'closed')}")
                            print(f"  Status: {subtask['metadata']['status']}\n")
                    else:
                        print("\nFailed to get task status")
                else:
                    print("\nDecomposition failed - no tasks generated")
                    
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                
    finally:
        # Clean up MongoDB connection
        agent.cleanup()

if __name__ == "__main__":
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Run the test
    test_agent()
