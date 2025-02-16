from typing import Dict, List, Optional
import logging
from backend.agent.agent import Agent
from backend.controllers.agentController import AgentController

logger = logging.getLogger(__name__)

class SimpleRouterAgent:
    def __init__(self):
        """Initialize the router with AgentController and Agent instances"""
        self.agent_controller = AgentController()
        self.agent = Agent()

    async def process_message(self, message: str) -> Dict:
        """
        Process user message by:
        1. First decomposing it using AgentController
        2. Then routing subtasks to appropriate handlers
        3. Finally combining and returning results
        """
        try:
            # Step 1: Decompose message using AgentController
            task_result = self.agent_controller.decompose_task(message)
            logger.info(f"Task decomposition result: {task_result}")
            
            # Step 2: Process each subtask based on its type
            results = []
            for subtask in task_result.get('subtasks', []):
                task_type = subtask.get('metadata', {}).get('type')
                logger.info(f"Processing subtask: {subtask}")
                
                if task_type == 'summarization':
                    # Handle summarization tasks using Agent's search capability
                    full_response = ""
                    for chunk in self.agent.summarize(subtask.get('text', ''), stream=True):
                        full_response += chunk
                    logger.info(f"Summarization result: {full_response}")
                    results.append({
                        'type': 'summarization',
                        'result': full_response
                    })
                    
                elif task_type == 'research_idea_generation':
                    # Handle research/search tasks using Agent's search capability
                    full_response = ""
                    for chunk in self.agent.search(subtask.get('text', ''), stream=True):
                        full_response += chunk
                    logger.info(f"Research result: {full_response}")
                    results.append({
                        'type': 'research',
                        'result': full_response
                    })
                elif task_type == 'code_generation':
                    # Handle code generation tasks
                    full_response = ""
                    for chunk in self.agent.generate_code(subtask.get('text', ''), stream=True):
                        full_response += chunk
                    logger.info(f"Code generation result: {full_response}")
                    results.append({
                        'type': 'code',
                        'result': full_response
                    })
            
            # Step 3: Combine results into a coherent response
            combined_response = self._combine_results(results)
            logger.info(f"Combined response: {combined_response}")
            return combined_response
            
        except Exception as e:
            logger.error(f"Error in SimpleRouterAgent: {str(e)}")
            raise

    # TODO: Handle results from all agents
    def _combine_results(self, results: List[Dict]) -> Dict:
        """
        Combine results from different subtasks into a coherent response
        """
        combined = []
        
        # Add research results first
        research_results = [r['result'] for r in results if r['type'] == 'research']
        if research_results:
            combined.extend(research_results)
        
        # Add summarization results
        summary_results = [r['result'] for r in results if r['type'] == 'summarization']
        if summary_results:
            combined.extend(summary_results)
        
        # Add code results
        code_results = [r['result'] for r in results if r['type'] == 'code']
        
        # Clean up the research/summary output by removing redundant sections and sources
        if combined:
            result = "\n".join(combined)
            
            # Extract all sources
            sources = []
            for section in result.split("Sources:"):
                if ":" in section:  # Skip the first split which is before "Sources:"
                    continue
                # Get the sources from this section
                source_lines = [line.strip() for line in section.split("\n") if line.strip()]
                sources.extend(source_lines)
            
            # Remove all "Sources:" sections from the main text
            main_text = result.split("Sources:")[0].strip()
            
            # Deduplicate sources
            seen = set()
            unique_sources = []
            for source in sources:
                if source not in seen:
                    seen.add(source)
                    unique_sources.append(source)
            
            # Combine main text with unique sources
            if unique_sources:
                result = f"{main_text}\n\nSources:\n" + "\n".join(unique_sources)
            else:
                result = main_text
        else:
            result = ""
            
        # Add code results after the research/summary section
        if code_results:
            if result:
                result += "\n\n## Generated Code:\n" + "\n".join(code_results)
            else:
                result = "\n".join(code_results)
            
        return {'response': result if result else "No results found."}

if __name__ == "__main__":
    import asyncio
    
    async def main():
        router = SimpleRouterAgent()
        result = await router.process_message("Analyze the latest AI research papers and suggest research directions")
        print(result)
    
    asyncio.run(main())
