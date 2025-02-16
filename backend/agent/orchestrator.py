from typing import List, Dict
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

class AggregatedOutput(BaseModel):
    summary: str
    research_ideas: List[str]

class ResponseAggregator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.parser = PydanticOutputParser(pydantic_object=AggregatedOutput)
        
        self.aggregation_prompt = ChatPromptTemplate.from_template("""
        Combine and refine these agent responses into a coherent output:
        
        {responses}
        
        Follow these rules from [MoA Framework][13]:
        1. Remove duplicate information
        2. Resolve conflicts using majority voting
        3. Maintain original technical terminology
        4. Order ideas by implementation feasibility
        
        Output format:
        {format_instructions}
        """)
        
        self.chain = self.aggregation_prompt | self.llm | self.parser

    def aggregate_responses(self, agent_responses: List[Dict]) -> AggregatedOutput:
        """Core aggregation logic using patterns from [11][17]"""
        # Validate inputs
        self._validate_responses(agent_responses)
        
        # Format responses for LLM processing
        formatted_responses = "\n\n".join(
            f"Agent {idx+1}:\n{res['result']}" 
            for idx, res in enumerate(agent_responses)
        )
        
        # Generate aggregated output
        return self.chain.invoke({
            "responses": formatted_responses,
            "format_instructions": self.parser.get_format_instructions()
        })

    def _validate_responses(self, responses: List[Dict]):
        """Validation logic from [1][9]"""
        required_fields = {'result', 'task_type', 'confidence'}
        
        for res in responses:
            if not required_fields.issubset(res.keys()):
                raise ValueError(f"Invalid agent response format: {res.keys()}")
                
            if not isinstance(res['result'], str):
                raise TypeError("Agent result must be a string")

# Integrated with AgentController
class EnhancedAgentController:
    def __init__(self):
        self.aggregator = ResponseAggregator()
        
    def process_responses(self, agent_responses: List[Dict]) -> Dict:
        """Full aggregation pipeline"""
        try:
            aggregated = self.aggregator.aggregate_responses(agent_responses)
            return {
                "status": "success",
                "result": aggregated.dict()
            }
        except Exception as e:
            return self._fallback_aggregation(agent_responses, str(e))
            
    def _fallback_aggregation(self, responses: List[Dict], error: str) -> Dict:
        """Fallback strategy from [7][13]"""
        warnings.warn(f"Aggregation failed: {error}. Using simple concatenation")
        return {
            "summary": "\n".join(res['result'] for res in responses),
            "research_ideas": []
        }
