from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
import httpx
from typing import Any, Dict, AsyncIterable, Literal
from pydantic import BaseModel
from services.latex import LaTeXService
from core.config import settings

memory = MemorySaver()

@tool
async def verify_latex_context(
    latex: str,
    image_id: str = None,
    ocr_result: str = None
):
    """Verify if the LaTeX is correct in context using the FastAPI endpoint and save results to local storage"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.API_BASE_URL}/latex/context-verify",
                json={
                    "latex": latex,
                    "image_id": image_id,
                    "ocr_result": ocr_result
                }
            )
            response.raise_for_status()

            data = response.json()
            return {
                "is_valid": data.get("is_valid", False),
                "correction": data.get("correction"),
                "saved_file_path": data.get("saved_file_path")
            }
    except httpx.HTTPError as e:
        return {"error": f"API request failed: {e}"}
    except ValueError:
        return {"error": "Invalid JSON response from API."}

class MathAgentResponse(BaseModel):
    """Respond to the user in this format."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str
    content: str

class MathAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for solving math problems. "
        "Do not attempt to answer unrelated questions with math. "
        "You can use provided tools in order to answer questions about user's query. "
        "After using tools, always explain your results to the user and ask for confirmation if needed. "
        "When handling LaTeX, verify with the user if your interpretation is correct. "
        "Set response status to input_required if the user needs to provide more information. "
        "Set response status to error if there is an error while processing the request. "
        "Set response status to completed if the request is complete. "
        "Always maintain a conversational tone and explain complex math concepts clearly."
    )
    def __init__(self):
        self.model = ChatOpenAI(model=settings.OPENAI_MODEL)
        self.tools = [verify_latex_context]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=MathAgentResponse
        )
    
    def invoke(self, query, sessionId) -> Dict[str, Any]:
        try:
            config = {"configurable": {"thread_id": sessionId}}
            self.graph.invoke({"messages": [("user", query)]}, config)        
            return self.get_agent_response(config)
        except Exception as e:
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"An error occurred: {str(e)}"
            }
    
    async def stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        try:
            inputs = {"messages": [("user", query)]}
            config = {"configurable": {"thread_id": sessionId}}

            for item in self.graph.stream(inputs, config, stream_mode="values"):
                message = item["messages"][-1]
                if (
                    isinstance(message, AIMessage)
                    and message.tool_calls
                    and len(message.tool_calls) > 0
                ):
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Analyzing math problem and processing query...",
                    }
                elif isinstance(message, ToolMessage):
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Verifying LaTeX and processing results...",
                    }
            
            yield self.get_agent_response(config)
        except Exception as e:
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"An error occurred while streaming: {str(e)}"
            }

    def get_agent_response(self, config):
        try:
            current_state = self.graph.get_state(config)        
            structured_response = current_state.values.get('structured_response')
            if structured_response and isinstance(structured_response, MathAgentResponse): 
                if structured_response.status == "input_required":
                    return {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": structured_response.message
                    }
                elif structured_response.status == "error":
                    return {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": structured_response.message
                    }
                elif structured_response.status == "completed":
                    return {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": structured_response.content
                    }

            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "We are unable to process your request at the moment. Please try again.",
            }
        except Exception as e:
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"Error retrieving agent response: {str(e)}"
            }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
