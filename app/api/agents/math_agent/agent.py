from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
import httpx
from typing import Any, Dict, AsyncIterable, Literal
from pydantic import BaseModel
from services.latex import LaTeXService

memory = MemorySaver()

@tool
def verify_latex_context(
    latex: str,
    image_id: str = None,
    ocr_result: str = None
):
    """Verify if the LaTeX is correct in context using the FastAPI endpoint and save results to local storage"""
    try:
        response = httpx.post(
            "http://localhost:8000/latex/context-verify",
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

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class MathAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for solving math problems."
        "Do not attempt to answer unrelated questions with math"
        "Set response status to input_required if the user needs to provide more information."
        "Set response status to error if there is an error while processing the request."
        "Set response status to completed if the request is complete."
    )
