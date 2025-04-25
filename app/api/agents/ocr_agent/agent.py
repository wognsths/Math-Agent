from typing import Any, Dict, List, Optional, Literal, AsyncIterable
from pydantic import BaseModel
import os
import uuid
from pathlib import Path
from datetime import datetime
import httpx
import json

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage

from core.config import settings

memory = MemorySaver()

@tool
async def process_image_ocr(
    image_id: str
):
    """Process an image using OCR to extract LaTeX from mathematical equations"""
    try:
        response = await httpx.AsyncClient().post(
            f"{settings.API_BASE_URL}/latex/ocr-image/" + image_id,
        )
        response.raise_for_status()

        data = response.json()
        return {
            "ocr_result": data.get("ocr_result", ""),
            "image_id": data.get("image_id", "")
        }
    except httpx.HTTPError as e:
        return {"error": f"API request failed: {e}"}
    except ValueError:
        return {"error": "Invalid JSON response from API."}

@tool
async def verify_latex_syntax(
    latex: str,
    image_id: str = None
):
    """Verify if the LaTeX syntax is correct"""
    try:
        response = await httpx.AsyncClient().post(
            f"{settings.API_BASE_URL}/latex/syntax-verify",
            json={
                "latex": latex,
                "image_id": image_id
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

@tool
async def save_ocr_correction(
    original_ocr: str,
    corrected_latex: str,
    image_id: str,
    user_feedback: str = None
):
    """Save the correction made by the user for OCR improvement"""
    try:
        # Create output directory if it doesn't exist
        ocr_corrections_dir = os.path.join(settings.OUTPUT_DIR, "ocr_corrections")
        os.makedirs(ocr_corrections_dir, exist_ok=True)
        
        # Generate unique ID and timestamp
        correction_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare correction data
        correction_data = {
            "id": correction_id,
            "timestamp": timestamp,
            "image_id": image_id,
            "original_ocr": original_ocr,
            "corrected_latex": corrected_latex,
            "user_feedback": user_feedback,
            "diff": {
                "added": corrected_latex.replace(original_ocr, ""),
                "removed": original_ocr.replace(corrected_latex, ""),
            }
        }
        
        # Save to file
        filename = f"{timestamp}_{correction_id}.json"
        file_path = os.path.join(ocr_corrections_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(correction_data, f, ensure_ascii=False, indent=2)
        
        return {
            "correction_id": correction_id,
            "saved_file_path": file_path,
            "message": "Correction saved successfully for future OCR improvement"
        }
        
    except Exception as e:
        return {"error": f"Failed to save correction: {e}"}

class OCRAgentResponse(BaseModel):
    """Response format for the OCR Agent"""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str
    content: str = ""
    original_ocr: Optional[str] = None
    corrected_latex: Optional[str] = None
    syntax_valid: Optional[bool] = None
    image_id: Optional[str] = None

class OCRAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialized OCR assistant for mathematical equations. "
        "Your job is to process images of math equations, extract LaTeX, and verify basic syntax. "
        "Follow these steps in order: "
        "1. First, process the image to extract LaTeX using OCR. "
        "2. Show the OCR result to the user and ask if they want to make corrections. "
        "3. If the user provides corrections, verify the syntax of their LaTeX. "
        "4. If there are syntax errors, explain them clearly and suggest fixes. "
        "5. Once syntax is valid, provide the final LaTeX to the user. "
        "6. Save any user corrections for future OCR improvement. "
        "Remember that you only need to verify basic LaTeX syntax correctness. "
        "You don't need to verify if the LaTeX makes mathematical sense in context - "
        "that will be handled by the Math Agent later. "
        "Always maintain a helpful, educational tone and explain LaTeX syntax errors in a way that helps users learn. "
        "Set response status to input_required if the user needs to provide more information. "
        "Set response status to error if there is an error while processing the request. "
        "Set response status to completed if the request is complete."
    )
    
    def __init__(self):
        self.model = ChatOpenAI(model=settings.OPENAI_MODEL)
        self.tools = [
            process_image_ocr,
            verify_latex_syntax,
            save_ocr_correction
        ]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=OCRAgentResponse
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
                    # Check which tool is being called
                    tool_name = message.tool_calls[0].name if message.tool_calls[0].name else "unknown"
                    
                    if tool_name == "process_image_ocr":
                        status_msg = "Processing image with OCR..."
                    elif tool_name == "verify_latex_syntax":
                        status_msg = "Verifying LaTeX syntax..."
                    elif tool_name == "save_ocr_correction":
                        status_msg = "Saving your corrections for OCR improvement..."
                    else:
                        status_msg = "Processing your request..."
                    
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": status_msg,
                    }
                elif isinstance(message, ToolMessage):
                    tool_result = "Tool execution completed"
                    if "ocr_result" in message.content:
                        tool_result = "OCR extraction completed"
                    elif "is_valid" in message.content:
                        is_valid = json.loads(message.content).get("is_valid", False)
                        if is_valid:
                            tool_result = "LaTeX syntax validation successful"
                        else:
                            tool_result = "LaTeX syntax validation found issues"
                    
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": tool_result,
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
            
            if structured_response and isinstance(structured_response, OCRAgentResponse): 
                if structured_response.status == "input_required":
                    # For input_required, prepare appropriate response based on the workflow stage
                    response = {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": structured_response.message
                    }
                    
                    # Add OCR results if available
                    if structured_response.original_ocr:
                        response["original_ocr"] = structured_response.original_ocr
                        response["image_id"] = structured_response.image_id
                        
                    return response
                    
                elif structured_response.status == "error":
                    return {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": structured_response.message
                    }
                elif structured_response.status == "completed":
                    response = {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": structured_response.content
                    }
                    
                    # Include additional metadata if available
                    if structured_response.corrected_latex:
                        response["corrected_latex"] = structured_response.corrected_latex
                    if structured_response.syntax_valid is not None:
                        response["syntax_valid"] = structured_response.syntax_valid
                    if structured_response.image_id:
                        response["image_id"] = structured_response.image_id
                        
                    return response

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

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "image/jpeg", "image/png"]