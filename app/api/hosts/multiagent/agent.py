"""
Main multiagent host implementation that coordinates between multiple agents
"""
from typing import Dict, Any, Optional, List, AsyncIterable, cast
from common.types import Task, TaskManager, PushNotificationSenderAuth
from common.types import (
    Task as CommonTask,
    TaskStatus, 
    TaskState, 
    Message, 
    TextPart,
    Artifact
)

# Import agents
from agents.ocr_agent.agent import OCRAgent
from agents.math_agent.agent import MathAgent
from core.config import settings
import logging
import asyncio
import os
import json
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

class MultiAgentManager(TaskManager):
    """
    Multiagent manager that coordinates between OCR and Math agents
    to provide a complete workflow for math problem solving
    """
    
    def __init__(
        self,
        notification_sender_auth: Optional[PushNotificationSenderAuth] = None
    ):
        """Initialize the multiagent manager with its child agents"""
        self.ocr_agent = OCRAgent()
        self.math_agent = MathAgent()
        self.tasks: Dict[str, CommonTask] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.notification_sender_auth = notification_sender_auth
        
        # Create a directory for storing task state if it doesn't exist
        self.tasks_dir = os.path.join(settings.OUTPUT_DIR, "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
        
        logger.info("MultiAgentManager initialized with OCR and Math agents")
        
    async def create_task(self, session_id: str, content: Dict[str, Any]) -> str:
        """
        Create a new task and determine which agent should handle it
        based on the content and type of request
        """
        task_id = str(uuid4())
        message_text = None
        file_content = None
        notification_url = content.get("notification_url")
        
        try:
            # Extract text and/or file content from the standard message format
            if "message" in content and "parts" in content["message"]:
                for part in content["message"]["parts"]:
                    if part.get("type") == "text":
                        message_text = part.get("text", "")
                    elif part.get("type") == "file":
                        file_content = part.get("file", {})
            
            # Create task record
            self.tasks[task_id] = CommonTask(
                id=task_id,
                status="running",
                notification_url=notification_url,
                input=content,
                result=None
            )
            
            # Determine which agent should handle the task
            if file_content and self._is_image_file(file_content.get("mimeType", "")):
                # This is an image file, should go to OCR agent
                # Convert file to format OCR agent expects
                ocr_content = {
                    "text": message_text or "Process this image",
                    "image": file_content
                }
                self.sessions[session_id] = {
                    "agent_type": "ocr", 
                    "task_id": task_id,
                    "content": ocr_content,
                    "state": "running"
                }
                
                # Run task in background without blocking
                asyncio.create_task(self._run_task(task_id, ocr_content, session_id, "ocr"))
            else:
                # This is text content, should go to Math agent
                math_content = {
                    "text": message_text or "No content provided"
                }
                self.sessions[session_id] = {
                    "agent_type": "math", 
                    "task_id": task_id,
                    "content": math_content,
                    "state": "running"
                }
                
                # Run task in background
                asyncio.create_task(self._run_task(task_id, math_content, session_id, "math"))
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            # Create failed task with error
            self.tasks[task_id] = CommonTask(
                id=task_id,
                status="failed",
                notification_url=notification_url,
                input=content,
                result={"error": str(e)}
            )
            return task_id
    
    async def get_task(self, task_id: str) -> Optional[CommonTask]:
        """
        Get task details from the appropriate agent
        """
        # Check if task exists in memory
        if task_id in self.tasks:
            return self.tasks[task_id]
            
        # Try to load from disk if not in memory
        try:
            file_path = os.path.join(self.tasks_dir, f"{task_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                    return CommonTask(**task_data)
        except Exception as e:
            logger.error(f"Failed to load task {task_id}: {e}")
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task
        """
        if task_id in self.tasks and self.tasks[task_id].status == "running":
            # Find the associated session
            session_id = None
            for sid, session_data in self.sessions.items():
                if session_data.get("task_id") == task_id:
                    session_id = sid
                    session_data["state"] = "cancelled"
                    break
            
            # Update task status
            self.tasks[task_id].status = "cancelled"
            
            # Save to disk
            self._save_task_state(task_id, self.tasks[task_id])
            return True
        
        return False
    
    async def _run_task(self, task_id: str, content: Dict, session_id: str, agent_type: str):
        """Run agent task asynchronously"""
        try:
            # Select the appropriate agent
            agent = self.ocr_agent if agent_type == "ocr" else self.math_agent
            
            # Process with agent
            result = await self._stream_for_task(task_id, content, session_id, agent)
            
            # Update task status if not cancelled
            if task_id in self.tasks and self.tasks[task_id].status != "cancelled":
                self.tasks[task_id].status = "completed"
                self.tasks[task_id].result = result
                
                # Save task state
                self._save_task_state(task_id, self.tasks[task_id])
                
                # Send notification if applicable
                if self.notification_sender_auth and self.tasks[task_id].notification_url:
                    await self.notification_sender_auth.send_notification(
                        self.tasks[task_id].notification_url,
                        {
                            "task_id": task_id,
                            "status": "completed",
                            "result": result
                        }
                    )
                
                # Special case: If OCR task completed successfully with LaTeX
                # Start math task automatically for analysis
                if (agent_type == "ocr" and 
                    result.get("is_task_complete") and 
                    "corrected_latex" in result and
                    result.get("syntax_valid", False)):
                    
                    # Create a follow-up math task
                    await self._create_follow_up_math_task(
                        latex=result["corrected_latex"],
                        parent_task_id=task_id,
                        session_id=session_id
                    )
                    
        except Exception as e:
            logger.error(f"Error running {agent_type} task {task_id}: {e}")
            
            if task_id in self.tasks:
                self.tasks[task_id].status = "failed"
                self.tasks[task_id].result = {"error": str(e)}
                
                # Save task state
                self._save_task_state(task_id, self.tasks[task_id])
                
                # Send notification for failure
                if self.notification_sender_auth and self.tasks[task_id].notification_url:
                    await self.notification_sender_auth.send_notification(
                        self.tasks[task_id].notification_url,
                        {
                            "task_id": task_id,
                            "status": "failed",
                            "error": str(e)
                        }
                    )
    
    async def _stream_for_task(
        self, task_id: str, content: Dict, 
        session_id: str, agent: Union[OCRAgent, MathAgent]
    ) -> Dict[str, Any]:
        """Process the streaming responses and return the final result"""
        final_response = None
        
        # Stream responses and process
        async for response in agent.stream(content, session_id):
            # Check if task was cancelled
            if task_id in self.tasks and self.tasks[task_id].status == "cancelled":
                break
                
            # Update intermediate results
            if task_id in self.tasks:
                self.tasks[task_id].result = response
                
            # Save final response
            final_response = response
            
        return final_response or {"error": "No response generated"}
    
    async def _create_follow_up_math_task(self, latex: str, parent_task_id: str, session_id: str):
        """Create a follow-up math task to analyze the extracted LaTeX"""
        # Generate a new task ID that references the parent
        math_task_id = f"{parent_task_id}_math"
        
        # Create content for math analysis
        math_content = {
            "text": f"Analyze this LaTeX expression: {latex}",
            "parent_task": parent_task_id,
            "latex": latex
        }
        
        # Create task record
        self.tasks[math_task_id] = CommonTask(
            id=math_task_id,
            status="running",
            notification_url=self.tasks[parent_task_id].notification_url if parent_task_id in self.tasks else None,
            input=math_content,
            result=None,
        )
        
        # Update session
        self.sessions[session_id] = {
            "agent_type": "math", 
            "task_id": math_task_id,
            "content": math_content,
            "state": "running",
            "parent_task_id": parent_task_id
        }
        
        # Run task in background
        asyncio.create_task(self._run_task(math_task_id, math_content, session_id, "math"))
        
        # Update parent task to reference this math task
        if parent_task_id in self.tasks and hasattr(self.tasks[parent_task_id], "result"):
            result = self.tasks[parent_task_id].result or {}
            if isinstance(result, dict):
                result["follow_up_task_id"] = math_task_id
                self.tasks[parent_task_id].result = result
                self._save_task_state(parent_task_id, self.tasks[parent_task_id])
    
    def _save_task_state(self, task_id: str, task: CommonTask):
        """Save task state to disk for persistence"""
        try:
            file_path = os.path.join(self.tasks_dir, f"{task_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                # Convert to dict for JSON serialization
                task_dict = task.dict() if hasattr(task, "dict") else vars(task)
                json.dump(task_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save task state for {task_id}: {e}")
    
    def _is_image_file(self, mime_type: str) -> bool:
        """Check if a file is an image based on MIME type"""
        return mime_type and mime_type.startswith("image/")