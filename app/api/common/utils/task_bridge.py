"""
Bridge between agent-specific task managers and the standardized A2A task format
"""
from typing import Dict, Any, Optional, List, Union
from ...common.types import (
    Task, TaskStatus, TaskState, Message, TextPart, FilePart, 
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent, Artifact
)
from uuid import uuid4
from datetime import datetime

class TaskBridge:
    """Bridge for translating between agent-specific task results and A2A task format"""
    
    @staticmethod
    def create_task_from_agent_response(
        task_id: str, 
        agent_response: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Task:
        """
        Convert agent response to standardized Task format
        
        Args:
            task_id: The task identifier
            agent_response: Response from agent's invoke or stream method
            session_id: Optional session identifier
            
        Returns:
            Standardized Task object
        """
        # Determine task state based on agent response
        is_completed = agent_response.get("is_task_complete", False)
        requires_input = agent_response.get("require_user_input", True)
        has_error = "error" in agent_response or agent_response.get("content", "").startswith("Error:")
        
        state = TaskState.COMPLETED if is_completed else (
            TaskState.INPUT_REQUIRED if requires_input else (
            TaskState.FAILED if has_error else TaskState.WORKING
        ))
        
        # Create message from content
        content = agent_response.get("content", "No content provided")
        
        # Create basic task
        task = Task(
            id=task_id,
            sessionId=session_id or str(uuid4()),
            status=TaskStatus(
                state=state,
                message=Message(
                    role="agent",
                    parts=[TextPart(text=content)]
                ),
                timestamp=datetime.now()
            ),
            history=[],
            artifacts=[]
        )
        
        # Add any additional data as artifacts
        additional_data = {k: v for k, v in agent_response.items() 
                         if k not in ["is_task_complete", "require_user_input", "content"]}
        
        if additional_data:
            artifact = Artifact(
                name="agent_data",
                description="Additional data from agent",
                parts=[TextPart(text=str(additional_data))],
                index=0
            )
            task.artifacts = [artifact]
        
        # Handle specific agent data types
        if "original_ocr" in agent_response:
            task.artifacts.append(Artifact(
                name="ocr_result",
                description="Original OCR result",
                parts=[TextPart(text=agent_response["original_ocr"])],
                index=len(task.artifacts)
            ))
            
        if "corrected_latex" in agent_response:
            task.artifacts.append(Artifact(
                name="latex_result",
                description="Corrected LaTeX expression",
                parts=[TextPart(text=agent_response["corrected_latex"])],
                index=len(task.artifacts)
            ))
            
        if "image_id" in agent_response:
            # Add metadata instead of creating a separate artifact
            task.metadata = task.metadata or {}
            task.metadata["image_id"] = agent_response["image_id"]
        
        return task
    
    @staticmethod
    def create_task_update_event(
        task_id: str,
        agent_response: Dict[str, Any],
        is_final: bool = False
    ) -> Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]:
        """
        Create a task update event from agent response
        
        Args:
            task_id: The task identifier
            agent_response: Response from agent
            is_final: Whether this is a final update
            
        Returns:
            Update event object
        """
        # Determine if this is status or artifact update
        if "require_user_input" in agent_response or "is_task_complete" in agent_response:
            # This is a status update
            is_completed = agent_response.get("is_task_complete", False)
            requires_input = agent_response.get("require_user_input", True)
            has_error = "error" in agent_response or agent_response.get("content", "").startswith("Error:")
            
            state = TaskState.COMPLETED if is_completed else (
                TaskState.INPUT_REQUIRED if requires_input else (
                TaskState.FAILED if has_error else TaskState.WORKING
            ))
            
            content = agent_response.get("content", "No content provided")
            
            return TaskStatusUpdateEvent(
                id=task_id,
                status=TaskStatus(
                    state=state,
                    message=Message(
                        role="agent",
                        parts=[TextPart(text=content)]
                    ),
                    timestamp=datetime.now()
                ),
                final=is_final
            )
        else:
            # This is an artifact update
            # For simplicity, we'll assume it's always a text artifact
            content = str(agent_response)
            
            return TaskArtifactUpdateEvent(
                id=task_id,
                artifact=Artifact(
                    name="update",
                    description="Agent update",
                    parts=[TextPart(text=content)],
                    index=0
                )
            )
    
    @staticmethod
    def extract_agent_request(message: Message) -> str:
        """
        Extract agent request from A2A Message format
        
        Args:
            message: The A2A format message
            
        Returns:
            Text content for agent
        """
        # For now, just extract text from the first text part
        for part in message.parts:
            if part.type == "text":
                return part.text
        
        return "" 