from typing import Dict, Any, Optional, List
from common.types import TaskManager, PushNotificationSenderAuth, Task
from .agent import OCRAgent
import logging
import traceback
import json
import asyncio

logger = logging.getLogger(__name__)

class AgentTaskManager(TaskManager):
    def __init__(
        self,
        agent: OCRAgent,
        notification_sender_auth: Optional[PushNotificationSenderAuth] = None,
    ):
        self.agent = agent
        self.notification_sender_auth = notification_sender_auth
        self.tasks: Dict[str, Task] = {}

    async def create_task(self, session_id: str, content: Dict[str, Any]) -> str:
        """Create a new task for the agent"""
        try:
            if not content.get("text"):
                return self._create_failed_task(session_id, "No text content provided")
            
            query = content.get("text")
            task_id = session_id
            
            # Create task
            self.tasks[task_id] = Task(
                id=task_id,
                status="running",
                notification_url=content.get("notification_url"),
                input=content,
                result=None
            )
            
            # Run task in background
            asyncio.create_task(self._run_task(task_id, query))
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            logger.error(traceback.format_exc())
            return self._create_failed_task(session_id, f"Failed to create task: {str(e)}")
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task details"""
        return self.tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.tasks:
            if self.tasks[task_id].status == "running":
                self.tasks[task_id].status = "cancelled"
                return True
        return False
    
    def _create_failed_task(self, task_id: str, error_message: str) -> str:
        """Create a failed task with error"""
        self.tasks[task_id] = Task(
            id=task_id,
            status="failed",
            notification_url=None,
            input={},
            result={
                "error": error_message
            }
        )
        return task_id
    
    async def _run_task(self, task_id: str, query: str):
        """Run the OCR agent for the given task"""
        try:
            # Execute agent
            result = await self._stream_for_task(task_id, query)
            
            # Update task status
            if task_id in self.tasks and self.tasks[task_id].status != "cancelled":
                self.tasks[task_id].status = "completed"
                self.tasks[task_id].result = result
                
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
                    
        except Exception as e:
            logger.error(f"Error running task {task_id}: {e}")
            logger.error(traceback.format_exc())
            
            if task_id in self.tasks:
                self.tasks[task_id].status = "failed"
                self.tasks[task_id].result = {"error": str(e)}
                
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
    
    async def _stream_for_task(self, task_id: str, query: str) -> Dict[str, Any]:
        """Process the streaming responses and return the final result"""
        final_response = None
        
        async for response in self.agent.stream(query, task_id):
            # Check if task was cancelled
            if task_id in self.tasks and self.tasks[task_id].status == "cancelled":
                break
                
            # Update intermediate results
            if task_id in self.tasks:
                self.tasks[task_id].result = response
                
            # Save final response
            final_response = response
            
        return final_response or {"error": "No response generated"} 