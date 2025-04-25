from typing import Dict, Any, Optional, List
from common.types import TaskManager, PushNotificationSenderAuth, Task
from .agent import PDFAgent
import logging
import traceback
import json
import asyncio
import os
from core.config import settings

logger = logging.getLogger(__name__)

class AgentTaskManager(TaskManager):
    def __init__(
        self,
        agent: Optional[PDFAgent] = None,
        notification_sender_auth: Optional[PushNotificationSenderAuth] = None,
    ):
        self.agent = agent or PDFAgent()
        self.notification_sender_auth = notification_sender_auth
        self.tasks: Dict[str, Task] = {}
        
        # Create output directory for PDF processing results
        self.output_dir = os.path.join(settings.OUTPUT_DIR, "pdf_processing")
        os.makedirs(self.output_dir, exist_ok=True)

    async def create_task(self, session_id: str, content: Dict[str, Any]) -> str:
        """Create a new task for the PDF agent"""
        try:
            # Check if content contains PDF file or image file
            has_pdf = False
            query_text = "Process this document"
            pdf_data = None
            pdf_name = "document.pdf"
            
            if not content:
                return self._create_failed_task(session_id, "No content provided")
            
            # Extract data from content
            if "message" in content and "parts" in content["message"]:
                for part in content["message"]["parts"]:
                    if part.get("type") == "text":
                        query_text = part.get("text", query_text)
                    elif part.get("type") == "file":
                        file_content = part.get("file", {})
                        if file_content and file_content.get("mimeType") == "application/pdf":
                            has_pdf = True
                            pdf_data = file_content.get("bytes")
                            if file_content.get("name"):
                                pdf_name = file_content.get("name")
            
            # Check if we have a PDF to process
            if not has_pdf or not pdf_data:
                return self._create_failed_task(session_id, "No PDF file found in the request")
            
            # Create task data
            task_data = {
                "text": query_text,
                "pdf_data": pdf_data,
                "pdf_name": pdf_name
            }
            
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
            asyncio.create_task(self._run_task(task_id, task_data))
            
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
    
    async def _run_task(self, task_id: str, task_data):
        """Run the PDF agent for the given task"""
        try:
            # Execute agent
            result = await self._stream_for_task(task_id, task_data)
            
            # Update task status
            if task_id in self.tasks and self.tasks[task_id].status != "cancelled":
                self.tasks[task_id].status = "completed"
                self.tasks[task_id].result = result
                
                # Save results to disk
                self._save_results(task_id, result)
                
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
    
    async def _stream_for_task(self, task_id: str, task_data) -> Dict[str, Any]:
        """Process the streaming responses and return the final result"""
        query = task_data.get("text", "Process this document")
        
        final_response = None
        
        async for response in self.agent.stream(task_data, task_id):
            # Check if task was cancelled
            if task_id in self.tasks and self.tasks[task_id].status == "cancelled":
                break
                
            # Update intermediate results
            if task_id in self.tasks:
                self.tasks[task_id].result = response
                
            # Save final response
            final_response = response
            
        return final_response or {"error": "No response generated"}
    
    def _save_results(self, task_id: str, result: Dict[str, Any]):
        """Save the processing results to disk"""
        try:
            # Create task-specific output directory
            task_output_dir = os.path.join(self.output_dir, task_id)
            os.makedirs(task_output_dir, exist_ok=True)
            
            # Save full result as JSON
            result_path = os.path.join(task_output_dir, "result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Save extracted math formulas separately if available
            if result.get("math_formulas"):
                formulas_path = os.path.join(task_output_dir, "math_formulas.json")
                with open(formulas_path, 'w', encoding='utf-8') as f:
                    json.dump(result["math_formulas"], f, ensure_ascii=False, indent=2)
                
            # Save text content if available
            if result.get("pdf_text"):
                text_path = os.path.join(task_output_dir, "content.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(result["pdf_text"])
            
            logger.info(f"Saved PDF processing results for task {task_id} to {task_output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results for task {task_id}: {e}")