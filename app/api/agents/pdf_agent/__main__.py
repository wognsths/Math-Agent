"""
Main module for running the PDF agent as a standalone server
"""
import os
import logging
import argparse
import json
import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

from core.config import settings
from .agent import PDFAgent
from .task_manager import AgentTaskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "pdf_agent.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pdf_agent")

# Create FastAPI app
app = FastAPI(
    title="PDF Agent API",
    description="API for processing PDF documents and extracting mathematical expressions",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create task manager
agent = PDFAgent()
task_manager = AgentTaskManager(agent=agent)

class TaskRequest(BaseModel):
    session_id: str
    content: Dict[str, Any]

class TaskResponse(BaseModel):
    task_id: str

@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """Create a new PDF processing task"""
    task_id = await task_manager.create_task(request.session_id, request.content)
    return {"task_id": task_id}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details and status"""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    success = await task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    return {"message": "Task cancelled successfully"}

@app.get("/")
async def root():
    """Root endpoint for the API"""
    return {
        "name": "PDF Agent API",
        "version": "0.1.0",
        "status": "active"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

def main():
    """Main function to run the server"""
    parser = argparse.ArgumentParser(description="Run the PDF Agent server")
    parser.add_argument("--host", type=str, default=settings.API_HOST, help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the server to")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.IMAGE_DIR, exist_ok=True)
    
    uvicorn.run(
        "app.api.agents.pdf_agent.__main__:app",
        host=args.host,
        port=args.port,
        reload=True
    )

if __name__ == "__main__":
    main()