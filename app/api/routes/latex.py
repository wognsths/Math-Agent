from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel
from services.latex import LaTeXService
from services.ocr import OCRService
from services.solution import SolutionService
from typing import Dict, Any, Optional, List
import os
import json
import uuid
from datetime import datetime
import shutil
from pathlib import Path
from core.config import settings

router = APIRouter(prefix="/latex", tags=["latex"])

# Use configured directories
OUTPUT_DIR = settings.OUTPUT_DIR
IMAGE_DIR = settings.IMAGE_DIR

class LatexVerifyRequest(BaseModel):
    latex: str
    image_id: Optional[str] = None
    ocr_result: Optional[str] = None

class LatexVerifyResponse(BaseModel):
    is_valid: bool
    correction: Optional[str] = None
    saved_file_path: Optional[str] = None

class ImageUploadResponse(BaseModel):
    image_id: str
    image_path: str

class OCRResponse(BaseModel):
    ocr_result: str
    image_id: str

class SolutionRequest(BaseModel):
    latex: str
    image_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class SolutionStep(BaseModel):
    step_number: int
    explanation: str
    latex: str

class SolutionResponse(BaseModel):
    problem: str
    steps: List[SolutionStep]
    final_answer: str
    image_id: Optional[str] = None
    saved_file_path: Optional[str] = None

@router.post("/upload-image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and save it to the output folder"""
    try:
        # Save image
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        filename = f"{timestamp}_{image_id}{file_extension}"
        file_path = os.path.join(IMAGE_DIR, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return ImageUploadResponse(
            image_id=image_id,
            image_path=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ocr-image/{image_id}", response_model=OCRResponse)
async def ocr_image(image_id: str):
    """Perform OCR on an uploaded image to extract LaTeX expressions"""
    try:
        # Find the image
        image_files = os.listdir(IMAGE_DIR)
        image_path = None
        
        for file in image_files:
            if image_id in file:
                image_path = os.path.join(IMAGE_DIR, file)
                break
        
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Process image using OCR service
        ocr_service = OCRService()
        ocr_result = ocr_service.process_image(image_path)
        
        return OCRResponse(
            ocr_result=ocr_result,
            image_id=image_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/syntax-verify", response_model=LatexVerifyResponse)
async def verify_latex_syntax(request: LatexVerifyRequest):
    """Verify if the LaTeX syntax is correct and save results to output folder"""
    try:
        # Verify LaTeX syntax only
        latex_service = LaTeXService()
        is_valid, correction = latex_service.syntax_verify(request.latex)
        
        # Save results
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_syntax_{result_id}.json"
        file_path = os.path.join(settings.VERIFICATION_DIR, filename)
        
        # Prepare data for saving
        data = {
            "id": result_id,
            "timestamp": timestamp,
            "image_id": request.image_id,
            "ocr_result": request.ocr_result,
            "latex": request.latex,
            "is_valid": is_valid,
            "correction": correction,
            "verification_type": "syntax"
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return LatexVerifyResponse(
            is_valid=is_valid,
            correction=correction,
            saved_file_path=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/context-verify", response_model=LatexVerifyResponse)
async def verify_latex_context(request: LatexVerifyRequest):
    """Verify if the LaTeX is correct in context and save results to output folder"""
    try:
        # Verify LaTeX in context
        latex_service = LaTeXService()
        is_valid, correction = latex_service.context_verify(request.latex)
        
        # Save results
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_context_{result_id}.json"
        file_path = os.path.join(settings.VERIFICATION_DIR, filename)
        
        # Prepare data for saving
        data = {
            "id": result_id,
            "timestamp": timestamp,
            "image_id": request.image_id,
            "ocr_result": request.ocr_result,
            "latex": request.latex,
            "is_valid": is_valid,
            "correction": correction,
            "verification_type": "context"
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return LatexVerifyResponse(
            is_valid=is_valid,
            correction=correction,
            saved_file_path=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-solution", response_model=SolutionResponse)
async def generate_solution(request: SolutionRequest):
    """Generate step-by-step solution for a mathematical problem expressed in LaTeX"""
    try:
        # Generate solution using solution service
        solution_service = SolutionService()
        problem, steps, final_answer = solution_service.generate_solution(request.latex, options=request.options)
        
        # Save results
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_solution_{result_id}.json"
        file_path = os.path.join(settings.SOLUTION_DIR, filename)
        
        # Prepare data for saving
        data = {
            "id": result_id,
            "timestamp": timestamp,
            "image_id": request.image_id,
            "problem": problem,
            "steps": [step.dict() for step in steps],
            "final_answer": final_answer,
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return SolutionResponse(
            problem=problem,
            steps=steps,
            final_answer=final_answer,
            image_id=request.image_id,
            saved_file_path=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 