from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel
from services.latex import LaTeXService
from typing import Dict, Any, Optional
import os
import json
import uuid
from datetime import datetime
import shutil
from pathlib import Path

router = APIRouter(prefix="/latex", tags=["latex"])

# 저장할 디렉토리 생성
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 이미지 저장 디렉토리
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

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

@router.post("/upload-image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and save it to the output folder"""
    try:
        # 이미지 저장
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        filename = f"{timestamp}_{image_id}{file_extension}"
        file_path = os.path.join(IMAGE_DIR, filename)
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return ImageUploadResponse(
            image_id=image_id,
            image_path=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/context-verify", response_model=LatexVerifyResponse)
async def verify_latex_context(request: LatexVerifyRequest):
    """Verify if the LaTeX is correct in context and save results to output folder"""
    try:
        # LaTeX 검증
        latex_service = LaTeXService()
        is_valid, correction = latex_service.context_verify(request.latex)
        
        # 결과 저장
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{result_id}.json"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # 저장할 데이터 구성
        data = {
            "id": result_id,
            "timestamp": timestamp,
            "image_id": request.image_id,
            "ocr_result": request.ocr_result,
            "latex": request.latex,
            "is_valid": is_valid,
            "correction": correction
        }
        
        # 파일에 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return LatexVerifyResponse(
            is_valid=is_valid,
            correction=correction,
            saved_file_path=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 