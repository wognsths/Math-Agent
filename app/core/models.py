from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class EquationBase(BaseModel):
    """Base Equation model"""
    latex: str
    
class EquationCreate(EquationBase):
    """Model for creating a new equation"""
    image_path: Optional[str] = None

class EquationResponse(EquationBase):
    """Response model for equations"""
    rendered_latex: str
    
class EquationInDB(EquationBase):
    """Database model for equations"""
    id: str
    image_path: str
    user_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class SolutionRequest(BaseModel):
    """Request model for solution verification"""
    latex: str
    solution: str

class SolutionRequestWithPrompt(BaseModel):
    """Request model for solution verification with custom prompt"""
    prompt: str
    solution: str

class SolutionResponse(BaseModel):
    """Response model for solution verification"""
    is_correct: bool
    explanation: str
    step_by_step: List[str]
    
class SolutionInDB(BaseModel):
    """Database model for solutions"""
    id: str
    equation_id: str
    solution_text: str
    is_correct: bool
    user_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        orm_mode = True