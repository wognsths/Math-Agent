"""
PDF Agent for processing PDF documents and extracting mathematical expressions
"""
from typing import Any, Dict, List, Optional, Literal, AsyncIterable
from pydantic import BaseModel
import os
import uuid
from pathlib import Path
from datetime import datetime
import httpx
import json
import io
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import base64
import tempfile

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage

from core.config import settings

memory = MemorySaver()

@tool
async def extract_images_from_pdf(
    pdf_data: str,
    pdf_name: str = "document.pdf"
):
    """
    Extract images from a PDF file.
    
    Args:
        pdf_data: Base64 encoded PDF data
        pdf_name: Optional name of the PDF file
        
    Returns:
        A list of extracted image paths and information
    """
    try:
        # Create a directory for extracted images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(settings.IMAGE_DIR, f"pdf_extract_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Decode base64 data
        pdf_bytes = base64.b64decode(pdf_data)
        
        # Save PDF temporarily to process with PyMuPDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        # Open the PDF
        doc = fitz.open(temp_pdf_path)
        
        extracted_images = []
        
        # Iterate through pages
        for page_num, page in enumerate(doc):
            # Get images
            image_list = page.get_images(full=True)
            
            # Process each image
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                image_bytes = base_img["image"]
                image_ext = base_img["ext"]
                
                # Save the image
                image_filename = f"page_{page_num+1}_img_{img_idx+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Create image info
                image_info = {
                    "page": page_num + 1,
                    "image_id": f"pdf_{page_num+1}_{img_idx+1}_{uuid.uuid4()}",
                    "path": image_path,
                    "filename": image_filename,
                    "width": base_img["width"],
                    "height": base_img["height"]
                }
                
                extracted_images.append(image_info)
        
        # Clean up temp file
        os.unlink(temp_pdf_path)
        
        # Return only if we found images
        if extracted_images:
            return {
                "status": "success",
                "message": f"Extracted {len(extracted_images)} images from PDF",
                "images": extracted_images,
                "output_directory": output_dir
            }
        else:
            # If no images were found, try to extract using rendering
            return await extract_math_regions_from_pdf(pdf_data, pdf_name)
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to extract images from PDF: {str(e)}"
        }

@tool
async def extract_math_regions_from_pdf(
    pdf_data: str,
    pdf_name: str = "document.pdf"
):
    """
    Extract regions that might contain mathematical expressions from a PDF using rendering and vision analysis.
    This is used when direct image extraction doesn't yield results.
    
    Args:
        pdf_data: Base64 encoded PDF data
        pdf_name: Optional name of the PDF file
        
    Returns:
        A list of extracted math region images
    """
    try:
        # Create a directory for extracted math regions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(settings.IMAGE_DIR, f"pdf_math_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Decode base64 data
        pdf_bytes = base64.b64decode(pdf_data)
        
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        # Open the PDF
        doc = fitz.open(temp_pdf_path)
        
        extracted_regions = []
        
        # Iterate through pages
        for page_num, page in enumerate(doc):
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV format
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to identify potential math regions
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (to avoid detecting single characters)
            min_width, min_height = 50, 20
            potential_math_regions = []
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w > min_width and h > min_height:
                    # Calculate density of black pixels to filter out text blocks
                    roi = binary[y:y+h, x:x+w]
                    density = cv2.countNonZero(roi) / (w * h)
                    
                    # Math formulas typically have a medium density
                    if 0.05 < density < 0.6:
                        # Add padding around the region
                        padding = 10
                        x_pad = max(0, x - padding)
                        y_pad = max(0, y - padding)
                        w_pad = min(pix.width - x_pad, w + 2*padding)
                        h_pad = min(pix.height - y_pad, h + 2*padding)
                        
                        potential_math_regions.append((x_pad, y_pad, w_pad, h_pad))
            
            # Extract and save potential math regions
            for i, region in enumerate(potential_math_regions):
                x, y, w, h = region
                math_region = img[y:y+h, x:x+w]
                
                # Save the region
                region_filename = f"page_{page_num+1}_math_{i+1}.png"
                region_path = os.path.join(output_dir, region_filename)
                cv2.imwrite(region_path, math_region)
                
                # Create region info
                region_info = {
                    "page": page_num + 1,
                    "image_id": f"pdf_math_{page_num+1}_{i+1}_{uuid.uuid4()}",
                    "path": region_path,
                    "filename": region_filename,
                    "width": w,
                    "height": h,
                    "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                }
                
                extracted_regions.append(region_info)
        
        # Clean up temp file
        os.unlink(temp_pdf_path)
        
        return {
            "status": "success",
            "message": f"Extracted {len(extracted_regions)} potential math regions from PDF",
            "regions": extracted_regions,
            "output_directory": output_dir
        }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to extract math regions from PDF: {str(e)}"
        }

@tool
async def process_image_with_ocr(
    image_path: str,
    image_id: str
):
    """
    Process an extracted image with OCR to identify LaTeX content.
    
    Args:
        image_path: Path to the image file
        image_id: Unique ID for the image
        
    Returns:
        OCR results with extracted LaTeX
    """
    try:
        # Call OCR service API
        response = await httpx.AsyncClient().post(
            f"{settings.API_BASE_URL}/latex/ocr-image/{image_id}",
            json={"image_path": image_path}
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        return {"error": f"API request failed: {e}"}
    except Exception as e:
        return {"error": f"Failed to process image with OCR: {str(e)}"}

@tool
async def filter_math_images(
    images: List[Dict[str, Any]]
):
    """
    Filter images to identify those that likely contain mathematical equations
    
    Args:
        images: List of image information dictionaries
        
    Returns:
        Filtered list of images that likely contain mathematical content
    """
    try:
        # Load the OCR model client for local processing
        from services.ocr import OCRService
        ocr_service = OCRService()
        
        filtered_images = []
        math_indicators = [
            '+', '-', '=', '÷', '×', '/', '*', '^', 'sqrt', 'sum',
            'integral', 'frac', 'alpha', 'beta', 'gamma', 'delta', 
            'theta', 'lambda', 'sigma', 'pi'
        ]
        
        for img_info in images:
            try:
                # Process with local OCR if available, otherwise just pass image through
                if hasattr(ocr_service, 'has_local_ocr') and ocr_service.has_local_ocr:
                    # Quick check with OCR to see if it contains math symbols
                    image_path = img_info.get("path")
                    if not image_path or not os.path.exists(image_path):
                        continue
                        
                    img = Image.open(image_path).convert("RGB")
                    result = ocr_service._process_with_local_ocr(img, image_path)
                    
                    # Check if result contains math indicators
                    is_math = any(indicator in result.lower() for indicator in math_indicators)
                    
                    if is_math:
                        img_info["preview_text"] = result
                        img_info["is_math"] = True
                        filtered_images.append(img_info)
                else:
                    # If no local OCR, just add the image for further processing
                    img_info["is_math"] = True
                    filtered_images.append(img_info)
            except Exception as e:
                # If error in processing, include the image anyway
                img_info["error"] = str(e)
                filtered_images.append(img_info)
        
        return {
            "status": "success",
            "filtered_images": filtered_images,
            "total_images": len(images),
            "math_images": len(filtered_images)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to filter images: {str(e)}"
        }

class PDFAgentResponse(BaseModel):
    """Response format for the PDF Agent"""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str
    content: str = ""
    images: Optional[List[Dict[str, Any]]] = None
    math_formulas: Optional[List[Dict[str, Any]]] = None
    pdf_text: Optional[str] = None

class PDFAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialized PDF processing assistant for mathematical documents. "
        "Your job is to extract and process mathematical expressions from PDF files. "
        "Follow these steps: "
        "1. Extract images and potential mathematical regions from the PDF. "
        "2. Filter the extracted images to identify those with mathematical content. "
        "3. Process the math-containing images with OCR to extract LaTeX representations. "
        "4. Compile and organize the extracted math formulas. "
        "Always maintain a helpful tone and explain the processing steps clearly. "
        "Set response status to input_required if you need more information from the user. "
        "Set response status to error if there is an error during processing. "
        "Set response status to completed when the processing is finished."
    )
    
    def __init__(self):
        self.model = ChatOpenAI(model=settings.OPENAI_MODEL)
        self.tools = [
            extract_images_from_pdf,
            extract_math_regions_from_pdf,
            process_image_with_ocr,
            filter_math_images
        ]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=PDFAgentResponse
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
                "content": f"An error occurred while processing the PDF: {str(e)}"
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
                    
                    if tool_name == "extract_images_from_pdf":
                        status_msg = "Extracting images from PDF..."
                    elif tool_name == "extract_math_regions_from_pdf":
                        status_msg = "Analyzing PDF for mathematical expressions..."
                    elif tool_name == "process_image_with_ocr":
                        status_msg = "Processing image with OCR..."
                    elif tool_name == "filter_math_images":
                        status_msg = "Identifying mathematical content in images..."
                    else:
                        status_msg = "Processing your request..."
                    
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": status_msg,
                    }
                elif isinstance(message, ToolMessage):
                    tool_result = "Processing step completed"
                    
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
                "content": f"An error occurred while processing: {str(e)}"
            }

    def get_agent_response(self, config):
        try:
            current_state = self.graph.get_state(config)        
            structured_response = current_state.values.get('structured_response')
            
            if structured_response and isinstance(structured_response, PDFAgentResponse): 
                if structured_response.status == "input_required":
                    response = {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": structured_response.message
                    }
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
                    if structured_response.images:
                        response["images"] = structured_response.images
                    if structured_response.math_formulas:
                        response["math_formulas"] = structured_response.math_formulas
                    if structured_response.pdf_text:
                        response["pdf_text"] = structured_response.pdf_text
                        
                    return response

            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "We are unable to process your PDF at the moment. Please try again.",
            }
        except Exception as e:
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"Error retrieving agent response: {str(e)}"
            }

    SUPPORTED_CONTENT_TYPES = ["application/pdf", "image/jpeg", "image/png"]