import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from PIL import Image
import io
from core.config import settings

# Add these imports for local fallback
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2

class OCRService:
    """ Service for Optical Character Recognition of handwritten math equations """
    def __init__(self):
        self.model_path = settings.OCR_MODEL
        self.corrections_path = settings.OCR_CORRECTIONS_DIR
        self.image_dir = settings.IMAGE_DIR
        self.api_token = os.getenv("HF_API_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_path}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Ensure corrections directory exists
        os.makedirs(self.corrections_path, exist_ok=True)
        
        # Check if API token is available
        if not self.api_token:
            logging.warning("Hugging Face API token not found. Will use local OCR fallback.")
            # Check if tesseract is installed
            try:
                pytesseract.get_tesseract_version()
                logging.info("Local OCR using Tesseract is available as fallback")
                self.has_local_ocr = True
            except Exception as e:
                logging.error(f"Local OCR fallback not available: {str(e)}")
                logging.error("Please install Tesseract OCR for local fallback capability")
                self.has_local_ocr = False
        else:
            self.has_local_ocr = True  # We'll still use this as fallback if API fails
            
    def process_image(self, image_path: str) -> str:
        """
        Process an image containing handwritten math equations and output LaTeX
        using Hugging Face Inference API or local fallback
        
        Args:
            image_path: Path to the image file
            
        Returns:
            LaTeX representation of the equation
        """
        try:
            # Open the image using PIL
            img = Image.open(image_path).convert("RGB")
            
            # Try online API if token is available
            if self.api_token:
                try:
                    latex_text = self._process_with_api(img)
                    if latex_text and not latex_text.startswith("Error:"):
                        # Log the OCR processing
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        self._save_correction_data(image_path, latex_text, timestamp, "api")
                        return latex_text
                except Exception as e:
                    logging.error(f"API OCR failed, falling back to local: {str(e)}")
            
            # If API fails or no token, use local OCR if available
            if self.has_local_ocr:
                latex_text = self._process_with_local_ocr(img, image_path)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                self._save_correction_data(image_path, latex_text, timestamp, "local")
                return latex_text
            else:
                return "Error: No OCR method available. Please configure HF_API_TOKEN or install Tesseract."
            
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return f"Error: {str(e)}"
    
    def _process_with_api(self, img: Image.Image) -> str:
        """Process image with Hugging Face API"""
        # Convert image to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        # Call Hugging Face API
        response = requests.post(
            self.api_url,
            headers=self.headers,
            data=image_bytes,
            timeout=30
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Different models return results in different formats
        if isinstance(result, list) and len(result) > 0:
            if 'generated_text' in result[0]:
                return result[0]['generated_text']
            else:
                return str(result[0])
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            return str(result)
    
    def _process_with_local_ocr(self, img: Image.Image, image_path: str) -> str:
        """Fallback using local OCR with preprocessing optimized for math equations"""
        try:
            # Convert to OpenCV format for preprocessing
            img_np = np.array(img)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Preprocessing for better OCR results
            # 1. Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 2. Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 3. Apply some noise reduction
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # 4. Dilate to connect components
            dilated = cv2.dilate(opening, kernel, iterations=1)
            
            # Save preprocessed image for debugging
            debug_path = f"{os.path.dirname(image_path)}/debug_{os.path.basename(image_path)}"
            cv2.imwrite(debug_path, dilated)
            
            # Convert back to PIL for tesseract
            preprocessed = Image.fromarray(dilated)
            
            # Use tesseract with math mode configuration
            # Configure tesseract to treat the image as a single line of math text
            custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=(){}[]<>^_\'"'
            text = pytesseract.image_to_string(preprocessed, config=custom_config)
            
            # Convert to basic LaTeX format
            latex_text = self._convert_to_latex(text.strip())
            
            return latex_text
        except Exception as e:
            logging.error(f"Local OCR processing error: {str(e)}")
            return f"Error in local OCR: {str(e)}"
    
    def _convert_to_latex(self, text: str) -> str:
        """Convert raw OCR text to LaTeX format"""
        # Basic conversion - this can be expanded based on needs
        replacements = {
            'x': '\\times ',
            '^': '^{placeholder}',  # Will need additional processing
            'sqrt': '\\sqrt{placeholder}',
            'pi': '\\pi ',
            'alpha': '\\alpha ',
            'beta': '\\beta ',
            'gamma': '\\gamma ',
            'delta': '\\delta ',
            'epsilon': '\\epsilon ',
            'theta': '\\theta ',
            'lambda': '\\lambda ',
            'mu': '\\mu ',
            'sigma': '\\sigma ',
            'phi': '\\phi ',
            'omega': '\\omega ',
            '>=': '\\geq ',
            '<=': '\\leq ',
            '!=': '\\neq ',
            'inf': '\\infty '
        }
        
        result = text
        
        # Apply replacements
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        # Handle fractions (a/b) -> \frac{a}{b}
        fraction_pattern = r'(\d+)/(\d+)'
        import re
        result = re.sub(fraction_pattern, r'\\frac{\1}{\2}', result)
        
        # Clean up multiple spaces
        result = ' '.join(result.split())
        
        return result
    
    def _save_correction_data(self, image_path: str, latex_text: str, timestamp: str, ocr_method: str = "unknown"):
        """
        Save the OCR processing data for future model improvement
        
        Args:
            image_path: Path to the processed image
            latex_text: Generated LaTeX text
            timestamp: Processing timestamp
            ocr_method: Method used for OCR (api or local)
        """
        correction_data = {
            "image_path": image_path,
            "latex_text": latex_text,
            "timestamp": timestamp,
            "corrected": False,
            "ocr_method": ocr_method
        }
        
        # Create a unique filename for the correction data
        filename = f"{self.corrections_path}/correction_{os.path.basename(image_path)}_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(correction_data, f, indent=2)