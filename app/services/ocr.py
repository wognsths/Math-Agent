import os
import json
import requests
from datetime import datetime
from typing import Dict, Any
import logging
from PIL import Image
import io
from core.config import settings

class OCRService:
    """ Service for Optical Character Recognition of handwritten math equations """
    def __init__(self):
        self.model_path = settings.OCR_MODEL
        self.corrections_path = os.getenv("CORRECTIONS_PATH", "data/corrections")
        self.api_token = os.getenv("HF_API_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_path}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Ensure corrections directory exists
        os.makedirs(self.corrections_path, exist_ok=True)
        
        # Check if API token is available
        if not self.api_token:
            logging.warning("Hugging Face API token not found. Set HF_API_TOKEN environment variable.")
            
    def process_image(self, image_path: str) -> str:
        """
        Process an image containing handwritten math equations and output LaTeX
        using Hugging Face Inference API
        
        Args:
            image_path: Path to the image file
            
        Returns:
            LaTeX representation of the equation
        """
        try:
            # Open the image using PIL
            img = Image.open(image_path).convert("RGB")
            
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
            # Assuming the model returns a structure with generated text
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    latex_text = result[0]['generated_text']
                else:
                    latex_text = str(result[0])
            elif isinstance(result, dict) and 'generated_text' in result:
                latex_text = result['generated_text']
            else:
                latex_text = str(result)
            
            # Log the OCR processing
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self._save_correction_data(image_path, latex_text, timestamp)
            
            return latex_text
        
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error: {str(e)}")
            return f"Error: {str(e)}"
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return f"Error: {str(e)}"
    
    def _save_correction_data(self, image_path: str, latex_text: str, timestamp: str):
        """
        Save the OCR processing data for future model improvement
        
        Args:
            image_path: Path to the processed image
            latex_text: Generated LaTeX text
            timestamp: Processing timestamp
        """
        correction_data = {
            "image_path": image_path,
            "latex_text": latex_text,
            "timestamp": timestamp,
            "corrected": False
        }
        
        # Create a unique filename for the correction data
        filename = f"{self.corrections_path}/correction_{os.path.basename(image_path)}_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(correction_data, f, indent=2)