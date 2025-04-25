import os
import json
import logging
import re
from datetime import datetime
from typing import Tuple, Optional
from openai import OpenAI
from core.config import settings
from core.exceptions import ApiKeyNotFoundError

# Add these imports
import io
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from PIL import Image

class LaTeXService:
    """Service for LaTeX verification and processing"""
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.output_dir = settings.OUTPUT_DIR
        self.latex_errors_path = settings.LATEX_ERRORS_DIR
        self.rendered_path = settings.RENDERED_LATEX_DIR
        self.verification_path = settings.VERIFICATION_DIR
        
        # We don't need to create directories here as they're already created in config.py
    
    def render(self, latex: str) -> str:
        """
        Render LaTeX to an image
        
        Args:
            latex: LaTeX code to render
            
        Returns:
            Path to rendered image
        """
        try:
            # Configure matplotlib for LaTeX rendering
            rcParams['text.usetex'] = True
            rcParams['font.family'] = 'serif'
            rcParams['font.serif'] = ['Computer Modern Roman']
            rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts}'
            
            # Create figure with transparent background
            fig = plt.figure(figsize=(10, 2))
            fig.patch.set_alpha(0)
            
            # Plot the LaTeX expression
            plt.text(0.5, 0.5, f"${latex}$", 
                    fontsize=14, ha='center', va='center')
            plt.axis('off')
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       pad_inches=0.1, transparent=True)
            plt.close(fig)
            
            # Reset matplotlib settings to default
            rcParams['text.usetex'] = False
            
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"latex_render_{timestamp}.png"
            file_path = os.path.join(self.rendered_path, filename)
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            return file_path
            
        except Exception as e:
            logging.error(f"Error rendering LaTeX: {str(e)}")
            
            # Generate error image with text
            fig = plt.figure(figsize=(10, 2))
            plt.text(0.5, 0.5, f"Error rendering LaTeX: {str(e)}", 
                    fontsize=12, ha='center', va='center', color='red')
            plt.axis('off')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Create a unique filename for error
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"latex_render_error_{timestamp}.png"
            file_path = os.path.join(self.rendered_path, filename)
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            return file_path
    
    def syntax_verify(self, latex: str) -> Tuple[bool, Optional[str]]:
        """
        Verify if the LaTeX expression has correct syntax
        
        Args:
            latex: The LaTeX expression to verify
            
        Returns:
            Tuple of (is_valid, correction)
        """
        try:
            # Perform basic syntax check
            is_valid, correction = self._check_syntax(latex)
            
            # Log the verification
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self._save_verification_data(
                latex, 
                is_valid, 
                correction, 
                timestamp,
                verification_type="syntax"
            )
            
            return is_valid, correction
                
        except Exception as e:
            logging.error(f"Error verifying LaTeX syntax: {str(e)}")
            return False, f"Error during syntax verification: {str(e)}"
    
    def context_verify(self, latex: str) -> Tuple[bool, Optional[str]]:
        """
        Verify if the LaTeX expression is correct in context
        
        Args:
            latex: The LaTeX expression to verify
            
        Returns:
            Tuple of (is_valid, correction)
        """
        try:
            # First do a basic syntax check
            basic_valid, basic_correction = self._check_syntax(latex)
            if not basic_valid:
                # Log the verification failure at syntax level
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                self._save_verification_data(
                    latex, 
                    False, 
                    basic_correction, 
                    timestamp,
                    verification_type="context_failed_at_syntax"
                )
                return False, basic_correction
            
            # If syntax is okay, use the LLM for context verification
            if self.api_key:
                return self._llm_context_verify(latex)
            else:
                # Fallback to rule-based verification if no API key
                return self._rule_based_verify(latex)
                
        except Exception as e:
            logging.error(f"Error verifying LaTeX context: {str(e)}")
            return False, f"Error during context verification: {str(e)}"
    
    def _check_syntax(self, latex: str) -> Tuple[bool, Optional[str]]:
        """
        Check basic LaTeX syntax
        
        Args:
            latex: The LaTeX expression to check
            
        Returns:
            Tuple of (is_valid, correction)
        """
        # Check for balanced braces
        if latex.count('{') != latex.count('}'):
            return False, "Unbalanced braces: The number of opening '{' and closing '}' braces don't match"
        
        # Check for balanced environment tags
        begin_envs = re.findall(r'\\begin\{([^}]+)\}', latex)
        end_envs = re.findall(r'\\end\{([^}]+)\}', latex)
        
        if len(begin_envs) != len(end_envs):
            return False, "Unbalanced environment tags: The number of \\begin and \\end tags don't match"
        
        for env in begin_envs:
            if env not in end_envs:
                return False, f"Missing \\end{{{env}}}: Environment {env} is not closed"
        
        # Check for common LaTeX command errors
        common_errors = {
            r'\\fract{': '\\frac{',
            r'\\integal': '\\integral',
            r'\\sumation': '\\sum',
            r'\\aplha': '\\alpha',
            r'\\bta': '\\beta',
            r'\\lamda': '\\lambda'
        }
        
        for error, correction in common_errors.items():
            if re.search(error, latex):
                corrected = re.sub(error, correction, latex)
                return False, f"Command error: {error} should be {correction}. Suggested correction: {corrected}"
        
        return True, None
    
    def _rule_based_verify(self, latex: str) -> Tuple[bool, Optional[str]]:
        """
        Verify the LaTeX in context using rule-based checks (fallback method)
        
        Args:
            latex: The LaTeX expression to verify
            
        Returns:
            Tuple of (is_valid, correction)
        """
        # Check division by zero
        if re.search(r'\\frac\{[^}]+\}\{0\}', latex):
            return False, "Mathematical error: Division by zero detected"
        
        # Check for missing multiplication signs
        if re.search(r'[0-9][a-zA-Z]', latex):
            corrected = re.sub(r'([0-9])([a-zA-Z])', r'\1 \cdot \2', latex)
            return False, f"Missing multiplication sign. Suggested correction: {corrected}"
        
        # All other checks passed
        return True, None
    
    def _llm_context_verify(self, latex: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the LaTeX is correct in context using LLM
        
        Args:
            latex: LaTeX code to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.api_key:
            raise ApiKeyNotFoundError("API key is missing! Please set it before using the client.")
        
        client = OpenAI(api_key=self.api_key)
        
        prompt = f"""
Instructions:
- Here is the user's handwritten math solution, formatted in LaTeX.
- Only verify if the LaTeX format appears natural and appropriate in context.
- Do not evaluate whether the solution is mathematically correct.
- Only check if there is any LaTeX formatting that is significantly inconsistent with the context of the user's handwritten solution.

Respond in the following format exactly:
Output: correct
(or)
Output: incorrect #### <corrected LaTeX>

User's LaTeX-formatted solution: {latex}
"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a LaTeX format validator. Your job is to check whether the given LaTeX-formatted math solution appears natural and consistent with the context. Do not evaluate whether the math is correct—only verify the LaTeX formatting in context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content

        # Log the verification
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if content.lower().startswith("correct"):
            is_valid = True
            correction = None
        elif content.lower().startswith("incorrect"):
            is_valid = False
            # Extract everything after '####'
            match = re.search(r"\s*incorrect\s*####\s*(.+)", content, re.IGNORECASE | re.DOTALL)
            if match:
                correction = match.group(1).strip()
            else:
                correction = "Output marked as incorrect but correction could not be parsed."
        else:
            is_valid = False
            correction = "Unexpected response format."
        
        # Save verification data
        self._save_verification_data(
            latex, 
            is_valid, 
            correction, 
            timestamp,
            verification_type="context"
        )
        
        return is_valid, correction
    
    def _save_verification_data(self, latex: str, is_valid: bool, correction: Optional[str], timestamp: str, verification_type: str = "unknown"):
        """
        Save the LaTeX verification data for future reference
        
        Args:
            latex: The verified LaTeX
            is_valid: Whether the LaTeX is valid
            correction: Suggested correction if not valid
            timestamp: Verification timestamp
            verification_type: Type of verification performed
        """
        verification_data = {
            "latex": latex,
            "is_valid": is_valid,
            "correction": correction,
            "timestamp": timestamp,
            "verification_type": verification_type
        }
        
        # Create filename
        if is_valid:
            filename = f"{self.verification_path}/valid_{verification_type}_{timestamp}.json"
        else:
            filename = f"{self.latex_errors_path}/invalid_{verification_type}_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(verification_data, f, indent=2)