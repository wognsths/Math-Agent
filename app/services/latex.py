import os
import re
import tempfile
import subprocess
from typing import Optional, Tuple
import logging
from core.config import settings
from openai import OpenAI
from core.exceptions import ApiKeyNotFoundError


class LaTeXService:
    """
    Service for rendering LaTeX equations
    """
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
    
    def render(self, latex: str) -> str:
        """
        Render LaTeX to a user-friendly format (HTML representation)
        
        Args:
            latex: LaTeX code to render
            
        Returns:
            HTML representation of the rendered LaTeX
        """
        # Wrap the LaTeX in MathJax compatible format
        mathjax_latex = f"""
        <div class="math-container">
            <span class="math">$${latex}$$</span>
        </div>
        """
        
        return mathjax_latex
    
    def syntax_verify(self, latex: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the LaTeX is syntactically correct
        
        Args:
            latex: LaTeX code to validate
            
        Returns:
            True if valid, False otherwise
        """
        full_latex = f"""
        \\documentclass{{article}}
        \\usepackage{{amsmath,amsfonts}}
        \\begin{{document}}
        {latex}
        \\end{{document}}
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "temp.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(full_latex)

            try:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=tmpdir,
                    timeout=10
                )
                return result.returncode == 0
            except Exception as e:
                return False, f"Error: {e}"

        return True, None
    
    def context_verify(self, latex: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the LaTeX is correct in context
        
        Args:
            latex: LaTeX code to validate
            
        Returns:
            True if valid, False otherwise
        """
        if self.api_key:
            client = OpenAI(api_key=self.api_key)
        else:
            raise ApiKeyNotFoundError("API key is missing! Please set it before using the client.")
        
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
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a LaTeX format validator. Your job is to check whether the given LaTeX-formatted math solution appears natural and consistent with the context. Do not evaluate whether the math is correct—only verify the LaTeX formatting in context."},
                {"role": "user", "content": prompt}
            ],
            reasoning={"effort": "medium"},
            temperature=0.1
        )

        content = response.choices[0].message.content

        if content.lower().startswith("correct"):
            return True, None
        elif content.lower().startswith("incorrect"):
            # Extract everything after '####'
            match = re.search(r"\s*incorrect\s*####\s*(.+)", content, re.IGNORECASE | re.DOTALL)
            if match:
                correction = match.group(1).strip()
                return False, correction
            else:
                return False, "Output marked as incorrect but correction could not be parsed."
        else:
            return False, "Unexpected response format."