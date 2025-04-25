# PDF Agent

PDF Agent is a specialized component of the Math Agent system designed to process PDF documents containing mathematical expressions. It extracts images and regions containing mathematical content, processes them with OCR, and converts them to LaTeX format for further analysis.

## Features

- Extract images from PDF documents
- Detect and isolate regions containing mathematical expressions
- Process images with OCR to extract LaTeX
- Filter and organize mathematical formulas
- Integrate with OCR and Math agents for comprehensive processing

## Architecture

The PDF Agent consists of the following components:

1. **Agent (agent.py)**: Core LLM-based agent that orchestrates the PDF processing workflow
2. **Task Manager (task_manager.py)**: Manages asynchronous tasks and handles state persistence
3. **API Server (\_\_main\_\_.py)**: FastAPI server for exposing the agent's capabilities via REST API

## Requirements

- PyMuPDF (fitz) for PDF processing
- OpenCV for image processing
- Tesseract for local OCR fallback
- FastAPI for API server
- LangChain for agent framework

## Usage

### Standalone Mode

Run the PDF Agent as a standalone service:

```bash
cd app/api/agents/pdf_agent
python -m app.api.agents.pdf_agent
```

This will start the FastAPI server on port 8001 by default.

### API Usage

To create a new PDF processing task:

```
POST /tasks
{
  "session_id": "unique-session-id",
  "content": {
    "message": {
      "parts": [
        {
          "type": "text",
          "text": "Extract math formulas from this PDF"
        },
        {
          "type": "file",
          "file": {
            "name": "math_document.pdf",
            "mimeType": "application/pdf",
            "bytes": "base64-encoded-pdf-content"
          }
        }
      ]
    }
  }
}
```

### Integration with Math Agent

The PDF Agent can be integrated with the broader Math Agent system. When math formulas are extracted, they can be automatically sent to the Math Agent for further analysis and solving.

## Output Format

The agent produces structured output containing:

- Extracted images
- Identified math formulas with LaTeX representations
- OCR processing results
- Error details (if any)

## Limitations

- Complex mathematical notations may not be perfectly recognized
- Handwritten equations have lower accuracy than printed ones
- PDF documents with non-standard encodings may not process correctly

## Future Improvements

- Support for MathML format
- Improved region detection using machine learning
- Better handling of multi-column layouts
- Support for chemical equations and diagrams