# OCR Agent for Mathematical Equations

A specialized AI agent that processes images containing mathematical equations, extracts LaTeX using OCR, and verifies syntax correctness.

## Features

- **Image Processing**: Extract mathematical formulas from images using OCR technology
- **LaTeX Syntax Verification**: Check basic syntax of extracted LaTeX expressions
- **Interactive Correction**: Allow users to correct OCR results, with the corrections saved for future OCR improvements
- **Step-by-Step Workflow**: Guide users through a structured process for formula extraction and validation

## Workflow

1. User uploads an image containing mathematical equations
2. OCR Agent processes the image and extracts LaTeX
3. The extracted LaTeX is shown to the user for verification
4. User can accept or modify the extracted LaTeX
5. If modified, the system saves the correction for future OCR training
6. The system verifies the syntax of the LaTeX expression
7. If syntax errors are found, the agent explains them and suggests fixes
8. Once syntax is correct, the verified LaTeX is ready for mathematical processing

## Requirements

- Python 3.10+
- OpenAI API key
- Dependencies listed in `pyproject.toml`

## Usage

To start the OCR Agent server:

```bash
python -m app.api.agents.ocr_agent --host localhost --port 8001
```

## API

The agent exposes a RESTful API that follows the Agent-to-Agent (A2A) protocol for integration with other systems. It supports both synchronous and streaming interaction modes.

## Integration with Math Agent

The OCR Agent works seamlessly with the Math Agent in a pipeline:
1. OCR Agent extracts and syntax-checks LaTeX from images
2. Math Agent handles context validation and mathematical problem-solving

This separation of concerns allows each agent to focus on its specific expertise, resulting in more accurate and efficient processing.

## License

MIT 