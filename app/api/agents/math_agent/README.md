# Math Agent

A specialized AI agent for solving mathematical problems and verifying LaTeX expressions. The Math Agent leverages large language models to help users solve mathematical equations, verify the correctness of LaTeX syntax, and provide step-by-step solutions to complex math problems.

## Features

- **Math Problem Solving**: Solves various mathematical problems including algebra, calculus, and more
- **LaTeX Verification**: Verifies and corrects LaTeX expressions
- **Context Verification**: Validates LaTeX within the context of mathematical problems
- **Streaming Support**: Provides real-time responses as the agent processes information

## Usage

To start the Math Agent server:

```bash
python -m app.api.agents.math_agent --host localhost --port 10000
```

## Requirements

- Python 3.10+
- OpenAI API key
- Dependencies listed in `pyproject.toml`

## API

The agent exposes a RESTful API that follows the Agent-to-Agent (A2A) protocol for integration with other systems. It supports both synchronous and streaming interaction modes.

## License

MIT 