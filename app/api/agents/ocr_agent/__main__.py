from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from common.utils.push_notification_auth import PushNotificationSenderAuth
from agents.ocr_agent.task_manager import AgentTaskManager
from agents.ocr_agent.agent import OCRAgent
from app.core.config import settings
import click
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default=settings.API_HOST)
@click.option("--port", "port", default=settings.API_PORT + 1)  # Use different port than main API
def main(host, port):
    """Starts the OCR agent server."""
    try:
        if not settings.OPENAI_API_KEY:
            raise MissingAPIKeyError("OPENAI_API_KEY is not set in settings.")
        
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id="ocr_math",
            description="Process images of mathematical equations, extract LaTeX with OCR, and verify syntax and context",
            tags=["ocr", "latex", "math", "equation"],
            examples=[
                "Extract LaTeX from this math equation image",
                "Check if this LaTeX syntax is correct: \\frac{x^2}{y}",
                "Verify this equation and explain any errors"
            ]
        )
        agent_card = AgentCard(
            name=f"{settings.PROJECT_NAME} - OCR Agent",
            description="Process images of mathematical equations, extract LaTeX using OCR, and verify syntax and context",
            url=f"http://{host}:{port}/",
            version=settings.VERSION,
            defaultInputModes=OCRAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=OCRAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=OCRAgent(),
                notification_sender_auth=notification_sender_auth
                ),
            host=host,
            port=port,
        )

        server.app.route(
            "/.well-known/jwks.json", notification_sender_auth.handle_jwks_endpoint, methods=["GET"]
        )

        logger.info(f"Starting {settings.PROJECT_NAME} OCR agent server on {host}:{port}")
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    main() 