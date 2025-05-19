import logging
from typing import Dict, List

from src.agents import AgentHandler
from src.configurator import Configurator
from src.services.chroma_db import ChromaDBService


logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config: Configurator):
        self.config = config

        # Initialize class -- need to abstract these?
        logger.info("Initializing Chroma DB...")
        self.chroma_db = ChromaDBService(config=self.config,recreate_collection=True)
        self.chroma_db.initialize_db()

        logger.info("Initializing agents...")
        self.agent_handler = AgentHandler(self.config)

    def get_response(self, messages: List[Dict]) -> str:
        logger.info(f"Received input, updating memory...")
        self.agent_handler.update_memory(messages)

        # Agent flow goes here

        return ""


if __name__ == "__main__":

    config = Configurator()
    controller = Controller(config)
    controller.get_response([{"role": "user", "content": "This is a test"}])
    logger.info("Wait!")
