# Main entry point for the LoRA Co-Serving system
import logging
import sys

from utils.logger_config import setup_logger
from utils.config_loader import load_config, SystemConfig
from core.models import load_base_model
import torch

if __name__ == "__main__":
    # 1. Setup Logger
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Logging configured.")

    # 2. Load Configuration
    config: SystemConfig = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        sys.exit(1)
    logger.info(f"Configuration loaded: {config}")

    # 3. Load Base Model (using config)
    logger.info(f"Loading base model: {config.model.name}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Target device: {device}")
    model, tokenizer = load_base_model(config.model.name, device=device)

    if model is None or tokenizer is None:
        logger.error("Failed to load base model. Exiting.")
        sys.exit(1)
    logger.info("Base model and tokenizer loaded successfully.")


    # TODO: Phase 3 onwards - Initialize Managers
    # TODO: Phase 3 onwards - Initialize Inference Engine
    # TODO: Phase 4 onwards - Initialize Prioritization Strategy
    # TODO: Phase 4 onwards - Initialize Main Controller
    # TODO: Phase 4 onwards - Start the controller loop

    logger.info("System initialization complete (Phase 1). Shutting down for now.")
    pass # End of initial setup 