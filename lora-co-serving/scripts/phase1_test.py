# Script for testing Phase 1 functionality (Setup & Basic Inference)
import sys
import os
import logging
import torch

from utils.logger_config import setup_logger
from utils.config_loader import load_config
from core.models import load_base_model
from managers.adapter_manager import LoRAAdapterManager

if __name__ == "__main__":
    # --- Setup ---
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info(" === Running Phase 1 Test === ")

    # --- Load Config ---
    logger.info("Loading configuration...")
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    if not config:
        logger.error(f"Failed to load config from {config_path}. Exiting.")
        sys.exit(1)
    logger.info(f"Configuration loaded successfully.")

    # --- 1. Load Base Model ---
    logger.info(f"Attempting to load base model: {config.model.name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    model, tokenizer = load_base_model(config.model.name, device=device)

    if model and tokenizer:
        logger.info("Successfully loaded base model and tokenizer.")
        logger.info(f"Model Class: {model.__class__.__name__}")
        logger.info(f"Tokenizer Class: {tokenizer.__class__.__name__}")
    else:
        logger.error("Failed to load base model or tokenizer.")
        sys.exit(1)

    # --- 2. Load a single pre-trained LoRA adapter ---
    logger.info("Initializing LoRA Adapter Manager...")
    adapter_cache_size = getattr(config.managers, 'adapter_cache_size', 3)
    adapter_manager = LoRAAdapterManager(base_model=model, cache_size=adapter_cache_size)
    logger.info(f"Adapter manager initialized (Cache Size: {adapter_cache_size}).")

    placeholder_adapter_path = "adapters/my_test_adapter"
    logger.info(f"Attempting to load placeholder adapter from: {placeholder_adapter_path}")
    
    loaded_model_with_adapter = adapter_manager.load_inference_adapter(placeholder_adapter_path)

    if loaded_model_with_adapter:
        logger.info(f"Adapter loading call completed (check logs for success/warnings).")
    else:
        logger.error(f"Failed during adapter loading call for {placeholder_adapter_path}. Exiting.")
        sys.exit(1)

    # --- Placeholder for future steps --- 
    # 3. Run basic inference (non-batched)
    logger.info("Skipping non-batched inference test for now.")

    # 4. Run batched inference (same adapter)
    logger.info("Skipping batched inference test for now.")

    logger.info(" === Phase 1 Test Partial Completion (Base Model Load) === ")

    print("Phase 1 Test Complete.") 