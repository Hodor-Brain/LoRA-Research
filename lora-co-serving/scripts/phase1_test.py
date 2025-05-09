# Script for testing Phase 1 functionality (Setup & Basic Inference)
import sys
import os
import logging
import torch

from utils.logger_config import setup_logger
from utils.config_loader import load_config
from core.models import load_base_model
from managers.adapter_manager import LoRAAdapterManager
from core.engine import InferenceEngine

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

    # --- 3. Run basic inference (non-batched) ---
    logger.info("Initializing Inference Engine...")
    inference_engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        adapter_manager=adapter_manager,
        device=device
    )
    logger.info("Inference engine initialized.")

    test_prompt = "Once upon a time,"
    logger.info(f"Running single inference with prompt: '{test_prompt}' (Base Model)")
    generated_text_base = inference_engine.run_inference_single(prompt=test_prompt, adapter_path=None)

    if generated_text_base:
        logger.info(f"Generated Text (Base Model): '{generated_text_base}'")
    else:
        logger.error("Inference failed for base model.")
    
    # Test with placeholder adapter (should also use base model due to loading skip)
    logger.info(f"Running single inference with prompt: '{test_prompt}' (Placeholder Adapter Path)")
    generated_text_adapter = inference_engine.run_inference_single(prompt=test_prompt, adapter_path=placeholder_adapter_path)

    if generated_text_adapter:
        logger.info(f"Generated Text (Placeholder Adapter Path): '{generated_text_adapter}'")
    else:
        logger.error(f"Inference failed for placeholder adapter path.")

    # --- 4. Run batched inference (Mixed Adapters Test) ---
    logger.info("Testing batched inference with mixed adapters (Base and Placeholder)...")
    mixed_batch_requests = [
        {'prompt': "The capital of France is", 'adapter_path': None, 'max_new_tokens': 10}, # Base
        {'prompt': "What is PEFT?", 'adapter_path': placeholder_adapter_path, 'max_new_tokens': 25}, # Placeholder (will use base)
        {'prompt': "Tell me a joke.", 'adapter_path': None, 'max_new_tokens': 30}, # Base
        {'prompt': "Calculate 5 * 8", 'adapter_path': placeholder_adapter_path, 'max_new_tokens': 5} # Placeholder (will use base)
    ]

    mixed_batch_results = inference_engine.process_batch(mixed_batch_requests)

    if mixed_batch_results:
        logger.info("Mixed batch inference call completed.")
        for i, result in enumerate(mixed_batch_results):
            req = mixed_batch_requests[i]
            adapter_info = req['adapter_path'] if req['adapter_path'] else "Base Model"
            if result:
                logger.info(f"MixedBatch [{i}] Adapter: '{adapter_info}' Prompt: '{req['prompt']}'")
                logger.info(f"MixedBatch [{i}] Result: '{result}'")
            else:
                logger.error(f"MixedBatch [{i}] Adapter: '{adapter_info}' Prompt: '{req['prompt']}' -> FAILED")
    else:
        logger.error("Mixed batch inference call failed entirely.")

    logger.info(" === Phase 1 & 3 (Partial) Test Completed === ")

    print("Phase 1/3 Test Complete.") 