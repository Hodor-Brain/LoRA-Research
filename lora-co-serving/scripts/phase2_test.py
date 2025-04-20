# Script for testing Phase 2 functionality (Basic Training - Single Adapter)
import sys
import os
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from utils.logger_config import setup_logger
from utils.config_loader import load_config
from core.models import load_base_model
from core.training import setup_lora_training, execute_training_step, SimpleTextDataset

if __name__ == "__main__":
    # --- Setup ---
    setup_logger(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(" === Running Phase 2 Test === ")

    # --- Load Config ---
    logger.info("Loading configuration...")
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    if not config:
        logger.error(f"Failed to load config from {config_path}. Exiting.")
        sys.exit(1)
    logger.info(f"Configuration loaded successfully.")

    # --- Load Base Model ---
    logger.info(f"Loading base model: {config.model.name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    base_model, tokenizer = load_base_model(config.model.name, device=device)
    if not base_model or not tokenizer:
        logger.error("Failed to load base model or tokenizer. Exiting.")
        sys.exit(1)
    logger.info("Base model and tokenizer loaded.")

    # --- Prepare Dummy Data ---
    dummy_texts = [
        "This is the first sentence for training.",
        "Here is another example sentence.",
        "LoRA training requires text data.",
        "This is just a small dummy dataset.",
        "The Qwen model will learn from this.",
        "Hopefully the training step works.",
        "One more sentence for the batch.",
        "Final example text for training this adapter."
    ]
    logger.info(f"Using {len(dummy_texts)} dummy text samples for training.")
    train_dataset = SimpleTextDataset(dummy_texts, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    logger.info("Dummy dataset and dataloader created.")

    # --- Setup LoRA Training ---
    lora_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
    }
    logger.info(f"Setting up LoRA with config: {lora_config}")
    peft_model = setup_lora_training(base_model, tokenizer, lora_config)
    if not peft_model:
        logger.error("Failed to setup LoRA training. Exiting.")
        sys.exit(1)

    # --- Setup Optimizer ---
    optimizer = AdamW(peft_model.parameters(), lr=5e-5)
    logger.info(f"Optimizer AdamW initialized.")

    # --- Run Training Steps ---
    num_training_steps = 5
    logger.info(f"Running {num_training_steps} training steps...")
    step_losses = []
    peft_model.train()

    step_count = 0
    for batch in train_dataloader:
        if step_count >= num_training_steps:
            break
        
        loss = execute_training_step(peft_model, optimizer, batch, device)
        step_count += 1

        if loss is not None:
            logger.info(f"Step {step_count}/{num_training_steps}, Loss: {loss:.4f}")
            step_losses.append(loss)
        else:
            logger.error(f"Training step {step_count} failed.")
            sys.exit(1)
    
    if len(step_losses) < num_training_steps:
         logger.warning("Did not complete all requested training steps.")

    logger.info("Training steps completed.")

    # --- Save Adapter ---
    output_adapter_path = "./adapters/phase2_test_adapter"
    logger.info(f"Saving trained adapter to: {output_adapter_path}")
    try:
        peft_model.save_pretrained(output_adapter_path)
        logger.info(f"Adapter saved successfully.")
        tokenizer.save_pretrained(output_adapter_path)
        logger.info(f"Tokenizer saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save adapter and/or tokenizer: {e}", exc_info=True)

    logger.info(" === Phase 2 Test Completed === ") 