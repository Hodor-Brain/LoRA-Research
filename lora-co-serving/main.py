# Main entry point for the LoRA Co-Serving system
import logging
import sys
import time
from dataclasses import asdict

from utils.logger_config import setup_logger
from utils.config_loader import load_config, SystemConfig
from core.models import load_base_model
import torch

from core.engine import InferenceEngine
from managers.queue_manager import QueueManager
from managers.adapter_manager import LoRAAdapterManager
from managers.job_manager import ActiveTrainingJobManager
from prioritization.round_robin import RoundRobinStrategy
from core.controller import MainController
from core.training import setup_lora_training

if __name__ == "__main__":
    # 1. Setup Logger
    # Explicitly set log level to DEBUG to see detailed logs
    setup_logger(log_level=logging.DEBUG) 
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

    # 4. Prepare Base Model for LoRA Training (Important step)
    logger.info("Setting up base model for LoRA training...")
    try:
        lora_config_dict = asdict(config.training.lora_config)
        logger.info(f"Using LoRA config dict: {lora_config_dict}")
        model = setup_lora_training(model, tokenizer, lora_config_dict)
        logger.info("Base model prepared for LoRA training.")
    except Exception as e:
        logger.error(f"Failed to setup base model for LoRA training: {e}", exc_info=True)
        sys.exit(1)


    # 5. Initialize Managers and Engine
    logger.info("Initializing managers and inference engine...")
    queue_manager = QueueManager(config.queue.max_inference_queue_size, config.queue.max_training_queue_size)
    adapter_manager = LoRAAdapterManager(model, config.managers.adapter_cache_size)
    job_manager = ActiveTrainingJobManager()
    engine = InferenceEngine(model, tokenizer, adapter_manager, device)
    logger.info("Managers and Inference Engine initialized.")

    # 6. Initialize Prioritization Strategy
    logger.info("Initializing prioritization strategy...")
    prioritization_strategy = RoundRobinStrategy()
    logger.info("Prioritization Strategy initialized.")

    # 7. Initialize Main Controller
    logger.info("Initializing Main Controller...")
    controller = MainController(
        config=config,
        engine=engine,
        job_manager=job_manager,
        adapter_manager=adapter_manager,
        queue_manager=queue_manager,
        prioritization_strategy=prioritization_strategy,
        device=device
    )
    logger.info("Main Controller initialized.")

    # 8. [Optional] Submit Dummy Jobs/Requests for Testing
    logger.info("Submitting dummy jobs/requests for testing...")
    dummy_training_job = {
        "job_id": "dummy_train_job_1",
        "base_model_id": config.model.name,
        "dataset_ref": "dummy_dataset_id",
        "max_steps": 10,
        "training_params": {
            "lr": 5e-5,
            "batch_size": 1,
        },
        "adapter_save_path": "./adapters/dummy_train_job_1"
    }
    queue_manager.add_training_job(dummy_training_job)
    logger.info(f"Submitted dummy training job: {dummy_training_job['job_id']}")

    dummy_inference_req_base = {
        "request_id": "dummy_inf_req_base_1",
        "prompt": "What is the capital of France?",
        "adapter_path": None
    }
    queue_manager.add_inference_request(dummy_inference_req_base)
    logger.info(f"Submitted dummy inference request (base): {dummy_inference_req_base['request_id']}")
    
    dummy_inference_req_adapter = {
        "request_id": "dummy_inf_req_adapter_1",
        "prompt": "Tell me about the dummy training job.",
        "adapter_path": "./adapters/dummy_train_job_1"
    }
    time.sleep(1) 
    queue_manager.add_inference_request(dummy_inference_req_adapter)
    logger.info(f"Submitted dummy inference request (adapter): {dummy_inference_req_adapter['request_id']}")


    # 9. Start the Controller Loop
    logger.info("Starting the Main Controller loop...")
    try:
        controller.run_loop()
    except Exception as e:
        logger.critical(f"Critical error during controller loop execution: {e}", exc_info=True)
    finally:
        logger.info("System shutting down.")
