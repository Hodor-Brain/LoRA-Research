# Main entry point for the LoRA Co-Serving system
import logging
import sys
import time
import threading
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
from prioritization.stagnation_aware import ForwardStagnationAwareStrategy, ReverseStagnationAwareStrategy
from prioritization.least_progress_first import LeastProgressFirstStrategy
from core.controller import MainController
from core.training import setup_lora_training

if __name__ == "__main__":
    # 1. Setup Logger
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

    # 6. Initialize Prioritization Strategy based on Config
    logger.info("Initializing prioritization strategy based on config...")
    strategy_name = config.prioritization.strategy
    strategy_params = config.prioritization.params if hasattr(config.prioritization, 'params') else {}

    if strategy_name == "RoundRobin":
        prioritization_strategy = RoundRobinStrategy(**strategy_params)
        logger.info(f"Using RoundRobin strategy.")
    elif strategy_name == "ForwardStagnationAware":
        prioritization_strategy = ForwardStagnationAwareStrategy(**strategy_params)
        logger.info(f"Using ForwardStagnationAware strategy with params: {strategy_params}")
    elif strategy_name == "ReverseStagnationAware":
        prioritization_strategy = ReverseStagnationAwareStrategy(**strategy_params)
        logger.info(f"Using ReverseStagnationAware strategy with params: {strategy_params}")
    elif strategy_name == "LeastProgressFirst":
        prioritization_strategy = LeastProgressFirstStrategy(**strategy_params)
        logger.info(f"Using LeastProgressFirst strategy with params: {strategy_params}")
    else:
        logger.warning(f"Unknown prioritization strategy '{strategy_name}' configured. Falling back to RoundRobin.")
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

    # 8. Submit Initial Dummy Jobs/Requests
    logger.info("Submitting initial dummy jobs/requests...")
    
    dummy_training_job_1 = {
        "job_id": "dummy_train_job_1",
        "base_model_id": config.model.name,
        "dataset_ref": "dummy_dataset_id_1",
        "max_steps": 10,
        "training_params": {
            "lr": 5e-5,
            "batch_size": 1,
        },
        "adapter_save_path": "./adapters/dummy_train_job_1"
    }
    queue_manager.add_training_job(dummy_training_job_1)
    logger.info(f"Submitted dummy training job: {dummy_training_job_1['job_id']}")

    dummy_training_job_2 = {
        "job_id": "dummy_train_job_2",
        "base_model_id": config.model.name,
        "dataset_ref": "dummy_dataset_id_2",
        "max_steps": 15,
        "training_params": {
            "lr": 3e-5,
            "batch_size": 1,
        },
        "adapter_save_path": "./adapters/dummy_train_job_2"
    }
    queue_manager.add_training_job(dummy_training_job_2)
    logger.info(f"Submitted dummy training job: {dummy_training_job_2['job_id']}")

    dummy_inference_req_base = {
        "request_id": "dummy_inf_req_base_1",
        "prompt": "What is the capital of France?",
        "adapter_path": None
    }
    queue_manager.add_inference_request(dummy_inference_req_base)
    logger.info(f"Submitted dummy inference request (base): {dummy_inference_req_base['request_id']}")
    
    logger.info("Starting controller loop in background thread...")
    controller_thread = threading.Thread(target=controller.run_loop, daemon=True)
    controller_thread.start()
    logger.info("Controller thread started.")

    WAIT_TIME_SECONDS = 5
    logger.info(f"Main thread sleeping for {WAIT_TIME_SECONDS} seconds to allow initial jobs to progress...")
    time.sleep(WAIT_TIME_SECONDS)
    logger.info("Resuming main thread. Submitting additional jobs/requests.")

    dummy_training_job_3 = {
        "job_id": "dummy_train_job_3",
        "base_model_id": config.model.name,
        "dataset_ref": "dummy_dataset_id_3",
        "max_steps": 6,
        "training_params": {
            "lr": 4e-5,
            "batch_size": 1,
        },
        "adapter_save_path": "./adapters/dummy_train_job_3"
    }
    queue_manager.add_training_job(dummy_training_job_3)
    logger.info(f"Submitted dynamic training job: {dummy_training_job_3['job_id']}")

    dummy_inference_req_adapter_1 = {
        "request_id": "dummy_inf_req_adapter_1_post",
        "prompt": "Tell me about the first dummy training job.",
        "adapter_path": "./adapters/dummy_train_job_1"
    }
    queue_manager.add_inference_request(dummy_inference_req_adapter_1)
    logger.info(f"Submitted post-training inference request (adapter 1): {dummy_inference_req_adapter_1['request_id']}")

    dummy_inference_req_adapter_2 = {
        "request_id": "dummy_inf_req_adapter_2_post",
        "prompt": "Tell me about the second dummy training job.",
        "adapter_path": "./adapters/dummy_train_job_2"
    }
    queue_manager.add_inference_request(dummy_inference_req_adapter_2)
    logger.info(f"Submitted post-training inference request (adapter 2): {dummy_inference_req_adapter_2['request_id']}")

    logger.info("Main thread waiting indefinitely to observe controller. Use Ctrl+C to exit.")
    try:
        while controller_thread.is_alive():
             time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received in main thread. Requesting controller stop...")
        controller.stop_loop()
        logger.info("Waiting for controller thread to finish...")
        controller_thread.join(timeout=5.0)
        if controller_thread.is_alive():
             logger.warning("Controller thread did not stop gracefully within timeout.")
    finally:
        logger.info("System shutting down from main thread.")
