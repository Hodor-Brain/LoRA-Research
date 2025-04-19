# Manager for Loading/Unloading/Caching LoRA Adapters

import logging
import os
from peft import PeftModel, PeftConfig
from transformers import PreTrainedModel
from typing import Optional

logger = logging.getLogger(__name__)

ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"

class LoRAAdapterManager:
    def __init__(self, base_model: PreTrainedModel, cache_size: int = 3):
        self.base_model = base_model
        self.cache_size = cache_size
        self.inference_cache = {}
        self.training_adapter_state = None # (job_id, adapter, optimizer_state)
        # TODO: Implement LRU logic for cache eviction later

    def load_inference_adapter(self, adapter_path: str) -> Optional[PeftModel]:
        """Loads a LoRA adapter for inference.

        Checks cache first, then loads from disk if not found.
        Explicitly checks for local files before attempting to load.

        Args:
            adapter_path (str): The local directory path containing the adapter files.

        Returns:
            Optional[PeftModel]: The loaded PeftModel (base model + adapter), or None on error.
        """
        if adapter_path in self.inference_cache:
            logger.debug(f"Adapter {adapter_path} already loaded (cached). Returning model.")
            return self.inference_cache[adapter_path] 

        if not os.path.isdir(adapter_path):
            logger.error(f"Adapter directory not found: {adapter_path}")
            return None

        config_path = os.path.join(adapter_path, CONFIG_NAME)
        if not os.path.isfile(config_path):
            logger.error(f"Adapter config file not found: {config_path}")
            return None

        weights_path_bin = os.path.join(adapter_path, ADAPTER_WEIGHTS_NAME)
        weights_path_safe = os.path.join(adapter_path, ADAPTER_SAFE_WEIGHTS_NAME)
        has_weights = os.path.isfile(weights_path_bin) or os.path.isfile(weights_path_safe)

        if not has_weights:
            logger.warning(f"Adapter loading skipped: No adapter weights ({ADAPTER_WEIGHTS_NAME} or {ADAPTER_SAFE_WEIGHTS_NAME}) found in {adapter_path}. Returning base model.")
            self.inference_cache[adapter_path] = self.base_model 
            return self.base_model


        logger.info(f"Loading LoRA adapter from local path: {adapter_path}")
        try:
            peft_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                is_local=True 
            )
            peft_model.eval()

            logger.info(f"Successfully loaded adapter {adapter_path}. Attached to base model.")

            self.inference_cache[adapter_path] = peft_model
            # TODO: Implement proper LRU cache eviction if size > self.cache_size

            return peft_model

        except Exception as e:
            logger.error(f"Failed to load LoRA adapter from {adapter_path} even after file checks: {e}", exc_info=True)
            return None

    def load_training_state(self, job_id):
        # TODO: Implement loading adapter weights AND optimizer state for a specific training job
        logger.warning("load_training_state not implemented yet.")
        pass

    def save_training_state(self, job_id, adapter, optimizer_state):
        # TODO: Implement saving adapter weights AND optimizer state for a specific training job
        logger.warning("save_training_state not implemented yet.")
        pass 