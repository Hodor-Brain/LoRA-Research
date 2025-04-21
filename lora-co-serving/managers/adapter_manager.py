# Manager for Loading/Unloading/Caching LoRA Adapters

import logging
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import PreTrainedModel
from typing import Optional, Tuple, Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
OPTIMIZER_NAME = "optimizer.pt"

class LoRAAdapterManager:
    def __init__(self, base_model: PreTrainedModel, cache_size: int = 3):
        self.base_model = base_model
        self.cache_size = max(1, cache_size)
        self.inference_cache: OrderedDict[str, PeftModel] = OrderedDict()
        self.active_training_adapter_job_id: Optional[str] = None
        logger.info(f"LoRAAdapterManager initialized with cache size: {self.cache_size}")

    def _evict_lru_adapter(self):
        """Evicts the least recently used adapter from the cache."""
        if len(self.inference_cache) >= self.cache_size:
            evicted_path, evicted_model = self.inference_cache.popitem(last=False)
            logger.info(f"Inference cache full. Evicting LRU adapter: {evicted_path}")

    def load_inference_adapter(self, adapter_path: str) -> Optional[PeftModel]:
        """Loads a LoRA adapter for inference using LRU caching.

        Checks cache first. If found, moves to end (most recent). 
        If not found, loads from disk, adds to cache, and evicts LRU if cache is full.
        Explicitly checks for local files before attempting to load.

        Args:
            adapter_path (str): The local directory path containing the adapter files.

        Returns:
            Optional[PeftModel]: The loaded PeftModel (base model + adapter), or the base model if loading skipped/failed.
        """
        if adapter_path in self.inference_cache:
            logger.debug(f"Adapter {adapter_path} found in cache. Marking as recently used.")
            self.inference_cache.move_to_end(adapter_path)
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
            logger.warning(f"Adapter loading skipped: No adapter weights ({ADAPTER_WEIGHTS_NAME} or {ADAPTER_SAFE_WEIGHTS_NAME}) found in {adapter_path}. Returning base model (not caching failure).")
            return self.base_model

        logger.info(f"Loading LoRA adapter from local path: {adapter_path}")
        try:
            adapter_name_for_peft = adapter_path
            peft_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=adapter_name_for_peft,
                is_local=True
            )
            peft_model.eval()

            logger.info(f"Successfully loaded adapter '{adapter_name_for_peft}' from {adapter_path}. Attached to base model.")

            self._evict_lru_adapter()

            self.inference_cache[adapter_path] = peft_model
            self.inference_cache.move_to_end(adapter_path)

            return peft_model

        except Exception as e:
            logger.error(f"Failed to load LoRA adapter from {adapter_path} even after file checks: {e}", exc_info=True)
            return None

    def load_training_state(self, job_id: str, adapter_path: str) -> Tuple[Optional[PeftModel], Optional[Dict[str, Any]]]:
        """Loads adapter weights and optimizer state for a specific training job.

        Args:
            job_id (str): The unique identifier for the training job (used as adapter_name).
            adapter_path (str): The directory where adapter files and optimizer state are stored.

        Returns:
            Tuple[Optional[PeftModel], Optional[Dict[str, Any]]]: 
                A tuple containing the PeftModel with the adapter loaded and set active,
                and the optimizer state dictionary (or None if not found/error).
        """
        logger.info(f"Attempting to load training state for job '{job_id}' from: {adapter_path}")
        optimizer_state_dict = None
        loaded_peft_model = None

        optimizer_path = os.path.join(adapter_path, OPTIMIZER_NAME)

        try:
            config_path = os.path.join(adapter_path, CONFIG_NAME)
            if not os.path.isfile(config_path):
                 logger.warning(f"Training state load: Config file not found at {config_path}. Cannot load adapter.")

            logger.info(f"Loading adapter weights for '{job_id}' from {adapter_path}...")
            self.base_model.load_adapter(adapter_path, adapter_name=job_id, is_local=True)
            logger.info(f"Adapter weights for '{job_id}' loaded.")
            self.base_model.set_adapter(job_id)
            logger.info(f"Adapter '{job_id}' set as active for training.")
            loaded_peft_model = self.base_model 
            loaded_peft_model.train()
            self.active_training_adapter_job_id = job_id

            if os.path.isfile(optimizer_path):
                logger.info(f"Loading optimizer state from: {optimizer_path}")
                optimizer_state_dict = torch.load(optimizer_path, map_location='cpu')
                logger.info(f"Optimizer state loaded successfully for job '{job_id}'.")
            else:
                logger.info(f"Optimizer state file not found at {optimizer_path}. Starting optimizer from scratch.")

            return loaded_peft_model, optimizer_state_dict

        except FileNotFoundError:
            logger.warning(f"Adapter weights ({ADAPTER_WEIGHTS_NAME} or {ADAPTER_SAFE_WEIGHTS_NAME}) not found in {adapter_path}. Assuming first run for job '{job_id}'.")
            try:
                self.base_model.set_adapter(job_id)
                logger.info(f"Adapter '{job_id}' (no weights loaded) set as active for training.")
                self.active_training_adapter_job_id = job_id
                return self.base_model, None
            except Exception as inner_e:
                 logger.error(f"Failed to set adapter '{job_id}' even without weights: {inner_e}", exc_info=True)
                 return None, None
        except Exception as e:
            logger.error(f"Error loading training state for job '{job_id}' from {adapter_path}: {e}", exc_info=True)
            return None, None

    def save_training_state(self, job_id: str, model_to_save: PeftModel, optimizer_state_dict: Dict[str, Any], adapter_save_path: str):
        """Saves the adapter weights and optimizer state for a training job.

        Args:
            job_id (str): The job ID (used as the adapter name being saved).
            model_to_save (PeftModel): The current PeftModel instance containing the trained adapter.
            optimizer_state_dict (Dict[str, Any]): The state dictionary from the optimizer.
            adapter_save_path (str): The directory where adapter files and optimizer state should be saved.
        """
        logger.info(f"Attempting to save training state for job '{job_id}' to: {adapter_save_path}")
        
        if not os.path.exists(adapter_save_path):
            try:
                os.makedirs(adapter_save_path)
                logger.info(f"Created directory for saving state: {adapter_save_path}")
            except OSError as e:
                logger.error(f"Error creating directory {adapter_save_path} for saving state: {e}")
                return

        try:
            model_to_save.save_pretrained(adapter_save_path, selected_adapters=[job_id])
            logger.info(f"Adapter weights for job '{job_id}' saved successfully to {adapter_save_path}.")

            optimizer_path = os.path.join(adapter_save_path, OPTIMIZER_NAME)
            torch.save(optimizer_state_dict, optimizer_path)
            logger.info(f"Optimizer state for job '{job_id}' saved successfully to {optimizer_path}.")

        except Exception as e:
            logger.error(f"Error saving training state for job '{job_id}' to {adapter_save_path}: {e}", exc_info=True)

    def unload_training_adapter(self, job_id: str):
         logger.info(f"Unloading/disabling training adapter '{job_id}' (Placeholder)...")
         if self.active_training_adapter_job_id == job_id:
              self.active_training_adapter_job_id = None
         # Consider self.base_model.disable_adapter() ? 