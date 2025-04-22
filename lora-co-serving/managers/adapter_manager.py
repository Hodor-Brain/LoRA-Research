# Manager for Loading/Unloading/Caching LoRA Adapters

import logging
import os
import torch
import json
from peft import PeftModel, PeftConfig, LoraConfig
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

        normalized_adapter_path = os.path.normpath(adapter_path)

        if not os.path.isdir(normalized_adapter_path):
            logger.error(f"Adapter directory not found: {normalized_adapter_path}")
            return None

        config_path = os.path.join(normalized_adapter_path, CONFIG_NAME)
        if not os.path.isfile(config_path):
            logger.error(f"Adapter config file not found: {config_path}")
            return None

        weights_path_bin = os.path.join(normalized_adapter_path, ADAPTER_WEIGHTS_NAME)
        weights_path_safe = os.path.join(normalized_adapter_path, ADAPTER_SAFE_WEIGHTS_NAME)
        has_weights = os.path.isfile(weights_path_bin) or os.path.isfile(weights_path_safe)

        if not has_weights:
            logger.warning(f"Adapter loading skipped: No adapter weights ({ADAPTER_WEIGHTS_NAME} or {ADAPTER_SAFE_WEIGHTS_NAME}) found in {normalized_adapter_path}. Returning base model (not caching failure).")
            return self.base_model

        logger.info(f"Loading LoRA adapter from local path: {normalized_adapter_path}")
        try:
            adapter_name_for_peft = normalized_adapter_path 
            peft_model = PeftModel.from_pretrained(
                self.base_model,
                normalized_adapter_path,
                adapter_name=adapter_name_for_peft,
                is_local=True
            )
            peft_model.eval()

            logger.info(f"Successfully loaded adapter '{adapter_name_for_peft}' from {normalized_adapter_path}. Attached to base model.")

            self._evict_lru_adapter()

            self.inference_cache[normalized_adapter_path] = peft_model
            self.inference_cache.move_to_end(normalized_adapter_path)

            return peft_model

        except Exception as e:
            logger.error(f"Failed to load LoRA adapter from {normalized_adapter_path} even after file checks: {e}", exc_info=True)
            return None

    def load_training_state(self, 
                            job_id: str, 
                            adapter_path: str, 
                            lora_config_dict: Dict[str, Any]
                           ) -> Tuple[Optional[PeftModel], Optional[Dict[str, Any]]]:
        """Loads or initializes adapter weights and optimizer state for a training job.
        Assumes adapter files are stored in a subdirectory named {job_id} under adapter_path.
        Checks if {adapter_path}/{job_id}/adapter_config.json exists.
        """
        logger.info(f"Attempting to load/init training state for job '{job_id}' from base path: {adapter_path}")
        optimizer_state_dict = None
        loaded_or_added_peft_model = None

        normalized_base_path = os.path.normpath(adapter_path)
        logger.debug(f"Normalized base path: {normalized_base_path}")

        actual_adapter_dir = os.path.join(normalized_base_path, job_id)
        logger.debug(f"Expected actual adapter directory: {actual_adapter_dir}")

        config_path = os.path.join(actual_adapter_dir, CONFIG_NAME)
        optimizer_path = os.path.join(normalized_base_path, OPTIMIZER_NAME)

        logger.debug(f"Checking for config file existence at: '{config_path}'")
        config_exists = os.path.isfile(config_path)
        logger.debug(f"os.path.isfile result: {config_exists}")

        is_first_run = not config_exists

        try:
            if is_first_run:
                logger.info(f"Config file not found at '{config_path}'. Assuming first run for job '{job_id}'. Adding adapter.")
                lora_config_dict.setdefault("task_type", "CAUSAL_LM")
                peft_config = LoraConfig(**lora_config_dict)
                
                if not isinstance(self.base_model, PeftModel):
                     logger.warning("Base model is not a PeftModel instance during add_adapter. This might indicate an issue.")

                self.base_model.add_adapter(job_id, peft_config)
                logger.info(f"Added adapter configuration '{job_id}' to the base model.")
                self.base_model.set_adapter(job_id)
                logger.info(f"Adapter '{job_id}' set as active for training (first run).")
                loaded_or_added_peft_model = self.base_model
                optimizer_state_dict = None

            else:
                logger.info(f"Config file found at '{config_path}'. Loading existing adapter weights for '{job_id}' from '{actual_adapter_dir}'...")
                self.base_model.load_adapter(actual_adapter_dir, adapter_name=job_id, is_local=True)
                logger.info(f"Adapter weights for '{job_id}' loaded successfully from {actual_adapter_dir}.")
                self.base_model.set_adapter(job_id)
                logger.info(f"Adapter '{job_id}' set as active for training (resuming run).")
                loaded_or_added_peft_model = self.base_model 
                
                if os.path.isfile(optimizer_path):
                    logger.info(f"Loading optimizer state from: {optimizer_path}")
                    optimizer_state_dict = torch.load(optimizer_path, map_location='cpu')
                    logger.info(f"Optimizer state loaded successfully for job '{job_id}'.")
                else:
                    logger.info(f"Optimizer state file not found at {optimizer_path}. Starting optimizer from scratch.")

            if loaded_or_added_peft_model:
                 loaded_or_added_peft_model.train()
                 self.active_training_adapter_job_id = job_id
            
            return loaded_or_added_peft_model, optimizer_state_dict

        except Exception as e:
            logger.error(f"Error loading/adding training state for job '{job_id}' from base path {normalized_base_path}: {e}", exc_info=True)
            if self.active_training_adapter_job_id == job_id:
                self.active_training_adapter_job_id = None
            return None, None

    def save_training_state(self, job_id: str, model_to_save: PeftModel, optimizer_state_dict: Dict[str, Any], adapter_save_path: str):
        normalized_base_path = os.path.normpath(adapter_save_path)
        logger.info(f"Attempting to save training state for job '{job_id}' to base path: {normalized_base_path}")
        
        actual_adapter_dir = os.path.join(normalized_base_path, job_id)
        
        if not os.path.exists(normalized_base_path):
            try:
                os.makedirs(normalized_base_path)
                logger.info(f"Created base directory for saving state: {normalized_base_path}")
            except OSError as e:
                logger.error(f"Error creating base directory {normalized_base_path} for saving state: {e}")
                return
        
        if not os.path.exists(actual_adapter_dir):
             try:
                 os.makedirs(actual_adapter_dir)
                 logger.info(f"Created actual adapter directory for saving state: {actual_adapter_dir}")
             except OSError as e:
                 logger.error(f"Error creating actual adapter directory {actual_adapter_dir} for saving state: {e}")
                 return

        try:
            config_path = os.path.join(actual_adapter_dir, CONFIG_NAME)
            if job_id in model_to_save.peft_config:
                 adapter_config = model_to_save.peft_config[job_id]
                 logger.info(f"Manually saving adapter config for '{job_id}' to '{config_path}'")
                 try:
                     config_dict = adapter_config.to_dict()
                     for key, value in config_dict.items():
                         if isinstance(value, set):
                             logger.debug(f"Converting set to list for key '{key}' in config before saving.")
                             config_dict[key] = list(value)
                     with open(config_path, 'w', encoding='utf-8') as f:
                          json.dump(config_dict, f, indent=2)
                     logger.info(f"Manual config save successful for '{job_id}'.")
                 except Exception as config_save_e:
                      logger.error(f"Failed to manually save config for '{job_id}' to '{config_path}': {config_save_e}", exc_info=True)
            else:
                 logger.warning(f"Cannot manually save config for '{job_id}', adapter not found in model's peft_config.")
            
            logger.debug(f"[Debug] Saving adapter weights for '{job_id}' via save_pretrained to base path '{normalized_base_path}'...")
            model_to_save.save_pretrained(normalized_base_path, selected_adapters=[job_id])
            logger.info(f"Adapter weights save call completed for job '{job_id}' to base path {normalized_base_path}.")
            
            optimizer_path = os.path.join(normalized_base_path, OPTIMIZER_NAME)
            logger.debug(f"[Debug] Saving optimizer state to '{optimizer_path}'...")
            torch.save(optimizer_state_dict, optimizer_path)
            logger.info(f"Optimizer state for job '{job_id}' saved successfully to {optimizer_path}.")
            
        except Exception as e:
            logger.error(f"Error saving training state for job '{job_id}' to base path {normalized_base_path}: {e}", exc_info=True)

    def unload_training_adapter(self, job_id: str):
         """Disables the specified training adapter, reverting to base model state."""
         logger.info(f"Disabling training adapter '{job_id}'...")
         if self.active_training_adapter_job_id == job_id:
              try:
                  if hasattr(self.base_model, 'disable_adapter'):
                       logger.debug(f"Attempting to disable adapter '{job_id}' using disable_adapter()...")
                       self.base_model.disable_adapter()
                       logger.info(f"Adapter '{job_id}' disabled via disable_adapter().")
                  else:
                       logger.debug(f"disable_adapter() not found. Attempting to disable adapter '{job_id}' using set_adapter(None)...")
                       self.base_model.set_adapter(None)
                       logger.info(f"Adapter '{job_id}' potentially disabled via set_adapter(None). Check for PEFT warnings/errors.")
              except Exception as e:
                  logger.warning(f"Could not disable/set adapter '{job_id}' to None/default: {e}", exc_info=True)
              finally:
                  self.active_training_adapter_job_id = None
                  logger.debug(f"Marked job '{job_id}' as inactive in adapter manager.")
         else:
              logger.debug(f"Adapter '{job_id}' was not the active training adapter ('{self.active_training_adapter_job_id}'). No disable action taken.") 