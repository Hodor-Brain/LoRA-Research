# Manager for Loading/Unloading/Caching LoRA Adapters

import logging
import os
import torch
import json
from peft import PeftModel, PeftConfig, LoraConfig
from transformers import PreTrainedModel
from typing import Optional, Tuple, Dict, Any, List, Set
from collections import OrderedDict

logger = logging.getLogger(__name__)

ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
OPTIMIZER_NAME = "optimizer.pt"

class LoRAAdapterManager:
    def __init__(self, base_model: PreTrainedModel, cache_size: int = 3):
        self.base_model = base_model
        self.active_training_adapter_job_id: Optional[str] = None
        logger.info(f"LoRAAdapterManager initialized.")

    def load_inference_adapter(self, adapter_path: str) -> Optional[PeftModel]:
        """Loads or activates a LoRA adapter for inference.
        If switching from a training adapter, disables it but DOES NOT DELETE it.
        Assumes adapter files (config, weights) are in adapter_path/job_id/.
        Args:
            adapter_path (str): The base directory path (e.g., ./adapters/job_id).
        Returns:
            Optional[PeftModel]: The base model with the adapter loaded/activated, or None on failure.
        """
        # --- Handle switching away from active training adapter --- 
        previous_training_id = self.active_training_adapter_job_id
        if previous_training_id is not None:
            logger.warning(f"Active training adapter '{previous_training_id}' found when switching to inference/base model. Disabling it (but not deleting).)")
            try:
                if hasattr(self.base_model, 'disable_adapter'):
                    self.base_model.disable_adapter()
                    logger.debug(f"Disabled adapter '{previous_training_id}'.")
                else:
                    logger.warning("Model has no disable_adapter method.")
                self.active_training_adapter_job_id = None # Mark as inactive
            except Exception as e:
                logger.error(f"Error disabling previous training adapter '{previous_training_id}': {e}", exc_info=True)
                # Proceed? Or return None? Let's proceed but log the error.
        # --------------------------------------------------------

        if not adapter_path:
            logger.debug("No adapter path provided for inference, using base model.")
            # No need to delete here, already disabled above if one was active.
            self.base_model.eval() # Ensure model is in eval mode
            return self.base_model

        normalized_base_path = os.path.normpath(adapter_path)
        job_id = os.path.basename(normalized_base_path)
        actual_adapter_dir = os.path.join(normalized_base_path, job_id)
        adapter_name = job_id

        logger.debug(f"Request to load inference adapter '{adapter_name}' from '{actual_adapter_dir}'")

        try:
            # Previous training adapter was handled above (disabled, not deleted)
            adapter_already_in_model = adapter_name in getattr(self.base_model, 'peft_config', {})

            if adapter_already_in_model:
                logger.info(f"Inference adapter '{adapter_name}' already loaded in model. Setting active.")
                self.base_model.set_adapter(adapter_name)
            else:
                logger.info(f"Inference adapter '{adapter_name}' not found in model. Loading from '{actual_adapter_dir}'...")
                
                if not os.path.isdir(actual_adapter_dir):
                    logger.error(f"Actual adapter directory not found: {actual_adapter_dir}")
                    return None
                config_path = os.path.join(actual_adapter_dir, CONFIG_NAME)
                if not os.path.isfile(config_path):
                    logger.error(f"Adapter config file not found: {config_path}")
                    return None
                weights_path_bin = os.path.join(actual_adapter_dir, ADAPTER_WEIGHTS_NAME)
                weights_path_safe = os.path.join(actual_adapter_dir, ADAPTER_SAFE_WEIGHTS_NAME)
                has_weights = os.path.isfile(weights_path_bin) or os.path.isfile(weights_path_safe)
                if not has_weights:
                    logger.error(f"Adapter weights file not found in {actual_adapter_dir}.")
                    return None

                self.base_model.load_adapter(actual_adapter_dir, adapter_name=adapter_name, is_local=True)
                logger.info(f"Successfully loaded adapter '{adapter_name}' from {actual_adapter_dir}.")
                
                self.base_model.set_adapter(adapter_name)
            
            self.base_model.eval() 
            # self.active_training_adapter_job_id = None # Already set above
            logger.info(f"Adapter '{adapter_name}' active for inference.")
            return self.base_model 

        except Exception as e:
            logger.error(f"Failed during load/activation of inference adapter '{adapter_name}': {e}", exc_info=True)
            return None
            
    def _delete_adapter_safely(self, adapter_id: str) -> bool:
         """Internal helper to delete an adapter with logging and checks. Returns True on success/not found, False on error."""
         if not adapter_id:
              logger.debug("Skipping deletion: adapter_id is None or empty.")
              return True
              
         logger.debug(f"Attempting deletion of adapter: '{adapter_id}'")
         try:
             peft_config = getattr(self.base_model, 'peft_config', None)
             if hasattr(self.base_model, 'delete_adapter') and peft_config is not None and adapter_id in peft_config:
                 self.base_model.delete_adapter(adapter_id)
                 logger.info(f"Successfully deleted adapter '{adapter_id}'.")
                 return True
             elif not hasattr(self.base_model, 'delete_adapter'):
                 logger.warning(f"Cannot delete adapter '{adapter_id}': model has no 'delete_adapter' method.")
                 return False
             elif peft_config is None:
                 logger.warning(f"Cannot delete adapter '{adapter_id}': model has no 'peft_config' attribute.")
                 return False
             else:
                 logger.info(f"Adapter '{adapter_id}' not found in model's peft_config for deletion (already deleted?).")
                 return True
         except Exception as e:
             logger.error(f"Failed during deletion of adapter '{adapter_id}': {e}", exc_info=True)
         return False

    def load_training_state(self, 
                            job_id: str, 
                            adapter_path: str, 
                            lora_config_dict: Dict[str, Any]
                           ) -> Tuple[Optional[PeftModel], Optional[Dict[str, Any]]]:
        """Loads or initializes adapter weights and optimizer state for a training job.
        Ensures only the target training adapter is active and loaded by deleting previous one AND any others.
        Assumes adapter files are stored in adapter_path/job_id/
        Checks if adapter_path/job_id/adapter_config.json exists.
        """
        logger.info(f"Attempting to load/init training state for job '{job_id}' from base path: {adapter_path}")
        optimizer_state_dict = None
        loaded_or_added_peft_model = None

        normalized_base_path = os.path.normpath(adapter_path)
        actual_adapter_dir = os.path.join(normalized_base_path, job_id)
        config_path = os.path.join(actual_adapter_dir, CONFIG_NAME)
        optimizer_path = os.path.join(normalized_base_path, OPTIMIZER_NAME)

        previous_training_id = self.active_training_adapter_job_id
        if previous_training_id is not None and previous_training_id != job_id:
            logger.info(f"Switching training adapter. Deleting previous training adapter: '{previous_training_id}'")
            if not self._delete_adapter_safely(previous_training_id):
                 logger.error(f"Failed to safely delete previous training adapter '{previous_training_id}'. Aborting load.")
                 return None, None 
            self.active_training_adapter_job_id = None 

        config_exists = os.path.isfile(config_path)
        is_first_run = not config_exists

        try:
            adapter_already_in_model = job_id in getattr(self.base_model, 'peft_config', {})

            if is_first_run:
                logger.info(f"Config file not found at '{config_path}'. Assuming first run for job '{job_id}'. Adding adapter.")
                if adapter_already_in_model:
                     logger.warning(f"Adapter '{job_id}' unexpectedly exists in model. Removing before add.")
                     if not self._delete_adapter_safely(job_id):
                          logger.error(f"Failed to remove existing adapter '{job_id}' before first run add. Aborting.")
                          return None, None
                
                lora_config_dict.setdefault("task_type", "CAUSAL_LM")
                peft_config = LoraConfig(**lora_config_dict)
                self.base_model.add_adapter(job_id, peft_config)
                logger.info(f"Added adapter configuration '{job_id}'.")
                self.base_model.set_adapter(job_id)
                
                logger.info(f"Adapter '{job_id}' set active (first run).")
                loaded_or_added_peft_model = self.base_model
                optimizer_state_dict = None

            else:
                logger.info(f"Config file found at '{config_path}'. Loading weights for '{job_id}' from '{actual_adapter_dir}'...")
                if not adapter_already_in_model:
                    logger.warning(f"Adapter '{job_id}' config found, but not in model. Loading from path...")
                    self.base_model.load_adapter(actual_adapter_dir, adapter_name=job_id, is_local=True)
                    logger.info(f"Adapter '{job_id}' loaded from {actual_adapter_dir} (implicitly added)." )
                else:
                     self.base_model.load_adapter(actual_adapter_dir, adapter_name=job_id, is_local=True)
                     logger.info(f"Adapter weights for '{job_id}' loaded into existing adapter from {actual_adapter_dir}.")
                
                self.base_model.set_adapter(job_id)
                logger.info(f"Adapter '{job_id}' set active (resuming run).")
                loaded_or_added_peft_model = self.base_model 
                
                if os.path.isfile(optimizer_path):
                    logger.info(f"Loading optimizer state from: {optimizer_path}")
                    optimizer_state_dict = torch.load(optimizer_path, map_location='cpu')
                    logger.info(f"Optimizer state loaded for job '{job_id}'.")
                else:
                    logger.warning(f"Optimizer state file not found at {optimizer_path}. Starting optimizer fresh.")

            peft_config = getattr(self.base_model, 'peft_config', {})
            adapters_to_delete = [name for name in peft_config if name != job_id]
            if adapters_to_delete:
                 logger.info(f"Cleaning up other adapters before training '{job_id}': {adapters_to_delete}")
                 for adapter_name_to_delete in adapters_to_delete:
                      self._delete_adapter_safely(adapter_name_to_delete)
                 peft_config_after_cleanup = getattr(self.base_model, 'peft_config', {})
                 if len(peft_config_after_cleanup) > 1:
                      logger.error(f"Cleanup failed! Adapters remaining after attempted deletion: {list(peft_config_after_cleanup.keys())}")
                      self.active_training_adapter_job_id = None
                      return None, None
                 logger.info(f"Model adapters cleaned. Only '{job_id}' should remain.")
            
            if loaded_or_added_peft_model:
                 active_adapters = getattr(loaded_or_added_peft_model, 'active_adapters', [])
                 if active_adapters != [job_id]:
                      logger.warning(f"Model active adapters {active_adapters} != '{job_id}'. Forcing set.")
                      loaded_or_added_peft_model.set_adapter(job_id)
                 
                 loaded_or_added_peft_model.train()
                 self.active_training_adapter_job_id = job_id
                 logger.info(f"Adapter '{job_id}' confirmed active; model in train mode.")
            else:
                 logger.error(f"Model object is None after load/add attempt for '{job_id}'.")
                 self.active_training_adapter_job_id = None
                 return None, None
            
            return loaded_or_added_peft_model, optimizer_state_dict

        except Exception as e:
            logger.error(f"Error in load_training_state for '{job_id}': {e}", exc_info=True)
            if self.active_training_adapter_job_id == job_id: 
                self.active_training_adapter_job_id = None
            return None, None

    def save_training_state(self, job_id: str, model_to_save: PeftModel, optimizer_state_dict: Dict[str, Any], adapter_save_path: str):
        """Saves adapter weights/config and optimizer state for a training job.
        Uses PEFT save_pretrained for adapter config/weights (creates job_id/job_id subdir).
        Saves optimizer state separately in the base directory (job_id/).
        """
        normalized_base_path = os.path.normpath(adapter_save_path)
        logger.info(f"Attempting to save training state for job '{job_id}' to base path: {normalized_base_path}")
        
        if not os.path.exists(normalized_base_path):
            try:
                os.makedirs(normalized_base_path)
                logger.info(f"Created base directory for saving state: {normalized_base_path}")
            except OSError as e:
                logger.error(f"Error creating base directory {normalized_base_path}: {e}")
                return
        
        # NOTE: We do NOT manually create the nested job_id dir here.
        # PEFT's save_pretrained will handle creating {normalized_base_path}/{job_id}/

        try:
            logger.info(f"Saving adapter weights/config for '{job_id}' via save_pretrained to base dir '{normalized_base_path}'...")
            model_to_save.save_pretrained(
                 normalized_base_path, 
                 selected_adapters=[job_id],
                 safe_serialization=True
            )
            logger.info(f"PEFT save_pretrained completed for job '{job_id}' to base path {normalized_base_path}.")
            
            optimizer_path = os.path.join(normalized_base_path, OPTIMIZER_NAME)
            logger.info(f"Saving optimizer state for '{job_id}' to '{optimizer_path}'...")
            torch.save(optimizer_state_dict, optimizer_path)
            logger.info(f"Optimizer state for job '{job_id}' saved successfully to {optimizer_path}.")
            
        except Exception as e:
            logger.error(f"Error saving training state for job '{job_id}' to base path {normalized_base_path}: {e}", exc_info=True)

    def unload_training_adapter(self, job_id: str):
         """Disables AND Deletes the specified training adapter.
         Needed to prevent interference between concurrent jobs.
         """
         logger.info(f"Unloading training adapter '{job_id}' (disable and delete)...")
         is_active = self.active_training_adapter_job_id == job_id
         
         try:
             if is_active and hasattr(self.base_model, 'disable_adapter'):
                 logger.debug(f"Attempting disable_adapter() for active adapter '{job_id}'.")
                 self.base_model.disable_adapter()
                 logger.info(f"Adapter '{job_id}' disabled via disable_adapter().")
             
             deleted = self._delete_adapter_safely(job_id)
             if not deleted:
                  logger.warning(f"Adapter '{job_id}' may not have been fully deleted.")
                       
         except Exception as e:
             logger.warning(f"Error during disable/delete for adapter '{job_id}': {e}", exc_info=True)
         finally:
             if is_active:
                 self.active_training_adapter_job_id = None
                 logger.debug(f"Marked job '{job_id}' as inactive in adapter manager state.")
             else:
                 logger.debug(f"Unload called for '{job_id}', but '{self.active_training_adapter_job_id}' was active. Active state unchanged.") 

    def get_active_adapters(self) -> Set[str]:
        """Returns the set of adapter names currently loaded in the base model."""
        try:
            peft_config = getattr(self.base_model, 'peft_config', {})
            return set(peft_config.keys())
        except Exception as e:
            logger.error(f"Error getting active adapters: {e}", exc_info=True)
            return set() 