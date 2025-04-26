# Main Controller / Scheduler Logic

import logging
import time
import torch
import os
from typing import Dict, Optional, List
from dataclasses import asdict

from utils.config_loader import SystemConfig
from core.engine import InferenceEngine
from managers.job_manager import ActiveTrainingJobManager, JobStatus, TrainingJobState
from managers.adapter_manager import LoRAAdapterManager, OPTIMIZER_NAME
from managers.queue_manager import QueueManager
from prioritization.base_strategy import BasePrioritizationStrategy
from core.training import setup_lora_training, execute_training_step, SimpleTextDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        logger.info("CUDA is available and bfloat16 is supported. Using bfloat16.")
    else:
        model_dtype = torch.float16
        logger.info("CUDA is available but bfloat16 not supported. Using float16.")
else:
    model_dtype = torch.float32
    logger.info("CUDA not available. Using float32 on CPU.")

class MainController:
    def __init__(self, 
                 config: SystemConfig, 
                 engine: InferenceEngine, 
                 job_manager: ActiveTrainingJobManager, 
                 adapter_manager: LoRAAdapterManager, 
                 queue_manager: QueueManager, 
                 prioritization_strategy: BasePrioritizationStrategy,
                 device: str = 'cuda'):
        self.config = config
        self.engine = engine
        self.job_manager = job_manager
        self.adapter_manager = adapter_manager
        self.queue_manager = queue_manager
        self.prioritization_strategy = prioritization_strategy
        self.device = device
        self.model_dtype = model_dtype
        self.running = False
        self.active_training_jobs: Dict[str, Dict] = {}
        logger.info("MainController initialized.")

    def _check_for_new_jobs(self):
        """Checks queue for new training jobs and registers them."""
        new_job_details = self.queue_manager.get_new_training_job()
        if new_job_details:
            job_id = new_job_details.get('job_id')
            if not job_id:
                 logger.error("Received job details without job_id. Skipping.")
                 return
            
            if not new_job_details.get('adapter_save_path'):
                 default_save_path = os.path.join(".", "adapters", f"job_{job_id}")
                 logger.warning(f"adapter_save_path not provided for job {job_id}, defaulting to {default_save_path}")
                 new_job_details['adapter_save_path'] = default_save_path
            
            logger.info(f"Received new training job request: {job_id}")
            dataset_samples = new_job_details.get('dataset_samples')
            
            registered_id = self.job_manager.register_job(new_job_details)
            if registered_id:
                logger.info(f"Successfully registered job {registered_id}")
                self._prepare_training_job_runtime(registered_id, dataset_samples)
            else:
                 logger.error(f"Failed to register job {job_id}")

    def _prepare_training_job_runtime(self, job_id: str, dataset_samples: Optional[List[str]] = None):
        """Sets up optimizer and potentially dataloader for a newly registered job."""
        job_state: Optional[TrainingJobState] = self.job_manager.get_job_state(job_id)

        if not job_state:
            logger.error(f"Cannot prepare runtime for job {job_id}: Job state not found.")
            return
        if job_id in self.active_training_jobs:
            logger.warning(f"Job {job_id} runtime already prepared, skipping setup.")
            return

        logger.info(f"Preparing runtime for job {job_id}...")
        optimizer = None
        logger.debug(f"Optimizer placeholder set for job {job_id}.")
        
        train_dataset = None
        try:
            if not dataset_samples or not isinstance(dataset_samples, list):
                logger.error(f"Job {job_id} submitted without valid 'dataset_samples' (received: {type(dataset_samples)}). Failing job.")
                self.job_manager.fail_job(job_id, "Missing or invalid dataset_samples")
                return
            
            logger.debug(f"Using provided dataset samples for job {job_id}. Count: {len(dataset_samples)}")
            train_dataset = SimpleTextDataset(dataset_samples, self.engine.tokenizer)

            batch_size = job_state.training_params.get('batch_size', self.config.controller.training_batch_size or 1)
            logger.debug(f"Creating DataLoader for job {job_id} with batch size {batch_size}...")
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            logger.info(f"DataLoader created for job {job_id}.")

            logger.debug(f"Creating data iterator for job {job_id}...")
            data_iterator = iter(train_dataloader)
            logger.debug(f"Data iterator created for job {job_id}.")

            logger.debug(f"Assigning runtime info to active_training_jobs for job {job_id}...")
            self.active_training_jobs[job_id] = {
                'optimizer': optimizer,
                'dataloader': train_dataloader,
                'data_iterator': data_iterator
            }
            logger.info(f"Runtime successfully prepared and stored for job {job_id}.")

        except Exception as e:
            logger.error(f"Error preparing runtime for job {job_id} (Dataset: {train_dataset is not None}): {e}", exc_info=True)
            self.job_manager.fail_job(job_id, "Runtime preparation failed during Dataset/DataLoader creation")
            self.active_training_jobs.pop(job_id, None)
        
    def _decide_mode(self) -> str:
        """Decides whether to run Inference or Training next."""
        if self.queue_manager.get_inference_queue_size() > 0:
            logger.debug("Decision: Inference mode (queue not empty)")
            return "inference"
        
        runnable_jobs = self.job_manager.get_active_runnable_job_states()
        if runnable_jobs:
             runnable_job_ids = list(runnable_jobs.keys())
             prepared_runnable_jobs = [jid for jid in runnable_job_ids if jid in self.active_training_jobs]
             if prepared_runnable_jobs:
                 logger.debug("Decision: Training mode (runnable and prepared jobs exist)")
                 return "training"
             else:
                 logger.debug("Decision: Idle mode (runnable jobs exist but none have runtime prepared yet)")
                 return "idle"

        logger.debug("Decision: Idle mode (no pending requests or runnable jobs)")
        return "idle"

    def _execute_inference_batch(self):
        """Gets requests from queue and runs inference."""
        batch_size = self.config.controller.inference_batch_size
        requests = self.queue_manager.get_inference_batch(batch_size)
        if not requests:
            logger.debug("Inference batch execution skipped: Queue empty.")
            return

        logger.info(f"Executing inference batch of size {len(requests)}.")
        results = self.engine.process_batch(requests)
        logger.info(f"Inference batch execution finished. Results count: {len(results) if results else 0}")

    def _execute_training_batch(self):
        """Selects a job, loads its state, runs a step, saves state."""
        runnable_jobs = self.job_manager.get_active_runnable_job_states()
        prepared_runnable_jobs = {jid: state for jid, state in runnable_jobs.items() if jid in self.active_training_jobs}

        if not prepared_runnable_jobs:
             logger.debug("Training batch execution skipped: No runnable jobs with prepared runtime.")
             return
        
        selected_job_id = self.prioritization_strategy.select_next_job(prepared_runnable_jobs)
        if not selected_job_id:
            logger.debug("Training batch execution skipped: Prioritization returned no job from prepared set.")
            return
        
        logger.info(f"Selected training job {selected_job_id} for next step.")
        job_state = self.job_manager.get_job_state(selected_job_id)
        
        if selected_job_id not in self.active_training_jobs:
             logger.error(f"Job {selected_job_id} selected but has no runtime info in active_training_jobs. Skipping.")
             return
        if not job_state:
             logger.error(f"Job {selected_job_id} selected but has no state in job_manager. Skipping.")
             self.active_training_jobs.pop(selected_job_id, None)
             return

        adapter_path = job_state.adapter_save_path
        if not adapter_path:
            logger.error(f"adapter_save_path not defined for job {selected_job_id}. Skipping step.")
            self.job_manager.fail_job(selected_job_id, "adapter_save_path missing")
            self.active_training_jobs.pop(selected_job_id, None)
            return
        optimizer_path = os.path.join(adapter_path, OPTIMIZER_NAME)
        lora_config_dict = asdict(self.config.training.lora_config)

        logger.info(f"Loading/Init training state for job '{selected_job_id}' from {adapter_path}")
        loaded_adapter_model, loaded_optimizer_state_dict = self.adapter_manager.load_training_state(
            job_id=selected_job_id, 
            adapter_path=adapter_path,
            lora_config_dict=lora_config_dict
        )
        if loaded_adapter_model is None:
             logger.error(f"Adapter loading/init failed for job {selected_job_id}. Failing job.")
             self.job_manager.fail_job(selected_job_id, "Adapter loading/init failed")
             self.active_training_jobs.pop(selected_job_id, None) 
             return
        
        current_peft_model = loaded_adapter_model
        
        job_runtime = self.active_training_jobs[selected_job_id]
        optimizer = job_runtime.get('optimizer')
        if optimizer is None:
            logger.info(f"Creating optimizer for job {selected_job_id}...")
            try:
                 trainable_params = list(current_peft_model.parameters())
                 if not any(p.requires_grad for p in trainable_params):
                      logger.error(f"Model for job {selected_job_id} has no trainable parameters! Check LoRA setup.")
                      raise ValueError("No trainable parameters found.")
                 
                 optimizer = AdamW(current_peft_model.parameters(), lr=job_state.training_params.get('lr', 5e-5))
                 job_runtime['optimizer'] = optimizer
                 logger.info(f"Optimizer created for job {selected_job_id}.")
                 if loaded_optimizer_state_dict:
                      logger.warning(f"Optimizer state dict found on first step for job {selected_job_id}, but shouldn't exist yet. Ignoring.") 
            except Exception as e:
                logger.error(f"Failed to create optimizer for job {selected_job_id}: {e}", exc_info=True)
                self.job_manager.fail_job(selected_job_id, "Optimizer setup failed")
                self.active_training_jobs.pop(selected_job_id, None)
                return
        else:
            logger.debug(f"Using existing optimizer placeholder for job {selected_job_id}. Will load state.")
            if loaded_optimizer_state_dict:
                logger.info(f"Loading optimizer state for job {selected_job_id} from {optimizer_path}.")
                try:
                    logger.info(f"Re-creating optimizer for job {selected_job_id} before loading state...")
                    optimizer = AdamW(current_peft_model.parameters(), lr=job_state.training_params.get('lr', 5e-5))
                    job_runtime['optimizer'] = optimizer
                    logger.info(f"Optimizer re-created for job {selected_job_id}.")

                    logger.debug(f"Pre-casting loaded optimizer state dict tensors to device '{self.device}' and dtype '{self.model_dtype}'...")
                    casted_state_values = {}
                    original_state = loaded_optimizer_state_dict.get('state', {})
                    for param_id, state_tensors in original_state.items():
                        casted_tensors = {}
                        for k, v in state_tensors.items():
                             if isinstance(v, torch.Tensor):
                                 if k != 'step':
                                      casted_tensors[k] = v.to(device=self.device, dtype=self.model_dtype)
                                      # logger.debug(f"  Tensor {k} for param {param_id} CAST to {casted_tensors[k].device}, {casted_tensors[k].dtype}")
                                 else:
                                      casted_tensors[k] = v
                             else:
                                 casted_tensors[k] = v
                        casted_state_values[param_id] = casted_tensors
                    
                    pre_casted_optimizer_state_dict = {
                         'state': casted_state_values,
                         'param_groups': loaded_optimizer_state_dict.get('param_groups', [])
                    }
                    logger.debug("Pre-casting complete.")
                    
                    optimizer.load_state_dict(pre_casted_optimizer_state_dict)
                    logger.info(f"Optimizer state loaded from PRE-CAST dict for job {selected_job_id}.")

                    
                except Exception as state_load_e:
                    logger.error(f"Failed to load or move/cast optimizer state tensors for job {selected_job_id}: {state_load_e}", exc_info=True)
                    self.job_manager.fail_job(selected_job_id, "Optimizer state loading/casting failed")
                    self.active_training_jobs.pop(selected_job_id, None)
                    self.adapter_manager.unload_training_adapter(selected_job_id) 
                    return
            else:
                logger.error(f"Optimizer exists for job {selected_job_id} but no state dict found to load on step > 1. This is unexpected. Failing job.")
                self.job_manager.fail_job(selected_job_id, "Missing optimizer state on subsequent step")
                self.active_training_jobs.pop(selected_job_id, None)
                self.adapter_manager.unload_training_adapter(selected_job_id) 
                return

        if optimizer is None:
             logger.error(f"Optimizer is None for job {selected_job_id} before training step. Failing job.")
             self.job_manager.fail_job(selected_job_id, "Optimizer became None unexpectedly")
             self.active_training_jobs.pop(selected_job_id, None)
             self.adapter_manager.unload_training_adapter(selected_job_id) 
             return
             
        data_iterator = job_runtime.get('data_iterator')
        dataloader = job_runtime.get('dataloader')
        if not data_iterator or not dataloader:
             logger.error(f"Missing dataloader/iterator for job {selected_job_id}. Skipping step.")
             self.job_manager.fail_job(selected_job_id, "Data loading error")
             self.active_training_jobs.pop(selected_job_id, None)
             return
        
        try:
            batch = next(data_iterator)
            batch = {k: v.to(device=self.device, dtype=self.model_dtype if torch.is_floating_point(v) else v.dtype) 
                     for k, v in batch.items()}
        except StopIteration:
            logger.info(f"DataLoader epoch finished for job {selected_job_id}. Resetting iterator.")
            job_state.current_epoch += 1
            job_runtime['data_iterator'] = iter(dataloader)
            try:
                 batch = next(job_runtime['data_iterator'])
                 batch = {k: v.to(device=self.device, dtype=self.model_dtype if torch.is_floating_point(v) else v.dtype) 
                          for k, v in batch.items()}
            except StopIteration:
                 logger.error(f"DataLoader for job {selected_job_id} is empty even after reset. Failing job.")
                 self.job_manager.fail_job(selected_job_id, "Empty dataset")
                 self.active_training_jobs.pop(selected_job_id, None)
                 return
        except Exception as e:
             logger.error(f"Error getting/processing batch for job {selected_job_id}: {e}", exc_info=True)
             self.job_manager.fail_job(selected_job_id, "Data loading error")
             self.active_training_jobs.pop(selected_job_id, None)
             return

        loss = execute_training_step(current_peft_model, optimizer, batch, self.device)

        if loss is not None:
            self.job_manager.update_job_state(selected_job_id, step_increment=1, loss=loss)
            logger.info(f"Training Step {job_state.current_step} completed for job {selected_job_id}. Loss: {loss:.4f}")
            
            max_steps = job_state.max_steps 
            current_step_val = job_state.current_step
            
            cond1 = max_steps is not None
            cond2 = current_step_val >= max_steps if cond1 else False
            # # --- Start Re-enabled Logs ---
            # logger.debug(f"[Completion Check Breakdown] Job: {selected_job_id}, Current Step: {current_step_val}, Max Steps: {max_steps}")
            # logger.debug(f"[Completion Check Breakdown] Part 1 (max_steps is not None): {cond1}")
            # logger.debug(f"[Completion Check Breakdown] Part 2 (current_step_val >= max_steps): {cond2}")
            # # --- End Re-enabled Logs ---
            
            if cond1 and cond2:
                logger.info(f"Job {selected_job_id} reached max steps ({max_steps}). Marking as completed.")
                self.job_manager.complete_job(selected_job_id)
                self.adapter_manager.save_training_state(
                    job_id=selected_job_id, 
                    model_to_save=current_peft_model, 
                    optimizer_state_dict=optimizer.state_dict(), 
                    adapter_save_path=adapter_path
                )
                time.sleep(0.1)
                self.active_training_jobs.pop(selected_job_id, None)
                self.adapter_manager.unload_training_adapter(selected_job_id)
                return
            else:
                self.adapter_manager.save_training_state(
                    job_id=selected_job_id,
                    model_to_save=current_peft_model,
                    optimizer_state_dict=optimizer.state_dict(),
                    adapter_save_path=adapter_path
                )
                time.sleep(0.1)
        else:
            logger.error(f"Training step failed for job {selected_job_id} (loss is None). Failing job.")
            self.job_manager.fail_job(selected_job_id, "Training step execution failed")
            self.active_training_jobs.pop(selected_job_id, None)
            self.adapter_manager.unload_training_adapter(selected_job_id)

    def run_loop(self):
        """Runs the main control loop, alternating between modes."""
        self.running = True
        logger.info("Starting Main Controller loop...")
        while self.running:
            try:
                self._check_for_new_jobs()

                mode = self._decide_mode()
                if mode == "inference":
                    self._execute_inference_batch()
                elif mode == "training":
                    self._execute_training_batch()
                elif mode == "idle":
                    time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Stopping controller loop...")
                self.running = False
            except Exception as e:
                 logger.error(f"Unexpected error in controller loop: {e}", exc_info=True)
                 time.sleep(1)

        logger.info("Main Controller loop stopped.")

    def stop_loop(self):
        """Signals the controller loop to stop."""
        logger.info("Requesting controller loop stop.")
        self.running = False 