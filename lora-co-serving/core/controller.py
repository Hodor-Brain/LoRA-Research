# Main Controller / Scheduler Logic

import logging
import time
import torch
import os
from typing import Dict, Optional

from utils.config_loader import SystemConfig
from core.engine import InferenceEngine
from managers.job_manager import ActiveTrainingJobManager, JobStatus
from managers.adapter_manager import LoRAAdapterManager, OPTIMIZER_NAME
from managers.queue_manager import QueueManager
from prioritization.base_strategy import BasePrioritizationStrategy
from core.training import setup_lora_training, execute_training_step, SimpleTextDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

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
        self.running = False
        self.active_training_jobs: Dict[str, Dict] = {} # Store optimizer, dataloader etc. per job
        logger.info("MainController initialized.")

    def _check_for_new_jobs(self):
        """Checks queue for new training jobs and registers them."""
        new_job_details = self.queue_manager.get_new_training_job()
        if new_job_details:
            job_id = new_job_details.get('job_id')
            if not new_job_details.get('adapter_save_path'):
                 default_save_path = os.path.join("./adapters", f"job_{job_id}")
                 logger.warning(f"adapter_save_path not provided for job {job_id}, defaulting to {default_save_path}")
                 new_job_details['adapter_save_path'] = default_save_path
            
            logger.info(f"Received new training job request: {job_id}")
            registered_id = self.job_manager.register_job(new_job_details)
            if registered_id:
                # TODO: Prepare data loader, optimizer etc. for this job
                self._prepare_training_job_runtime(registered_id)
                logger.info(f"Successfully registered and prepared runtime for job {registered_id}")
            else:
                 logger.error(f"Failed to register job {job_id}")

    def _prepare_training_job_runtime(self, job_id: str):
        """Sets up optimizer and potentially dataloader for a newly registered job."""
        job_state = self.job_manager.get_job_state(job_id)
        if not job_state or job_id in self.active_training_jobs:
            logger.warning(f"Job {job_id} not found or already prepared, skipping runtime setup.")
            return

        logger.info(f"Preparing runtime for job {job_id}...")
        # --- Setup LoRA Model for this specific job ---
        # NOTE: LoRA layers are typically added ONCE using get_peft_model.

        optimizer = None
        logger.info(f"Optimizer placeholder set for job {job_id}.")

        # TODO: Replace with actual data loading based on job_state.dataset_ref
        dummy_texts = [f"Data for job {job_id}, sample {i}." for i in range(8)]
        train_dataset = SimpleTextDataset(dummy_texts, self.engine.tokenizer)
        batch_size = job_state.training_params.get('batch_size', self.config.controller.training_batch_size or 1)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"DataLoader created for job {job_id} (Batch Size: {batch_size}).")
        
        self.active_training_jobs[job_id] = {
            'optimizer': optimizer,
            'dataloader': train_dataloader,
            'data_iterator': iter(train_dataloader)
        }
        
    def _decide_mode(self) -> str:
        """Decides whether to run Inference or Training next."""
        if self.queue_manager.get_inference_queue_size() > 0:
            logger.debug("Decision: Inference mode (queue not empty)")
            return "inference"
        
        runnable_jobs = self.job_manager.get_active_runnable_job_states()
        if runnable_jobs:
             logger.debug("Decision: Training mode (runnable jobs exist)")
             return "training"

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
        # TODO: Handle results (e.g., send back via API, log more details)
        logger.info(f"Inference batch execution finished. Results count: {len(results) if results else 0}")

    def _execute_training_batch(self):
        """Selects a job, loads its state, runs a step, saves state."""
        runnable_jobs = self.job_manager.get_active_runnable_job_states()
        if not runnable_jobs:
             logger.debug("Training batch execution skipped: No runnable jobs.")
             return
        
        selected_job_id = self.prioritization_strategy.select_next_job(runnable_jobs)
        if not selected_job_id:
            logger.debug("Training batch execution skipped: Prioritization returned no job.")
            return
        
        logger.info(f"Selected training job {selected_job_id} for next step.")
        job_state = self.job_manager.get_job_state(selected_job_id)
        if not job_state or selected_job_id not in self.active_training_jobs:
             logger.error(f"Selected job {selected_job_id} has no state or runtime info. Skipping.")
             return

        adapter_path = job_state.adapter_save_path
        if not adapter_path:
            logger.error(f"adapter_save_path not defined for job {selected_job_id}. Skipping step.")
            self.job_manager.fail_job(selected_job_id, "adapter_save_path missing")
            return
        optimizer_path = os.path.join(adapter_path, OPTIMIZER_NAME)

        logger.info(f"Loading training state for job '{selected_job_id}' from {adapter_path}")
        loaded_adapter_model, loaded_optimizer_state_dict = self.adapter_manager.load_training_state(
            job_id=selected_job_id, 
            adapter_path=adapter_path 
        )
        if loaded_adapter_model is None:
             self.job_manager.fail_job(selected_job_id, "Adapter loading failed")
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
                      logger.info(f"Loading optimizer state for job {selected_job_id} from {optimizer_path}.")
                      optimizer.load_state_dict(loaded_optimizer_state_dict)
                      logger.info(f"Optimizer state loaded for job {selected_job_id}.")
                      optimizer.load_state_dict(loaded_optimizer_state_dict)
                      logger.info(f"Optimizer state loaded for job {selected_job_id}.")
                 else:
                      logger.info(f"No optimizer state found for job {selected_job_id}. Starting fresh.")
            except Exception as e:
                logger.error(f"Failed to create/load optimizer for job {selected_job_id}: {e}", exc_info=True)
                self.job_manager.fail_job(selected_job_id, "Optimizer setup failed")
                self.active_training_jobs.pop(selected_job_id, None)
                return
        else:
             logger.debug(f"Using existing optimizer for job {selected_job_id}.")

        data_iterator = job_runtime.get('data_iterator')
        dataloader = job_runtime.get('dataloader')
        if not data_iterator or not dataloader:
             logger.error(f"Missing dataloader/iterator for job {selected_job_id}. Skipping step.")
             self.job_manager.fail_job(selected_job_id, "Data loading error")
             self.active_training_jobs.pop(selected_job_id, None)
             return
        
        try:
            batch = next(data_iterator)
        except StopIteration:
            logger.info(f"DataLoader epoch finished for job {selected_job_id}. Resetting iterator.")
            job_state.current_epoch += 1
            job_runtime['data_iterator'] = iter(dataloader)
            try:
                 batch = next(job_runtime['data_iterator'])
            except StopIteration:
                 logger.error(f"DataLoader for job {selected_job_id} is empty even after reset. Failing job.")
                 self.job_manager.fail_job(selected_job_id, "Empty dataset")
                 self.active_training_jobs.pop(selected_job_id, None)
                 return
        except Exception as e:
             logger.error(f"Error getting batch for job {selected_job_id}: {e}", exc_info=True)
             self.job_manager.fail_job(selected_job_id, "Data loading error")
             self.active_training_jobs.pop(selected_job_id, None)
             return

        loss = execute_training_step(current_peft_model, optimizer, batch, self.device)

        if loss is not None:
            self.job_manager.update_job_state(selected_job_id, step_increment=1, loss=loss)
            logger.info(f"Training Step {job_state.current_step} completed for job {selected_job_id}. Loss: {loss:.4f}")
            max_steps = job_state.max_steps
            if max_steps is not None and job_state.current_step >= max_steps:
                logger.info(f"Job {selected_job_id} reached max steps ({max_steps}). Marking as completed.")
                self.job_manager.complete_job(selected_job_id)
                self.adapter_manager.save_training_state(
                    job_id=selected_job_id, 
                    model_to_save=current_peft_model, 
                    optimizer_state_dict=optimizer.state_dict(), 
                    adapter_save_path=adapter_path
                )
                self.active_training_jobs.pop(selected_job_id, None)
                self.adapter_manager.unload_training_adapter(selected_job_id)
            else:
                self.adapter_manager.save_training_state(
                    job_id=selected_job_id,
                    model_to_save=current_peft_model,
                    optimizer_state_dict=optimizer.state_dict(),
                    adapter_save_path=adapter_path
                )

        else:
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