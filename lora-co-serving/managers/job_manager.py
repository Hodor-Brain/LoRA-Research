# Manager for Active Training Job State

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Iterator
from enum import Enum
import time

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class TrainingJobState:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    lora_config: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    dataset_ref: Any = None
    priority_info: Any = None
    adapter_save_path: Optional[str] = None
    optimizer_state_path: Optional[str] = None # Path to save/load optimizer state (TODO)
    current_step: int = 0
    max_steps: Optional[int] = None
    current_epoch: int = 0
    loss_history: List[float] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None

class ActiveTrainingJobManager:
    def __init__(self):
        """Initializes the manager for active training jobs."""
        self.active_jobs: Dict[str, TrainingJobState] = {}
        logger.info("ActiveTrainingJobManager initialized.")

    def register_job(self, job_details: Dict[str, Any]) -> Optional[str]:
        """Registers a new training job based on details from the queue.

        Args:
            job_details (Dict[str, Any]): Dictionary containing job parameters.
                                            Requires at least 'job_id'.

        Returns:
            Optional[str]: The job_id if registration is successful, None otherwise.
        """
        job_id = job_details.get('job_id')
        if not job_id:
            logger.error("Cannot register job: 'job_id' missing from job_details.")
            return None
        
        if job_id in self.active_jobs:
            logger.warning(f"Job {job_id} is already registered. Ignoring duplicate registration.")
            return job_id
        
        try:
            job_state = TrainingJobState(
                job_id=job_id,
                status=JobStatus.ACTIVE,
                lora_config=job_details.get('lora_config', {}),
                training_params=job_details.get('training_params', {}),
                dataset_ref=job_details.get('dataset_ref'),
                priority_info=job_details.get('priority_info'),
                adapter_save_path=job_details.get('adapter_save_path'),
                max_steps=job_details.get('max_steps'),
                start_time=time.time()
            )
            self.active_jobs[job_id] = job_state
            logger.info(f"Registered and activated new training job: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to create TrainingJobState for job {job_id}: {e}", exc_info=True)
            return None

    def update_job_state(self, job_id: str, status: Optional[JobStatus] = None, 
                         step_increment: int = 0, loss: Optional[float] = None, 
                         error_message: Optional[str] = None) -> bool:
        """Updates the state of an active training job.

        Args:
            job_id (str): The ID of the job to update.
            status (Optional[JobStatus]): New status for the job.
            step_increment (int): Number of steps completed in this update (usually 1).
            loss (Optional[float]): Loss value for the completed step(s).
            error_message (Optional[str]): Error message if the job failed.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        if job_id not in self.active_jobs:
            logger.warning(f"Cannot update state for unknown job_id: {job_id}")
            return False
        
        job_state = self.active_jobs[job_id]

        if status is not None:
            job_state.status = status
            logger.info(f"Job {job_id} status updated to: {status.name}")
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                job_state.end_time = time.time()
                duration = job_state.end_time - (job_state.start_time or job_state.end_time)
                logger.info(f"Job {job_id} finished. Duration: {duration:.2f} seconds.")

        if step_increment > 0:
            job_state.current_step += step_increment
        
        if loss is not None:
            job_state.loss_history.append(loss)

        if error_message is not None:
            job_state.error_message = error_message
            job_state.status = JobStatus.FAILED
            if job_state.end_time is None:
                 job_state.end_time = time.time()
            logger.error(f"Job {job_id} failed with error: {error_message}")

        logger.debug(f"Job {job_id} state updated. Step: {job_state.current_step}")
        return True

    def get_job_state(self, job_id: str) -> Optional[TrainingJobState]:
        """Gets the current state of a specific job.

        Args:
            job_id (str): The ID of the job.

        Returns:
            Optional[TrainingJobState]: The job state object, or None if not found.
        """
        return self.active_jobs.get(job_id)

    def get_all_active_job_states(self) -> Dict[str, TrainingJobState]:
        """Gets the states of all currently registered jobs (including completed/failed)."""
        return self.active_jobs
    
    def get_active_runnable_job_states(self) -> Dict[str, TrainingJobState]:
        """Gets the states of jobs that are currently ACTIVE and not yet completed/failed."""
        return { jid: state for jid, state in self.active_jobs.items() 
                 if state.status == JobStatus.ACTIVE }

    def complete_job(self, job_id: str) -> bool:
        """Marks a job as completed (alternative to update_job_state with status)."""
        logger.info(f"Marking job {job_id} as completed.")
        return self.update_job_state(job_id, status=JobStatus.COMPLETED)
    
    def fail_job(self, job_id: str, error_message: str) -> bool:
         """Marks a job as failed (alternative to update_job_state with error)."""
         logger.info(f"Marking job {job_id} as failed.")
         return self.update_job_state(job_id, status=JobStatus.FAILED, error_message=error_message)

    # Optional: Method to remove completed/failed jobs after some time?
    # def cleanup_jobs(self):
    #     pass 