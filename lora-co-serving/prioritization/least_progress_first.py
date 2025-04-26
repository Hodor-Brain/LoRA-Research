# lora-co-serving/prioritization/least_progress_first.py

import logging
from typing import Optional, List, Dict, Any

from .base_strategy import BasePrioritizationStrategy

logger = logging.getLogger(__name__)

class LeastProgressFirstStrategy(BasePrioritizationStrategy):
    """
    Prioritizes the active training job that has completed the fewest steps.
    Handles ties by selecting the first job encountered with the minimum steps.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("LeastProgressFirstStrategy initialized.")

    def select_next_job(self, active_jobs: Dict[str, Any]) -> Optional[str]:
        """
        Selects the active job with the minimum completed steps.

        Args:
            active_jobs: A dictionary where keys are job IDs and values are
                         job state dictionaries (expected to contain 'steps_completed').

        Returns:
            The ID of the selected job, or None if no suitable job is found.
        """
        if not active_jobs:
            logger.warning("LPF: select_next_job called with no active jobs.")
            return None

        min_steps = float('inf')
        selected_job_id = None

        for job_id, job_state in active_jobs.items():
            try:
                steps_completed = job_state.current_step

                if isinstance(steps_completed, int) and steps_completed < min_steps:
                    min_steps = steps_completed
                    selected_job_id = job_id
                    logger.debug(f"LPF: New candidate {job_id} with {steps_completed} steps.")
                elif not isinstance(steps_completed, int):
                    logger.warning(f"LPF: Job {job_id} has non-integer current_step: {steps_completed}. Skipping.")

            except AttributeError:
                logger.error(f"LPF: Could not access 'current_step' attribute for job {job_id}. State type: {type(job_state)}, State: {job_state}")
                continue
            except Exception as e:
                logger.error(f"LPF: Error processing job {job_id}: {e}. State: {job_state}", exc_info=True)
                continue

        if selected_job_id:
            logger.info(f"LPF: Selected job {selected_job_id} with {min_steps} steps.")
        else:
            logger.warning("LPF: No suitable job found among active jobs (possibly due to missing step info).")

        return selected_job_id

    def update_state(self, event_type: str, event_details: Dict[str, Any]):
        """Update strategy state based on system events (e.g., job completion/addition)."""
        logger.debug(f"LPF: Received update event: {event_type}, Details: {event_details}. State not changed.")
        pass