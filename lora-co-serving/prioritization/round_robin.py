# Baseline Round-Robin Prioritization Strategy

import logging
from typing import Dict, Optional, List, Any

from .base_strategy import BasePrioritizationStrategy

logger = logging.getLogger(__name__)

class RoundRobinStrategy(BasePrioritizationStrategy):
    def __init__(self):
        self.current_index = 0
        self.job_order: List[str] = []
        self._last_job_ids = set()

    def select_next_job(self, active_jobs: Dict[str, Any]) -> Optional[str]:
        """Selects the next job ID in a round-robin fashion.

        Args:
            active_jobs (Dict[str, Any]): A dictionary of active, runnable job states,
                                         keyed by job_id.

        Returns:
            Optional[str]: The job_id of the next job, or None if no active jobs.
        """
        if not active_jobs:
            self.job_order = []
            self._last_job_ids = set()
            self.current_index = 0
            return None

        current_job_ids = set(active_jobs.keys())

        if current_job_ids != self._last_job_ids:
            logger.info("Active job set changed. Rebuilding round-robin order.")
            self.job_order = sorted(list(current_job_ids))
            self._last_job_ids = current_job_ids
            if self.current_index >= len(self.job_order):
                self.current_index = 0 
        
        if not self.job_order:
             logger.warning("RoundRobinStrategy: Job order is empty but active jobs exist. Rebuilding.")
             self.job_order = sorted(list(current_job_ids))
             self._last_job_ids = current_job_ids
             self.current_index = 0
             if not self.job_order:
                  return None

        selected_job_id = self.job_order[self.current_index]

        self.current_index = (self.current_index + 1) % len(self.job_order)

        logger.debug(f"RoundRobin selected job: {selected_job_id}")
        return selected_job_id 