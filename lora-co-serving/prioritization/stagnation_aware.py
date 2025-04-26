# lora-co-serving/prioritization/stagnation_aware.py

import logging
from typing import Optional, List, Dict, Any, Deque
from collections import deque

from .base_strategy import BasePrioritizationStrategy
from .round_robin import RoundRobinStrategy

logger = logging.getLogger(__name__)

class ForwardStagnationAwareStrategy(BasePrioritizationStrategy):
    """
    Prioritizes jobs that are showing better progress (e.g., faster loss decrease).
    De-prioritizes jobs whose progress has slowed down or stalled (stagnated).
    Goal: Maximize resource allocation towards currently 'productive' jobs.
    """
    def __init__(self, history_window: int = 10, stagnation_threshold: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.history_window = history_window
        self.stagnation_threshold = stagnation_threshold
        self.job_progress_history: Dict[str, Deque[float]] = {}
        logger.info(f"ForwardStagnationAwareStrategy initialized (window={history_window}, threshold={stagnation_threshold}).")

    def _calculate_progress_rate(self, job_id: str) -> Optional[float]:
        """Calculates the rate of progress (e.g., average loss decrease) for a job."""
        history = self.job_progress_history.get(job_id)
        if not history or len(history) < 2:
            return None

        diffs = [history[i] - history[i+1] for i in range(len(history) - 1)]
        if not diffs:
            return 0.0
        avg_decrease = sum(diffs) / len(diffs)
        return avg_decrease


    def select_next_job(self, active_jobs: Dict[str, Any]) -> Optional[str]:
        """
        Selects the active job with the best progress rate, excluding stagnated ones.

        Args:
            active_jobs: A dictionary where keys are job IDs and values are job states.

        Returns:
            The ID of the selected job, or None if no suitable job is found.
        """
        if not active_jobs:
            logger.debug("FSA: No active jobs to select from.")
            return None

        candidate_jobs = []

        for job_id in active_jobs.keys():
            progress_rate = self._calculate_progress_rate(job_id)

            if progress_rate is None:
                logger.debug(f"FSA: No progress history for job {job_id}, assigning neutral rate.")
                progress_rate = 0.0

            if progress_rate >= self.stagnation_threshold:
                 candidate_jobs.append((progress_rate, job_id))
                 logger.debug(f"FSA: Job {job_id} is progressing (rate={progress_rate:.4f}). Added as candidate.")
            else:
                 logger.debug(f"FSA: Job {job_id} is considered stagnated (rate={progress_rate:.4f} < threshold={self.stagnation_threshold}). Excluding.")


        if not candidate_jobs:
            logger.warning("FSA: No non-stagnated active jobs found. Selecting based on least progress?")
            rr_fallback = RoundRobinStrategy()
            selected_job_id = rr_fallback.select_next_job(active_jobs)
            logger.warning(f"FSA: Fallback to Round Robin selected: {selected_job_id}")
            return selected_job_id


        candidate_jobs.sort(key=lambda x: x[0], reverse=True)
        selected_job_id = candidate_jobs[0][1]
        logger.info(f"FSA: Selected job {selected_job_id} with progress rate {candidate_jobs[0][0]:.4f}.")

        return selected_job_id


    def update_state(self, event_type: str, event_details: Dict[str, Any]):
        """Update job progress history based on training events."""
        job_id = event_details.get('job_id')

        if not job_id:
            return # Cannot process without job_id

        # Check for completion status FIRST
        if event_type in ['STEP_COMPLETED', 'METRICS_REPORTED', 'STATUS_UPDATE'] and \
           event_details.get('status') == 'COMPLETED':
             if job_id in self.job_progress_history:
                 del self.job_progress_history[job_id]
                 logger.debug(f"FSA: Removed progress history for completed job {job_id}.")
             # Stop processing this event further after handling completion
             return

        # If not completed, check for loss updates from relevant events
        if event_type in ['STEP_COMPLETED', 'METRICS_REPORTED']:
            loss = event_details.get('loss')

            if loss is not None:
                try:
                    loss_float = float(loss)
                    # Only create deque if conversion succeeds and it doesn't exist
                    if job_id not in self.job_progress_history:
                        self.job_progress_history[job_id] = deque(maxlen=self.history_window)

                    self.job_progress_history[job_id].append(loss_float)
                    logger.debug(f"FSA: Updated progress history for {job_id}. New loss: {loss_float:.4f}. History size: {len(self.job_progress_history[job_id])}")
                except (ValueError, TypeError):
                     logger.warning(f"FSA: Received non-numeric loss value for job {job_id}: {loss}. Ignoring update.")


class ReverseStagnationAwareStrategy(BasePrioritizationStrategy):
    """
    Prioritizes jobs whose progress has slowed down or stalled.
    Goal: Help struggling jobs overcome plateaus by giving them more resources/steps.
    """
    def __init__(self, history_window: int = 10, slow_progress_threshold: float = 0.005, **kwargs):
        super().__init__(**kwargs)
        self.history_window = history_window
        self.slow_progress_threshold = slow_progress_threshold
        self.job_progress_history: Dict[str, Deque[float]] = {}
        logger.info(f"ReverseStagnationAwareStrategy initialized (window={history_window}, threshold={slow_progress_threshold}).")

    def _calculate_progress_rate(self, job_id: str) -> Optional[float]:
        """Calculates the rate of progress (e.g., average loss decrease) for a job."""
        history = self.job_progress_history.get(job_id)
        if not history or len(history) < 2:
            return None
        diffs = [history[i] - history[i+1] for i in range(len(history) - 1)]
        if not diffs: return 0.0
        return sum(diffs) / len(diffs)


    def select_next_job(self, active_jobs: Dict[str, Any]) -> Optional[str]:
        """
        Selects the active job with the slowest progress rate below a threshold.

        Args:
            active_jobs: A dictionary where keys are job IDs and values are job states.

        Returns:
            The ID of the selected job, or None if no suitable job is found.
        """
        if not active_jobs:
            logger.debug("RSA: No active jobs to select from.")
            return None

        struggling_jobs = []
        other_jobs = []

        for job_id in active_jobs.keys():
            progress_rate = self._calculate_progress_rate(job_id)

            if progress_rate is None:
                other_jobs.append(job_id)
                logger.debug(f"RSA: No progress history for job {job_id}. Considered non-struggling.")
            elif progress_rate < self.slow_progress_threshold:
                 struggling_jobs.append((progress_rate, job_id))
                 logger.debug(f"RSA: Job {job_id} is struggling (rate={progress_rate:.4f} < threshold={self.slow_progress_threshold}). Prioritizing.")
            else:
                 other_jobs.append(job_id)
                 logger.debug(f"RSA: Job {job_id} progressing adequately (rate={progress_rate:.4f}).")

        if struggling_jobs:
            struggling_jobs.sort(key=lambda x: x[0])
            selected_job_id = struggling_jobs[0][1]
            logger.info(f"RSA: Selected struggling job {selected_job_id} with progress rate {struggling_jobs[0][0]:.4f}.")
            return selected_job_id
        elif other_jobs:
            logger.info("RSA: No struggling jobs found. Falling back to Round Robin among remaining jobs.")
            fallback_active_jobs = {job_id: active_jobs[job_id] for job_id in other_jobs}
            rr_fallback = RoundRobinStrategy()
            selected_job_id = rr_fallback.select_next_job(fallback_active_jobs)
            logger.info(f"RSA: Fallback Round Robin selected: {selected_job_id}")
            return selected_job_id

        else:
            logger.warning("RSA: No jobs found to select (neither struggling nor other).")
            return None

    def update_state(self, event_type: str, event_details: Dict[str, Any]):
        """Update job progress history based on training events."""
        job_id = event_details.get('job_id')

        if not job_id:
            return # Cannot process without job_id

        # Check for completion status FIRST
        if event_type in ['STEP_COMPLETED', 'METRICS_REPORTED', 'STATUS_UPDATE'] and \
           event_details.get('status') == 'COMPLETED':
            if job_id in self.job_progress_history:
                del self.job_progress_history[job_id]
                logger.debug(f"RSA: Removed progress history for completed job {job_id}.")
            # Stop processing this event further after handling completion
            return

        # If not completed, check for loss updates from relevant events
        if event_type in ['STEP_COMPLETED', 'METRICS_REPORTED']:
            loss = event_details.get('loss')

            if loss is not None:
                try:
                    loss_float = float(loss)
                    # Only create deque if conversion succeeds and it doesn't exist
                    if job_id not in self.job_progress_history:
                        self.job_progress_history[job_id] = deque(maxlen=self.history_window)

                    self.job_progress_history[job_id].append(loss_float)
                    logger.debug(f"RSA: Updated progress history for {job_id}. New loss: {loss_float:.4f}. History size: {len(self.job_progress_history[job_id])}")
                except (ValueError, TypeError):
                    logger.warning(f"RSA: Received non-numeric loss value for job {job_id}: {loss}. Ignoring update.")
