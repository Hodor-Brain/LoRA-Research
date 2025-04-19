# Abstract Base Class / Interface for Prioritization Strategies

from abc import ABC, abstractmethod

class BasePrioritizationStrategy(ABC):

    @abstractmethod
    def select_next_job(self, active_jobs):
        """Selects the next training job ID to run based on the strategy.

        Args:
            active_jobs (dict): A dictionary containing the state of active training jobs.
                                 (Structure TBD, but likely includes job_id, loss_history, step_count etc.)

        Returns:
            str: The job_id of the selected training job, or None if no job is selected.
        """
        pass 