# Abstract Base Class / Interface for Prioritization Strategies

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BasePrioritizationStrategy(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def select_next_job(self, active_jobs: Dict[str, Any]) -> Optional[str]:
        """Selects the next training job ID to run based on the strategy.

        Args:
            active_jobs: A dictionary where keys are job IDs and values are job states.

        Returns:
            The ID of the selected job, or None if no suitable job is found.
        """
        pass 
        
    def update_state(self, event_type: str, event_details: Dict[str, Any]):
        """Allows strategies to update their internal state based on system events.
        
        This is optional for simple strategies but required for stateful ones
        (e.g., tracking loss progress).
        
        Args:
            event_type: The type of event (e.g., 'STEP_COMPLETED', 'JOB_ADDED').
            event_details: A dictionary containing details about the event.
        """
        pass