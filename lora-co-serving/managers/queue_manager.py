# Manager for Inference/Training Request Queues

import logging
from queue import Queue, Empty
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class QueueManager:
    def __init__(self, max_inference_queue_size=0, max_training_queue_size=0):
        """Initializes the queues.

        Args:
            max_inference_queue_size (int): Max size for inference queue (0 for unlimited).
            max_training_queue_size (int): Max size for training job queue (0 for unlimited).
        """
        self.inference_queue = Queue(maxsize=max_inference_queue_size)
        self.training_job_queue = Queue(maxsize=max_training_queue_size)
        logger.info(f"QueueManager initialized (Inf Max Size: {max_inference_queue_size}, Train Max Size: {max_training_queue_size})")

    def add_inference_request(self, request: Dict[str, Any]) -> bool:
        """Adds an inference request to the queue.

        Args:
            request (Dict[str, Any]): The request dictionary.

        Returns:
            bool: True if added successfully, False if queue is full.
        """
        try:
            self.inference_queue.put_nowait(request)
            logger.debug(f"Added inference request: {request.get('request_id', 'N/A')}")
            return True
        except Queue.Full:
            logger.warning("Inference request queue is full. Request rejected.")
            return False

    def get_inference_batch(self, max_batch_size: int) -> List[Dict[str, Any]]:
        """Gets a batch of inference requests from the queue.

        Args:
            max_batch_size (int): The maximum number of requests to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of inference requests (might be empty).
        """
        batch = []
        while len(batch) < max_batch_size:
            try:
                request = self.inference_queue.get_nowait()
                batch.append(request)
                logger.debug(f"Dequeued inference request: {request.get('request_id', 'N/A')}")
            except Empty:
                break
        if batch:
             logger.info(f"Returning inference batch of size {len(batch)}.")
        return batch

    def add_training_job(self, job_details: Dict[str, Any]) -> bool:
        """Adds a new training job request to the queue.

        Args:
            job_details (Dict[str, Any]): Dictionary containing job details.

        Returns:
            bool: True if added successfully, False if queue is full.
        """
        try:
            self.training_job_queue.put_nowait(job_details)
            logger.info(f"Added training job request: {job_details.get('job_id', 'N/A')}")
            return True
        except Queue.Full:
            logger.warning("Training job queue is full. Job request rejected.")
            return False

    def get_new_training_job(self) -> Optional[Dict[str, Any]]:
        """Gets the next pending training job from the queue.

        Returns:
            Optional[Dict[str, Any]]: Job details dictionary, or None if queue is empty.
        """
        try:
            job_details = self.training_job_queue.get_nowait()
            logger.info(f"Dequeued training job request: {job_details.get('job_id', 'N/A')}")
            return job_details
        except Empty:
            return None

    def get_inference_queue_size(self) -> int:
        """Returns the approximate size of the inference queue."""
        return self.inference_queue.qsize()

    def get_training_queue_size(self) -> int:
        """Returns the approximate size of the training job queue."""
        return self.training_job_queue.qsize() 