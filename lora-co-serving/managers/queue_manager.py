# Manager for Inference/Training Request Queues

class QueueManager:
    def __init__(self):
        self.inference_queue = [] # Simple list for now
        self.training_job_queue = []

    def add_inference_request(self, request):
        pass

    def get_inference_batch(self, max_batch_size):
        pass

    def add_training_job(self, job_details):
        pass

    def get_new_training_job(self):
        pass 