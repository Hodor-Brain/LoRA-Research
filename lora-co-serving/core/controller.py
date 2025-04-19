# Main Controller / Scheduler Logic

class MainController:
    def __init__(self, config, engine, job_manager, adapter_manager, queue_manager, prioritization_strategy):
        self.config = config
        self.engine = engine
        self.job_manager = job_manager
        self.adapter_manager = adapter_manager
        self.queue_manager = queue_manager
        self.prioritization_strategy = prioritization_strategy

    def run_loop(self):
        pass

    def _decide_mode(self):
        pass

    def _execute_inference_batch(self):
        pass

    def _execute_training_batch(self):
        pass 