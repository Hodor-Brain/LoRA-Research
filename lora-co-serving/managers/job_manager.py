# Manager for Active Training Job State

class ActiveTrainingJobManager:
    def __init__(self):
        self.active_jobs = {}

    def register_job(self, job_id, job_config):
        pass

    def update_job_state(self, job_id, state_update):
        pass

    def get_job_state(self, job_id):
        pass

    def get_all_active_job_states(self):
        return self.active_jobs

    def complete_job(self, job_id):
        pass 