# tests/test_prioritization_stagnation.py

import pytest
from collections import deque
from prioritization.stagnation_aware import ForwardStagnationAwareStrategy, ReverseStagnationAwareStrategy
from prioritization.least_progress_first import LeastProgressFirstStrategy
from dataclasses import dataclass

# --- Fixtures ---

@pytest.fixture
def forward_strategy():
    return ForwardStagnationAwareStrategy(history_window=5, stagnation_threshold=0.001)

@pytest.fixture
def reverse_strategy():
    return ReverseStagnationAwareStrategy(history_window=5, slow_progress_threshold=0.005)

@pytest.fixture
def lpf_strategy():
    return LeastProgressFirstStrategy()

# --- Mock Job State Object ---
@dataclass
class MockJobState:
    current_step: int = 0

# --- Test ForwardStagnationAwareStrategy ---

class TestForwardStagnationAwareStrategy:

    def test_calculate_progress_rate_no_history(self, forward_strategy):
        assert forward_strategy._calculate_progress_rate("job1") is None

    def test_calculate_progress_rate_short_history(self, forward_strategy):
        forward_strategy.job_progress_history["job1"] = deque([1.0], maxlen=5)
        assert forward_strategy._calculate_progress_rate("job1") is None

    def test_calculate_progress_rate_decreasing_loss(self, forward_strategy):
        forward_strategy.job_progress_history["job1"] = deque([1.0, 0.9, 0.8, 0.7], maxlen=5)
        assert forward_strategy._calculate_progress_rate("job1") == pytest.approx(0.1)

    def test_calculate_progress_rate_increasing_loss(self, forward_strategy):
        forward_strategy.job_progress_history["job1"] = deque([0.7, 0.8, 0.9, 1.0], maxlen=5)
        assert forward_strategy._calculate_progress_rate("job1") == pytest.approx(-0.1)

    def test_calculate_progress_rate_flat_loss(self, forward_strategy):
        forward_strategy.job_progress_history["job1"] = deque([0.8, 0.8, 0.8, 0.8], maxlen=5)
        assert forward_strategy._calculate_progress_rate("job1") == pytest.approx(0.0)

    def test_update_state_add_loss(self, forward_strategy):
        event = {'job_id': 'job1', 'loss': 0.95}
        forward_strategy.update_state('STEP_COMPLETED', event)
        assert 'job1' in forward_strategy.job_progress_history
        assert forward_strategy.job_progress_history['job1'] == deque([0.95])

        event2 = {'job_id': 'job1', 'loss': '0.90'} # Test string conversion
        forward_strategy.update_state('METRICS_REPORTED', event2)
        assert forward_strategy.job_progress_history['job1'] == deque([0.95, 0.90])

    def test_update_state_ignore_invalid_loss(self, forward_strategy):
        event = {'job_id': 'job1', 'loss': 'invalid'}
        forward_strategy.update_state('STEP_COMPLETED', event)
        assert 'job1' not in forward_strategy.job_progress_history

    def test_update_state_job_completion(self, forward_strategy):
        forward_strategy.job_progress_history['job1'] = deque([0.8, 0.7])
        event = {'job_id': 'job1', 'status': 'COMPLETED'}
        event_loss = {'job_id': 'job1', 'loss': 0.6, 'status': 'COMPLETED'} # Example event
        forward_strategy.update_state('STEP_COMPLETED', event_loss)
        assert 'job1' not in forward_strategy.job_progress_history

    def test_select_next_job_empty(self, forward_strategy):
        assert forward_strategy.select_next_job({}) is None

    def test_select_next_job_clear_winner(self, forward_strategy):
        active_jobs = {"job1": {}, "job2": {}, "job3": {}}
        # job1: Good progress
        forward_strategy.job_progress_history["job1"] = deque([0.5, 0.4, 0.3], maxlen=5)
        # job2: Stagnated
        forward_strategy.job_progress_history["job2"] = deque([0.6, 0.6, 0.6], maxlen=5)
        # job3: Little progress (below threshold 0.001)
        forward_strategy.job_progress_history["job3"] = deque([0.7, 0.7, 0.6995], maxlen=5)
        assert forward_strategy.select_next_job(active_jobs) == "job1"

    def test_select_next_job_all_stagnated_fallback(self, forward_strategy):
        active_jobs = {"job1": {}, "job2": {}}
        # job1: Stagnated (flat)
        forward_strategy.job_progress_history["job1"] = deque([0.5, 0.5, 0.5], maxlen=5)
        # job2: Stagnated (increasing loss)
        forward_strategy.job_progress_history["job2"] = deque([0.6, 0.7, 0.8], maxlen=5)
        assert forward_strategy.select_next_job(active_jobs) == "job1"


# --- Test ReverseStagnationAwareStrategy ---

class TestReverseStagnationAwareStrategy:

    def test_update_state_add_loss(self, reverse_strategy):
        event = {'job_id': 'job1', 'loss': 0.95}
        reverse_strategy.update_state('STEP_COMPLETED', event)
        assert reverse_strategy.job_progress_history['job1'] == deque([0.95])

    def test_select_next_job_empty(self, reverse_strategy):
        assert reverse_strategy.select_next_job({}) is None

    def test_select_next_job_clear_struggler(self, reverse_strategy):
        active_jobs = {"job1": {}, "job2": {}, "job3": {}}
        # job1: Good progress (above threshold 0.005)
        reverse_strategy.job_progress_history["job1"] = deque([0.5, 0.4, 0.3], maxlen=5)
        # job2: Struggling (just below threshold)
        reverse_strategy.job_progress_history["job2"] = deque([0.6, 0.598, 0.597], maxlen=5)
        # job3: Very slow / borderline flat (most struggling)
        reverse_strategy.job_progress_history["job3"] = deque([0.7, 0.7, 0.6999], maxlen=5)
        assert reverse_strategy.select_next_job(active_jobs) == "job3"

    def test_select_next_job_no_strugglers_fallback(self, reverse_strategy):
        active_jobs = {"job1": {}, "job2": {}}
        # job1: Good progress
        reverse_strategy.job_progress_history["job1"] = deque([0.5, 0.4, 0.3], maxlen=5)
        # job2: Also good progress
        reverse_strategy.job_progress_history["job2"] = deque([0.6, 0.55, 0.5], maxlen=5)
        assert reverse_strategy.select_next_job(active_jobs) == "job1"

    def test_select_next_job_mixed_with_no_history(self, reverse_strategy):
        active_jobs = {"job1": {}, "job2": {}, "job3": {}}
        # job1: Struggling
        reverse_strategy.job_progress_history["job1"] = deque([0.6, 0.598, 0.597], maxlen=5)
        # job2: Good progress
        reverse_strategy.job_progress_history["job2"] = deque([0.5, 0.4, 0.3], maxlen=5)
        # job3: No history (considered 'other', not struggling)
        assert reverse_strategy.select_next_job(active_jobs) == "job1"

    def test_select_next_job_only_no_history_fallback(self, reverse_strategy):
        active_jobs = {"job1": {}, "job2": {}}
        assert reverse_strategy.select_next_job(active_jobs) == "job1"


# --- Test LeastProgressFirstStrategy ---

class TestLeastProgressFirstStrategy:

    def test_select_next_job_empty(self, lpf_strategy):
        assert lpf_strategy.select_next_job({}) is None

    def test_select_next_job_one_job(self, lpf_strategy):
        active_jobs = {"job1": MockJobState(current_step=5)}
        assert lpf_strategy.select_next_job(active_jobs) == "job1"

    def test_select_next_job_clear_winner(self, lpf_strategy):
        active_jobs = {
            "job1": MockJobState(current_step=10),
            "job2": MockJobState(current_step=5),
            "job3": MockJobState(current_step=15)
        }
        assert lpf_strategy.select_next_job(active_jobs) == "job2"

    def test_select_next_job_tie(self, lpf_strategy):
        active_jobs = {
            "job1": MockJobState(current_step=10),
            "job2": MockJobState(current_step=5),
            "job3": MockJobState(current_step=15),
            "job4": MockJobState(current_step=5)
        }
        assert lpf_strategy.select_next_job(active_jobs) == "job2"

    def test_select_next_job_non_integer_steps(self, lpf_strategy):
        active_jobs = {
            "job1": MockJobState(current_step=10),
            "job2": MockJobState(current_step="invalid"),
            "job3": MockJobState(current_step=5)
        }
        assert lpf_strategy.select_next_job(active_jobs) == "job3"

    def test_select_next_job_missing_attribute(self, lpf_strategy):
        @dataclass
        class BadMockJobState:
            other_field: int = 0
            
        active_jobs = {
            "job1": MockJobState(current_step=10),
            "job2": BadMockJobState(other_field=5),
            "job3": MockJobState(current_step=5)
        }
        assert lpf_strategy.select_next_job(active_jobs) == "job3"

    def test_update_state_runs(self, lpf_strategy):
        try:
            lpf_strategy.update_state('ANY_EVENT', {'detail': 'value'})
        except Exception as e:
            pytest.fail(f"LPF update_state raised an exception: {e}")
        