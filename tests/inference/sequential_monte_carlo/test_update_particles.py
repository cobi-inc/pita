
import pytest
from pita.sampling.smc import Sequential_Monte_Carlo
from pita.inference.LLM_backend import Output

class TestUpdateParticles:
    def _create_full_output(self, id_val):
        """Helper to create a fully initialized Output object with distinct values based on id_val."""
        return Output(
            tokens=[id_val, id_val],
            top_k_logits=[float(id_val), float(id_val)],
            top_k_logprobs=[float(id_val)*0.1, float(id_val)*0.1],
            unprocessed_log_normalization_constant=[float(id_val)*0.01, float(id_val)*0.01],
            temp_processed_log_normalization_constant=[float(id_val)*0.001, float(id_val)*0.001],
            entropy=[float(id_val)*0.0001, float(id_val)*0.0001]
        )

    def test_update_particles_basic(self):
        """
        Test basic cloning behavior where one particle takes over others.
        Using fully initialized Output objects.
        """
        num_particles = 3
        smc = Sequential_Monte_Carlo(num_particles=num_particles)
        
        # Create fully initialized Output objects
        outputs = [
            self._create_full_output(0),
            self._create_full_output(1),
            self._create_full_output(2)
        ]
        
        finished = [False, False, False]
        token_metric_scores = [[0.1, 0.11], [0.2, 0.22], [0.3, 0.33]]
        step_scores = [[1.0, 1.1], [2.0, 2.2], [3.0, 3.3]]
        
        # New particles indices: particles 0 and 2 become copies of particle 1
        new_particles = [1, 1, 1]
        
        smc.update_particles(new_particles, outputs, finished, token_metric_scores, step_scores)
        
        # Verification
        # P0 should be P1 data
        assert outputs[0].tokens == [1, 1]
        assert outputs[0].top_k_logits == [1.0, 1.0]
        assert outputs[0].top_k_logprobs == [0.1, 0.1]
        assert outputs[0].unprocessed_log_normalization_constant == [0.01, 0.01]
        assert outputs[0].temp_processed_log_normalization_constant == [0.001, 0.001]
        assert outputs[0].entropy == [0.0001, 0.0001]
        assert outputs[1] is not outputs[0] # Deep copy check

        # P1 should be P1 data
        assert outputs[1].tokens == [1, 1]
        assert outputs[1].top_k_logits == [1.0, 1.0]
        assert outputs[1].top_k_logprobs == [0.1, 0.1]
        assert outputs[1].unprocessed_log_normalization_constant == [0.01, 0.01]
        assert outputs[1].temp_processed_log_normalization_constant == [0.001, 0.001]
        assert outputs[1].entropy == [0.0001, 0.0001]
        assert outputs[1] is not outputs[0] # Deep copy check

        # P2 should be P2 data
        assert outputs[2].tokens == [1, 1]
        assert outputs[2].top_k_logits == [1.0, 1.0]
        assert outputs[2].top_k_logprobs == [0.1, 0.1]
        assert outputs[2].unprocessed_log_normalization_constant == [0.01, 0.01]
        assert outputs[2].temp_processed_log_normalization_constant == [0.001, 0.001]
        assert outputs[2].entropy == [0.0001, 0.0001]
        assert outputs[2] is not outputs[1] # Deep copy check

        # Verify the token_metric scores changed correctly
        token_metric_scores = [[0.2, 0.22], [0.2, 0.22], [0.2, 0.22]]

        # Verify the step scores changed correctly
        step_scores = [[2.0, 2.2], [2.0, 2.2], [2.0, 2.2]]

    def test_update_particles_basic_finished(self):
        """
        Test basic cloning behavior where one particle takes over others.
        Using fully initialized Output objects.
        """
        num_particles = 3
        smc = Sequential_Monte_Carlo(num_particles=num_particles)
        
        # Create fully initialized Output objects
        outputs = [
            self._create_full_output(0),
            self._create_full_output(1),
            self._create_full_output(2)
        ]
        
        finished = [True, False, False]
        token_metric_scores = [[0.1, 0.11], [0.2, 0.22], [0.3, 0.33]]
        step_scores = [[1.0, 1.1], [2.0, 2.2], [3.0, 3.3]]
        
        # New particles indices: particle 1 becomes copy of 0, particle 2 becomes copy of 0
        new_particles = [0, 1, 1]
        
        smc.update_particles(new_particles, outputs, finished, token_metric_scores, step_scores)
        
        # Verification
        # outputs[0] should not change
        assert outputs[0].tokens == [0, 0]
        assert outputs[0].top_k_logits == [0.0, 0.0]
        assert outputs[0].top_k_logprobs == [0.0, 0.0]
        assert outputs[0].unprocessed_log_normalization_constant == [0.0, 0.0]
        assert outputs[0].temp_processed_log_normalization_constant == [0.0, 0.0]
        assert outputs[0].entropy == [0.0, 0.0]

        # outputs[1] should be outputs[1] data
        assert outputs[1].tokens == [1, 1]
        assert outputs[1].top_k_logits == [1.0, 1.0]
        assert outputs[1].top_k_logprobs == [0.1, 0.1]
        assert outputs[1].unprocessed_log_normalization_constant == [0.01, 0.01]
        assert outputs[1].temp_processed_log_normalization_constant == [0.001, 0.001]
        assert outputs[1].entropy == [0.0001, 0.0001]
        assert outputs[1] is not outputs[0] # Deep copy check

        # outputs[2] should be outputs[1] data
        assert outputs[2].tokens == [1, 1]
        assert outputs[2].top_k_logits == [1.0, 1.0]
        assert outputs[2].top_k_logprobs == [0.1, 0.1]
        assert outputs[2].unprocessed_log_normalization_constant == [0.01, 0.01]
        assert outputs[2].temp_processed_log_normalization_constant == [0.001, 0.001]
        assert outputs[2].entropy == [0.0001, 0.0001]
        assert outputs[2] is not outputs[1] # Deep copy check

        # Verify the token_metric scores changed correctly
        token_metric_scores = [[0.1, 0.11], [0.2, 0.22], [0.2, 0.22]]

        # Verify the step scores changed correctly
        step_scores = [[1.0, 1.1], [2.0, 2.2], [2.0, 2.2]]

    def test_update_particles_cyclic(self):
        """
        Test a cyclic shuffle: 0->1, 1->2, 2->0.
        """
        num_particles = 3
        smc = Sequential_Monte_Carlo(num_particles=num_particles)
        
        outputs = [
            self._create_full_output(10),
            self._create_full_output(11),
            self._create_full_output(12)
        ]
        
        finished = [False, False, False]
        token_metric_scores = [[0.1, 0.15], [0.2, 0.25], [0.3, 0.35]]
        step_scores = [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]
        
        # outputs[0] takes from outputs[1]
        # outputs[1] takes from outputs[2]
        # outputs[2] takes from outputs[0]
        new_particles = [1, 2, 0] 
        
        smc.update_particles(new_particles, outputs, finished, token_metric_scores, step_scores)
        
        # Check outputs[0] (should be from old outputs[1] id=11)
        assert outputs[0].tokens == [11, 11]
        assert outputs[0].top_k_logits == pytest.approx([11.0, 11.0])
        assert outputs[0].top_k_logprobs == pytest.approx([1.1, 1.1])
        assert outputs[0].unprocessed_log_normalization_constant == pytest.approx([0.11, 0.11])
        assert outputs[0].temp_processed_log_normalization_constant == pytest.approx([0.011, 0.011])
        assert outputs[0].entropy == pytest.approx([0.0011, 0.0011])
        
        # Check outputs[1] (should be from old outputs[2] id=12)
        assert outputs[1].tokens == [12, 12]
        assert outputs[1].top_k_logits == pytest.approx([12.0, 12.0])
        assert outputs[1].top_k_logprobs == pytest.approx([1.2, 1.2])
        assert outputs[1].unprocessed_log_normalization_constant == pytest.approx([0.12, 0.12])
        assert outputs[1].temp_processed_log_normalization_constant == pytest.approx([0.012, 0.012])
        assert outputs[1].entropy == pytest.approx([0.0012, 0.0012])
        
        # Check outputs[2] (should be from old outputs[0] id=10)
        assert outputs[2].tokens == [10, 10]
        assert outputs[2].top_k_logits == pytest.approx([10.0, 10.0])
        assert outputs[2].top_k_logprobs == pytest.approx([1.0, 1.0])
        assert outputs[2].unprocessed_log_normalization_constant == pytest.approx([0.10, 0.10])
        assert outputs[2].temp_processed_log_normalization_constant == pytest.approx([0.010, 0.010])
        assert outputs[2].entropy == pytest.approx([0.0010, 0.0010])
        
        # Check the token_metric_scores
        assert token_metric_scores == [[0.2, 0.25], [0.3, 0.35],[0.1, 0.15]]

        # Check the step scores
        assert step_scores == [[2.0, 2.5], [3.0, 3.5], [1.0, 1.5]]

    def test_update_particles_with_finished(self):
        """
        Test that finished particles are respected (not updated).
        """
        num_particles = 4
        smc = Sequential_Monte_Carlo(num_particles=num_particles)
        
        outputs = [
            self._create_full_output(0),
            self._create_full_output(1),
            self._create_full_output(2),
            self._create_full_output(3)
        ]
        
        finished = [False, True, False, True] # 1 and 3 are finished
        token_metric_scores = [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
        step_scores = [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]
        
        # P0 takes from P2
        # P1 finished
        # P2 stays
        # P3 finished
        new_particles = [2, 1, 2, 3]
        
        smc.update_particles(new_particles, outputs, finished, token_metric_scores, step_scores)
        
        # outputs[0] should be new outputs[2] (id=2)
        assert outputs[0].tokens == [2, 2]
        assert outputs[0].top_k_logits == pytest.approx([2.0, 2.0])
        assert outputs[0].top_k_logprobs == pytest.approx([0.2, 0.2])
        assert outputs[0].unprocessed_log_normalization_constant == pytest.approx([0.02, 0.02])
        assert outputs[0].temp_processed_log_normalization_constant == pytest.approx([0.002, 0.002])
        assert outputs[0].entropy == pytest.approx([0.0002, 0.0002])
        
        # outputs[1] should still be outputs[1] (id=1)
        assert outputs[1].tokens == [1, 1] 
        assert outputs[1].top_k_logits == pytest.approx([1.0, 1.0])
        assert outputs[1].top_k_logprobs == pytest.approx([0.1, 0.1])
        assert outputs[1].unprocessed_log_normalization_constant == pytest.approx([0.01, 0.01])
        assert outputs[1].temp_processed_log_normalization_constant == pytest.approx([0.001, 0.001])
        assert outputs[1].entropy == pytest.approx([0.0001, 0.0001])
        
        # outputs[2] should still be outputs[2] (id=2)
        assert outputs[2].tokens == [2, 2]
        assert outputs[2].top_k_logits == pytest.approx([2.0, 2.0])
        assert outputs[2].top_k_logprobs == pytest.approx([0.2, 0.2])
        assert outputs[2].unprocessed_log_normalization_constant == pytest.approx([0.02, 0.02])
        assert outputs[2].temp_processed_log_normalization_constant == pytest.approx([0.002, 0.002])
        assert outputs[2].entropy == pytest.approx([0.0002, 0.0002])
        
        # outputs[3] should still be outputs[3] (id=3)
        assert outputs[3].tokens == [3, 3] 
        assert outputs[3].top_k_logits == pytest.approx([3.0, 3.0])
        assert outputs[3].top_k_logprobs == pytest.approx([0.3, 0.3])
        assert outputs[3].unprocessed_log_normalization_constant == pytest.approx([0.03, 0.03])
        assert outputs[3].temp_processed_log_normalization_constant == pytest.approx([0.003, 0.003])
        assert outputs[3].entropy == pytest.approx([0.0003, 0.0003])
        
        # Check the token_metric_scores
        assert token_metric_scores == [[2.0, 2.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
        
        # Check the step scores
        assert step_scores == [[2.0, 2.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]
