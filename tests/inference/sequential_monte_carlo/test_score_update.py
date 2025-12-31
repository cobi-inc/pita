import pytest
import math
from pita.sampling.smc import Sequential_Monte_Carlo

class TestScoreUpdate:
    
    # Fixture for basic SMC setup? Not strictly necessary given simple init, 
    # but good for consistency. We'll instantiate in tests for clarity of params.

    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_score_update_last(self, metric):
        """Test 'last' aggregation for logprobs and power_distribution."""
        smc = Sequential_Monte_Carlo(token_metric=metric, aggregation="last")
        token_values = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        token_count = 3
        step_scores = [-0.1]
        
        # Expected: average of last 3 tokens: (-0.4 + -0.5 + -0.6) / 3 = -0.5
        new_score = smc.score_update(token_values, token_count, step_scores)
        # Check step_scores updated correctly
        assert step_scores[0] == -0.1
        assert step_scores[-1] == pytest.approx(-0.5)
        assert step_scores[-1] == (sum(token_values[-token_count:]) / token_count)
        # Test that the score was updated correctly
        assert new_score == pytest.approx(-0.5)
        assert new_score == (sum(token_values[-token_count:]) / token_count)

    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_score_update_minimum(self, metric):
        """Test 'minimum' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric=metric, aggregation="minimum")
        token_values = [-0.9, -0.9, -0.2, -0.1, -0.1, -0.1, -0.9, -0.4, -1.2] # avg last 3 is -0.833333
        token_count = 3
        # Existing step_scores has a lower value
        step_scores = [-0.667, -0.667] 
        
        # New step score calculation: average of last 3 tokens (-0.9 + -0.4 + -1.2) / 3 = -0.833333
        # After update, step_scores = [-0.667, -0.667, -0.833333] and the minimum is -0.833333
        new_score = smc.score_update(token_values, token_count, step_scores)

        # Check step_scores updated correctly
        assert len(step_scores) == 3
        assert step_scores[0] == pytest.approx(-0.667)
        assert step_scores[1] == pytest.approx(-0.667)
        assert step_scores[-1] == pytest.approx(-0.833333)
        assert step_scores[-1] == sum(token_values[-token_count:]) / token_count
        # Test that the score was updated correctly
        assert new_score == pytest.approx(-0.833333)
        assert new_score == min(step_scores)

        # Case where new score is the minimum
        token_values.append(-2)
        token_values.append(-2)
        token_values.append(-2)
        new_score = smc.score_update(token_values, token_count, step_scores)
        # Test that the step scores were updated correctly
        assert len(step_scores) == 4
        assert step_scores[-1] == pytest.approx(-2)
        assert step_scores[-1] == sum(token_values[-token_count:]) / token_count
        # avg -2, min of [-0.667, -0.667, -0.833, -2] is -2
        assert new_score == pytest.approx(-2)
        assert new_score == min(step_scores)


    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_score_update_product(self, metric):
        """Test 'product' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric=metric, aggregation="product")
        token_values = [-2.0, -2.0, -2.0]
        token_count = 3
        step_scores = [1, -2.0] 
        
        # New step val: (-2.0 + -2.0 + -2.0)/3 = -2.0
        # Product of [1, -2.0, -2.0] = -4.0
        # Return: -1 * abs(-4.0) = -4.0
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        
        # Test that the score was updated correctly
        assert new_score == pytest.approx(-4.0)
        assert new_score == -1*abs(step_scores[0]*step_scores[1]*step_scores[2])
        
        # Test that the step scores were updated correctly
        assert step_scores[-1] == pytest.approx(-2.0)
        assert step_scores[-1] == (sum(token_values) / len(token_values))

    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_score_update_model_aggregate(self, metric):
        """Test 'model_aggregate' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric=metric, aggregation="model_aggregate")
        token_values = [0.1, 0.2, 0.3, 0.4]
        token_count = 2
        step_scores = [0.15]
        
        # New step val (last 2): (0.3+0.4)/2 = 0.35 (appended to step_scores)
        # Return: average of ALL token_values: sum(0.1..0.4) = 1.0 / 4 = 0.25
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        # Check the step scores were updated correctly
        assert step_scores[-1] == pytest.approx(0.35)
        assert step_scores[-1] == (sum(token_values[2:]) / len(token_values[2:]))
        # Check the new score was updated correctly
        assert new_score == pytest.approx(0.25)
        assert new_score == (sum(token_values) / len(token_values))


    def test_score_update_entropy_last(self):
        """Test 'entropy' metric with 'last' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric="entropy", aggregation="last")
        token_values = [1.0, 2.0, 3.0]
        token_count = 3
        step_scores = [-1.0]
        
        # Step score: (1+2+3)/3 = 2.0
        # Return: -1 * 2.0 = -2.0
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        # Check that the step_scores were updated correctly
        assert step_scores[0] == -1.0
        assert step_scores[-1] == pytest.approx(-2.0)
        assert step_scores[-1] == -1 * sum(token_values[-token_count:]) / token_count
        # Check that the new score was updated correctly
        assert new_score == pytest.approx(-2.0)
        assert new_score == step_scores[-1]


    def test_score_update_entropy_minimum(self):
        """Test 'entropy' metric with 'minimum' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric="entropy", aggregation="minimum")
        token_values = [3.0]
        token_count = 1
        step_scores = [-1.0, -5.0]
        
        # Step score: 3.0
        # List becomes [1.0, 5.0, 3.0]
        # Return: -1 * max(step_scores) = -1 * 5.0 = -5.0
        # Note: logic in code is `-1 * max(step_scores)` for entropy minimum
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        # Check that the step_scores were updated correctly
        assert step_scores[-1] == pytest.approx(-3.0)
        assert step_scores[-1] == -1 * sum(token_values[-token_count:]) / token_count
        # Check that the new score was updated correctly
        assert new_score == pytest.approx(-5.0)
        assert new_score == step_scores[1]


    def test_score_update_entropy_product(self):
        """Test 'entropy' metric with 'product' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric="entropy", aggregation="product")
        token_values = [2.0]
        token_count = 1
        step_scores = [-3.0, -4.0]
        
        # Step score: 2.0
        # List: [3.0, 4.0, 2.0]
        # Return: -1 * prod = -1 * (24.0) = -24.0
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        # Check that the step_scores were updated correctly
        assert step_scores[-1] == pytest.approx(-2.0)
        assert step_scores[-1] == -1 * sum(token_values[-token_count:]) / token_count
        # Check that the new score was updated correctly
        assert new_score == pytest.approx(-24.0)
        assert new_score == -1 * abs(math.prod(step_scores))

    def test_score_update_entropy_model_aggregate(self):
        """Test 'entropy' metric with 'model_aggregate'."""
        smc = Sequential_Monte_Carlo(token_metric="entropy", aggregation="model_aggregate")
        token_values = [10.0, 20.0, 30.0]
        token_count = 1
        step_scores = [-10.0, -15.0]
        
        # Step: 20.0
        # Return: -1 * average(all) = -1 * 15.0 = -15.0
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        # Check that the step_scores were updated correctly
        assert step_scores[-1] == pytest.approx(-30.0)
        assert step_scores[-1] == -1 * sum(token_values[-token_count:]) / token_count
        # Check that the new score was updated correctly
        assert new_score == pytest.approx(-20)
        assert new_score == -1 * sum(token_values) / len(token_values)

    def test_invalid_aggregation(self):
        smc = Sequential_Monte_Carlo(token_metric="logprobs", aggregation="invalid")
        with pytest.raises(ValueError, match="Invalid aggregation method"):
            smc.score_update([1.0], 1, [])

    def test_invalid_metric(self):
        smc = Sequential_Monte_Carlo(token_metric="invalid", aggregation="last")
        with pytest.raises(ValueError, match="Invalid token metric"):
            smc.score_update([1.0], 1, [])