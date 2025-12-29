import pytest
import math
import numpy as np
from pita.sampling.smc import Sequential_Monte_Carlo

class TestScoreUpdate:
    
    # Fixture for basic SMC setup? Not strictly necessary given simple init, 
    # but good for consistency. We'll instantiate in tests for clarity of params.

    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_score_update_last(self, metric):
        """Test 'last' aggregation for logprobs and power_distribution."""
        smc = Sequential_Monte_Carlo(token_metric=metric, aggregation="last")
        token_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        token_count = 3
        step_scores = [0.1]
        
        # Expected: average of last 3 tokens: (0.4 + 0.5 + 0.6) / 3 = 0.5
        new_score = smc.score_update(token_values, token_count, step_scores)
        
        assert new_score == pytest.approx(0.5)
        # Check side effect: step_scores appended
        assert len(step_scores) == 2
        assert step_scores[-1] == pytest.approx(0.5)

    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_score_update_minimum(self, metric):
        """Test 'minimum' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric=metric, aggregation="minimum")
        token_values = [0.9, 0.9, 0.2, 0.3, 0.4] # avg last 3 is 0.3
        token_count = 3
        # Existing step_scores has a lower value
        step_scores = [0.1, 0.5] 
        
        # New step score calculation: (0.2 + 0.3 + 0.4) / 3 = 0.3
        # Minimum of [0.1, 0.5, 0.3] is 0.1
        new_score = smc.score_update(token_values, token_count, step_scores)
        
        assert new_score == pytest.approx(0.1)
        assert len(step_scores) == 3
        assert step_scores[-1] == pytest.approx(0.3)
        
        # Case where new score is the minimum
        step_scores_2 = [0.5, 0.6]
        new_score_2 = smc.score_update(token_values, token_count, step_scores_2)
        # avg still 0.3, min of [0.5, 0.6, 0.3] is 0.3
        assert new_score_2 == pytest.approx(0.3)

    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_score_update_product(self, metric):
        """Test 'product' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric=metric, aggregation="product")
        token_values = [2.0, 2.0, 2.0]
        token_count = 3
        step_scores = [0.5, -2.0] 
        
        # New step val: (2+2+2)/3 = 2.0
        # Product of [0.5, -2.0, 2.0] = -2.0
        # Return: -1 * abs(-2.0) = -2.0
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        
        assert new_score == pytest.approx(-2.0)
        assert step_scores[-1] == pytest.approx(2.0)

    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_score_update_model_aggregate(self, metric):
        """Test 'model_aggregate' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric=metric, aggregation="model_aggregate")
        token_values = [0.1, 0.2, 0.3, 0.4]
        token_count = 2
        step_scores = []
        
        # New step val (last 2): (0.3+0.4)/2 = 0.35 (appended to step_scores)
        # Return: average of ALL token_values: sum(0.1..0.4) = 1.0 / 4 = 0.25
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        
        assert new_score == pytest.approx(0.25)
        assert step_scores[-1] == pytest.approx(0.35)

    def test_score_update_entropy_last(self):
        """Test 'entropy' metric with 'last' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric="entropy", aggregation="last")
        token_values = [1.0, 2.0, 3.0]
        token_count = 3
        step_scores = []
        
        # Step score: (1+2+3)/3 = 2.0
        # Return: -1 * 2.0 = -2.0
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        assert new_score == pytest.approx(-2.0)

    def test_score_update_entropy_minimum(self):
        """Test 'entropy' metric with 'minimum' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric="entropy", aggregation="minimum")
        token_values = [3.0]
        token_count = 1
        step_scores = [1.0, 5.0]
        
        # Step score: 3.0
        # List becomes [1.0, 5.0, 3.0]
        # Return: -1 * max(step_scores) = -1 * 5.0 = -5.0
        # Note: logic in code is `-1 * max(step_scores)` for entropy minimum
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        assert new_score == pytest.approx(-5.0)

    def test_score_update_entropy_product(self):
        """Test 'entropy' metric with 'product' aggregation."""
        smc = Sequential_Monte_Carlo(token_metric="entropy", aggregation="product")
        token_values = [2.0]
        token_count = 1
        step_scores = [3.0, 4.0]
        
        # Step score: 2.0
        # List: [3.0, 4.0, 2.0]
        # Return: -1 * prod = -1 * (24.0) = -24.0
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        assert new_score == pytest.approx(-24.0)

    def test_score_update_entropy_model_aggregate(self):
        """Test 'entropy' metric with 'model_aggregate'."""
        smc = Sequential_Monte_Carlo(token_metric="entropy", aggregation="model_aggregate")
        token_values = [10.0, 20.0]
        token_count = 1
        step_scores = []
        
        # Step: 20.0
        # Return: -1 * average(all) = -1 * 15.0 = -15.0
        
        new_score = smc.score_update(token_values, token_count, step_scores)
        assert new_score == pytest.approx(-15.0)

    def test_invalid_aggregation(self):
        smc = Sequential_Monte_Carlo(token_metric="logprobs", aggregation="invalid")
        with pytest.raises(ValueError, match="Invalid aggregation method"):
            smc.score_update([1.0], 1, [])

    def test_invalid_metric(self):
        smc = Sequential_Monte_Carlo(token_metric="invalid", aggregation="last")
        with pytest.raises(ValueError, match="Invalid token metric"):
            smc.score_update([1.0], 1, [])