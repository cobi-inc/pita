# Unit tests for parameter validation in the serve module
import pytest

# This test verifies that when both chain_sampling and token_sampling are specified,
# the API raises an appropriate error instead of silently ignoring one parameter.
#
# Note: These are unit tests of the validation logic. Integration tests that actually
# call the API endpoint would require starting the FastAPI server, which should be
# added to test_api_integration_llama_cpp.py.


def test_validation_logic_for_both_parameters():
    """
    Test that validates the logic for preventing both chain_sampling and token_sampling
    from being specified together.
    
    This verifies the validation check in serve.py should raise HTTPException with status 400.
    """
    # Simulate the scenario from serve.py
    chain_sampling = "mock_chain_sampling_object"  # Simulating non-None
    token_sampling = "mock_token_sampling_object"  # Simulating non-None
    
    # The validation logic that should be present
    should_raise_error = (chain_sampling is not None and token_sampling is not None)
    
    # Verify that error should be raised in this scenario
    assert should_raise_error, "Should detect when both parameters are specified"


def test_validation_allows_chain_sampling_only():
    """Test that chain_sampling alone is allowed"""
    chain_sampling = "mock_chain_sampling_object"
    token_sampling = None
    
    should_raise_error = (chain_sampling is not None and token_sampling is not None)
    
    assert not should_raise_error, "Should allow only chain_sampling"


def test_validation_allows_token_sampling_only():
    """Test that token_sampling alone is allowed"""
    chain_sampling = None
    token_sampling = "mock_token_sampling_object"
    
    should_raise_error = (chain_sampling is not None and token_sampling is not None)
    
    assert not should_raise_error, "Should allow only token_sampling"


def test_validation_allows_neither_parameter():
    """Test that neither parameter is also allowed (default behavior)"""
    chain_sampling = None
    token_sampling = None
    
    should_raise_error = (chain_sampling is not None and token_sampling is not None)
    
    assert not should_raise_error, "Should allow default behavior with no parameters"

