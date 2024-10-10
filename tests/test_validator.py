import pytest
from guardrails import Guard
from validator import GroundedaiHallucination

# Initialize the Guard with the GroundedaiHallucination validator
guard = Guard.from_string(validators=[GroundedaiHallucination(quant=False)])

def test_hallucination_pass():
    test_input = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "reference": "Paris is the capital and most populous city of France."
    }
    result = guard.parse(test_input)
    
    assert result.validation_passed is True
    assert result.validated_output == test_input

def test_hallucination_fail():
    with pytest.raises(Exception) as exc_info:
        test_input = {
            "query": "What is the capital of France?",
            "response": "The capital of France is London.",
            "reference": "Paris is the capital and most populous city of France."
        }
        guard.parse(test_input)
    
    # Assert the exception has your error_message
    assert str(exc_info.value) == "Validation failed for field with errors: {The provided input was classified as a hallucination}"

def test_no_reference():
    test_input = {
        "query": "Who was the first person to walk on the moon?",
        "response": "Neil Armstrong was the first person to walk on the moon.",
        "reference": ""
    }
    result = guard.parse(test_input)
    
    assert result.validation_passed is True
    assert result.validated_output == test_input

def test_complex_query():
    test_input = {
        "query": "Explain the process of photosynthesis.",
        "response": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
        "reference": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities."
    }
    result = guard.parse(test_input)
    
    assert result.validation_passed is True
    assert result.validated_output == test_input