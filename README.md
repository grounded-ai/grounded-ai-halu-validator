# Overview

| Developed by | GroundedAI |
| --- | --- |
| Date of development | October 9, 2024 |
| Validator type | Hallucination |
| Blog |  |
| License | Apache 2 |
| Input/Output | Output |

## Description

### Intended Use
This validator uses a fine-tuned language model to detect hallucinations in AI-generated responses. It evaluates whether a given response is grounded in the provided context or if it contains factually incorrect or nonsensical information.

### Requirements

* Dependencies:
	- guardrails-ai>=0.4.0
	- torch
	- transformers
	- peft
	- jinja2

* Foundation model access:
	- Internet connection to download the required models

## Installation

```bash
$ guardrails hub install hub://guardrails/grounded-ai-hallucination
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to check if an AI-generated response is a hallucination.

```python
from guardrails.hub import GroundedAIHallucination
from guardrails import Guard

guard = Guard().use(GroundedAIHallucination(quant=True))

guard.validate(json.dumps({
    "query": "What is the capital of France?",
    "response": "The capital of France is London.",
    "reference": "The capital of France is Paris."
})) 

>>> # Validator fails

guard.validate(json.dumps({
    "query": "What is the capital of France?",
    "response": "The capital of France is Paris.",
    "reference": "The capital of France is Paris."
})) 

>>> # Validator passes
```

# API Reference

**`__init__(self, quant: bool, base_prompt: Optional[str] = HALLUCINATION_EVAL_BASE)`**
<ul>
Initializes a new instance of the GroundedAIHallucination class.

**Parameters**
- **`quant`** *(bool)*: Whether to use quantization for the model.
- **`base_prompt`** *(Optional[str])*: The base prompt template for hallucination evaluation. Defaults to HALLUCINATION_EVAL_BASE, but you may customize this to your need.
- **`on_fail`** *(str, Callable)*: The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.
</ul>
<br/>

**`validate(self, value: str, metadata: Dict = {}) -> ValidationResult`**
<ul>
Validates whether the given response is a hallucination based on the provided query and reference.

**Parameters**
- **`value`** *(str)*: A string (dictionary wrapped with json.dumps()) containing the following keys:
  - `query` (str): The original question or prompt.
  - `response` (str): The AI-generated response to validate.
  - `reference` (str): Optional reference information for fact-checking.
- **`metadata`** *(dict)*: Additional metadata (not used in this validator).

**Returns**
- **`ValidationResult`**: Indicates whether the validation passed or failed.
</ul>