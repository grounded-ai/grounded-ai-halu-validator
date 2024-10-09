from typing import Dict

import torch
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from jinja2 import Template
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

HALLUCINATION_EVAL_BASE = """
       {% set knowledge_line = "" if reference == "" else "[Knowledge]: " + reference + "\n        " %}
        Your job is to evaluate whether a machine learning model has hallucinated or not.
        A hallucination occurs when the response is coherent but factually incorrect or nonsensical
        outputs that are not grounded in the provided context.
        You are given the following information:
        ####INFO####
        {{ knowledge_line }}[User Input]: {{ query }}
        [Model Response]: {{ response }}
        ####END INFO####
        Based on the information provided is the model output a hallucination? Respond with only "yes" or "no"
"""


@register_validator(name="guardrails/grounded-ai-hallucination", data_type="string")
class ValidatorTemplate(Validator):
    """Validates that {fill in how you validator interacts with the passed value}.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/validator_template`   |
    | Supported data types          | `string`                          |
    | Programmatic fix              | `None`                            |

    Args:
        arg_1 (string): {Description of the argument here}
        arg_2 (string): {Description of the argument here}
    """  # noqa

    BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
    GROUNDEDAI_EVAL_ID = "grounded-ai/phi3.5-hallucination-judge"

    def __init__(
        self,
        quant: bool,
        base_prompt: str,
    ):
        super().__init__(quant, base_prompt)
        self._quantize = quant
        self._base_prompt = base_prompt

        self.warmup()

    def load_model(self):
        """Loads the base model with or without quantization."""
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        attn_implementation = (
            "flash_attention_2" if torch.cuda.is_bf16_supported() else "sdpa"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_ID)
        model_kwargs = {
            "attn_implementation": attn_implementation,
            "torch_dtype": compute_dtype,
        }
        if self.quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_ID, **model_kwargs
        )

        self.base_model = base_model
        self.tokenizer = tokenizer

    def merge_adapter(self, groundedai_eval_id: str):
        """Merges the PEFT adapter into the base model."""
        # TODO Error handling for adapter merging could be added here
        config = PeftConfig.from_pretrained(groundedai_eval_id)
        model_peft = PeftModel.from_pretrained(
            self.base_model, groundedai_eval_id, config=config
        )
        self.merged_model = model_peft.merge_and_unload()
        if not self.quantization:
            self.merged_model.to("cuda")

    def warmup(self):
        """Warmup the model by loading it and merging the adapter"""
        self.load_model()
        self.merge_adapter(self.GROUNDEDAI_EVAL_ID)

    def format_input(self, query: str, response: str, reference: str = None) -> str:
        template = Template(HALLUCINATION_EVAL_BASE)
        rendered_prompt = template.render(
            reference=reference, query=query, response=response
        )
        return rendered_prompt

    def run_model(self, query: str, response: str, reference: str = "") -> str:
        input = self.format_input(query, response, reference)
        messages = [{"role": "user", "content": input}]

        pipe = pipeline(
            "text-generation",
            model=self.merged_model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 2,
            "return_full_text": False,
            "temperature": 0.01,
            "do_sample": True,
        }

        output = pipe(messages, **generation_args)
        torch.cuda.empty_cache()
        return output[0]["generated_text"].strip().lower()

    def validate(self, value: Dict, metadata: Dict = {}) -> ValidationResult:
        """Validates that {fill in how you validator interacts with the passed value}."""
        hallucination = self.run_model(
            value["query"], value["response"], value["reference"]
        )
        if "yes" in hallucination:  # FIXME
            return FailResult(
                error_message="{The provided input was classified as a hallucination}",
            )
        return PassResult()
