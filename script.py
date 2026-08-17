import os
import json
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"


# Stage 1 schema: extracted signature
class Parameter(BaseModel):
    name: str
    type: str = Field(description="Python type annotation, or 'Any' if untyped")
    default: Optional[str | int | float | bool] = Field(
            default=None,
            description="Default value if any, in its native JSON type (number, string, bool), else null",
        )


class FunctionSignature(BaseModel):
    name: str
    params: list[Parameter]
    return_type: str = Field(description="Return type annotation, or 'Any' if untyped")


# Stage 2 schema: final documentation output

class DocumentationOutput(BaseModel):
    docstring: str = Field(description="Full Google-style docstring, ready to paste under the def line")
    parameter_notes: list[str] = Field(description="One plain-English note per parameter")
    example_usage: str = Field(description="A short, runnable example call of the function")
    complexity_note: str = Field(description="One sentence on time/space complexity or key behavior")


PRICE_PER_1K_INPUT = 0.00015
PRICE_PER_1K_OUTPUT = 0.00060


def price_model(usage) -> float:
    input_cost = (usage.prompt_tokens / 1000) * PRICE_PER_1K_INPUT
    output_cost = (usage.completion_tokens / 1000) * PRICE_PER_1K_OUTPUT
    return round(input_cost + output_cost, 6)


def call_model(messages, tools=None, tool_choice=None, response_format=None):
    kwargs = {"model": MODEL, "messages": messages, "temperature": 0}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if response_format:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    return response, price_model(response.usage)


# Stage 1: extract signature via function calling

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "record_signature",
        "description": "Record the extracted signature of a Python function.",
        "parameters": FunctionSignature.model_json_schema(),
    },
}


def extract_signature(code: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise Python static-analysis assistant. Read the given "
                "code and call record_signature with its exact name, parameters "
                "(with types and defaults), and return type. Never guess -- if a "
                "type isn't annotated, use 'Any'. If there are multiple functions, "
                "extract only the first one."
            ),
        },
        {"role": "user", "content": f"```python\n{code}\n```"},
    ]
    response, cost = call_model(
        messages,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_signature"}},
    )
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return FunctionSignature(**args), cost


# Stage 2: generate docstring + metadata

FEW_SHOT_EXAMPLE = """
Example input signature: name="add", params=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}], return_type="int"
Example output:
{
  "docstring": "Add two integers.\\n\\nArgs:\\n    a (int): First addend.\\n    b (int): Second addend.\\n\\nReturns:\\n    int: The sum of a and b.",
  "parameter_notes": ["a: the first number to add", "b: the second number to add"],
  "example_usage": "result = add(2, 3)  # returns 5",
  "complexity_note": "O(1) time and space -- a single arithmetic operation."
}
""".strip()


def generate_docs(code: str, signature: FunctionSignature, max_retries: int = 2):
    system_prompt = f"""
Role: You are a senior Python developer writing documentation for a code review.

Context: You'll be given the original source code and its already-extracted signature.
Use both -- the signature for structure, the source for actual behavior.

Task: Produce a Google-style docstring plus supporting metadata for this function.

Constraints:
- docstring must follow Google style exactly (Args:, Returns:, Raises: if applicable)
- parameter_notes must have exactly one entry per parameter, plain English, no jargon
- example_usage must be a single runnable line or short snippet
- complexity_note must be one sentence, factual, no filler

Output Format: Return ONLY valid JSON matching this schema, no markdown fences, no commentary:
{DocumentationOutput.model_json_schema()}

{FEW_SHOT_EXAMPLE}
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Signature: {signature.model_dump_json()}\n\nSource code:\n```python\n{code}\n```",
        },
    ]

    last_error = None
    for _ in range(max_retries + 1):
        response, cost = call_model(messages, response_format={"type": "json_object"})
        raw = response.choices[0].message.content
        try:
            return DocumentationOutput(**json.loads(raw)), cost
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That wasn't valid JSON matching the schema ({e}). Return corrected JSON only.",
            })
    raise ValueError(f"Failed after {max_retries} retries: {last_error}")


# End-to-end pipeline

def document_code(code: str) -> dict:
    signature, extract_cost = extract_signature(code)
    docs, generate_cost = generate_docs(code, signature)
    return {
        "signature": signature.model_dump(),
        "documentation": docs.model_dump(),
        "total_cost_usd": round(extract_cost + generate_cost, 6),
    }


if __name__ == "__main__":
    test_cases = [
        '''
def calculate_discount(price, percentage=10):
    return price - (price * percentage / 100)
''',
        '''
def find_duplicates(items: list) -> list:
    seen = set()
    dupes = []
    for item in items:
        if item in seen:
            dupes.append(item)
        seen.add(item)
    return dupes
''',
    ]

    for code in test_cases:
        result = document_code(code)
        print(json.dumps(result, indent=2))
        print("\n")