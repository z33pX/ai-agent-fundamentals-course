from langfuse.model import TextPromptClient
from langfuse import Langfuse
from pydantic import BaseModel
from openai import OpenAI

import instructor
import logging
import time

l = Langfuse()


def langfuse_json_model_wrapper(
    name: str,
    system_prompt: str,
    user_prompt: str,
    prompt: TextPromptClient,
    base_model: BaseModel,
    model: str = "gpt-3.5-turbo",
    temperature=0,
    trace=None,
    observation_id=None,
) -> BaseModel:
    """
    Wrapper for the langfuse model inference.

    Args:
        name (str): The name of the inference.
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        prompt (TextPromptClient): The prompt.
        base_model (BaseModel): The base model.
        model (str): The model to use.
        temperature (int): The temperature.
        trace (Trace): The trace.
        observation_id (str): The observation id.

    Returns:
        BaseModel: The result in form of a Pydantic model.
    """
    logging.info(f"Start json inference '{name}' - model {model}")
    if trace is None:
        trace = l.trace(name=name)
    if observation_id is None:
        if trace.id is not None:
            observation_id = trace.id

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    generation = trace.generation(
        name=name,
        model=model,
        input=messages,
        metadata={
            "temperature": temperature,
            "base_model": base_model.model_json_schema(),
        },
        prompt=prompt,
    )

    client = instructor.from_openai(OpenAI())

    start = time.time()
    obj, completion = client.chat.completions.create_with_completion(
        model=model,
        response_model=base_model,
        messages=messages,
    )
    duration = time.time() - start

    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    total_tokens = completion.usage.total_tokens

    generation.end(
        output={
            # Base model is a Pydantic model, so we can dump it to JSON
            "result": obj.model_dump(),
            "duration": duration,
        },
        usage={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
            "unit": "TOKENS",
        },
    )

    trace.score(
        name="ttps",
        value=total_tokens / duration,
        comment="The number of total tokens processed per second.",
        observation_id=observation_id,
    )
    trace.score(
        name="itps",
        value=input_tokens / duration,
        comment="The number of input tokens processed per second.",
        observation_id=observation_id,
    )
    trace.score(
        name="otps",
        value=output_tokens / duration,
        comment="The number of output tokens processed per second.",
        observation_id=observation_id,
    )

    return obj
