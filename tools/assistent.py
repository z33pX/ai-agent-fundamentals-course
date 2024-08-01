from utils.langfuse_model_wrapper import langfuse_model_wrapper
from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from eezo.interface import Context
from langfuse import Langfuse
from typing import Type
from eezo import Eezo

import os

l = Langfuse()
e = Eezo()

agent = e.get_agent(os.environ["TOOL_ASSISTENT"])
assistent = l.get_prompt("assistent")


class Assistent(BaseTool):
    name = agent.agent_id
    description = agent.description
    args_schema: Type[BaseModel] = agent.input_model
    eezo_interface: Context | None

    def __init__(self, eezo_interface: Context | None = None):
        super().__init__()
        self.eezo_interface = eezo_interface

    def _run(self, **kwargs):
        chat_history_str = ""
        if self.eezo_interface:
            chat_history_str = self.eezo_interface.get_thread(to_string=True)

        system_prompt = assistent.compile(
            chat_history=chat_history_str, user_prompt=kwargs["query"]
        )

        answer = langfuse_model_wrapper(
            name="Assistent",
            system_prompt=system_prompt,
            prompt=assistent,
            user_prompt=kwargs["query"],
            model="llama3-70b-8192",
            host="groq",
            temperature=0.7,
        )

        return answer
