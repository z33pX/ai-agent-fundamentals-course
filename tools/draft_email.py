from utils.langfuse_model_wrapper import langfuse_model_wrapper
from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from eezo.interface import Interface
from langfuse import Langfuse
from typing import Type
from eezo import Eezo

import os

l = Langfuse()
e = Eezo()

agent = e.get_agent(os.environ["TOOL_DRAFT_EMAIL"])
draft_email = l.get_prompt("draft-email")


class DraftEmail(BaseTool):
    name = agent.name
    description = agent.description
    args_schema: Type[BaseModel] = agent.input_model
    eezo_interface: Interface | None

    def __init__(self, eezo_interface: Interface | None = None):
        super().__init__()
        self.eezo_interface = eezo_interface

    def _run(self, **kwargs) -> str:
        chat_history_str = "No chat history available."
        if self.eezo_interface:
            chat_history_str = self.eezo_interface.get_thread(to_string=True)

        system_prompt = draft_email.compile(
            chat_history=chat_history_str, user_prompt=kwargs["query"]
        )

        email_draft = langfuse_model_wrapper(
            name="DraftEmail",
            system_prompt=system_prompt,
            prompt=draft_email,
            user_prompt=kwargs["query"],
            model="llama3-70b-8192",
            host="groq",
            temperature=0.7,
        )

        return email_draft
