from utils.langfuse_model_wrapper import langfuse_model_wrapper
from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from eezo.interface import Context
from nylas import Client as Nylas
from typing import Type, Dict, Any
from langfuse import Langfuse
from bs4 import BeautifulSoup
from eezo import Eezo

import os

nylas = Nylas(os.getenv("NYLAS_API_KEY"), os.getenv("NYLAS_API_URL"))
l = Langfuse()
e = Eezo()

agent = e.get_agent(os.environ["TOOL_SUMMARIZE_LAST_X_EMAILS"])
summarize_emails = l.get_prompt("summarize-emails")


class SummarizeLastXEmails(BaseTool):
    name = agent.agent_id
    description = agent.description
    args_schema: Type[BaseModel] = agent.input_model
    state: Dict[str, Any] | None

    def __init__(
        self,
        state: Dict[str, Any] | None = None,
    ):
        super().__init__()
        if state is None:
            self.state = {}
        else:
            self.state = state

    def _run(self, **kwargs) -> str | None:
        nr_of_emails = kwargs.get("nr_of_emails", 3)
        if "processed_emails" not in self.state:
            self.state["processed_emails"] = []

        response = nylas.messages.list(
            os.getenv("NYLAS_GRANT"), query_params={"limit": nr_of_emails}
        )

        emails = []

        for message in response[0]:
            if message.id not in self.state["processed_emails"]:
                body = BeautifulSoup(message.body, "html.parser")
                text = body.get_text()
                cleaned_text = "\n".join(
                    [line for line in text.splitlines() if line.strip()]
                )

                parsed_from = "\n".join(
                    [
                        f"{item.get('name', '-')}: {item.get('email', '-')}"
                        for item in message.from_
                    ]
                )

                emails.append(f"{parsed_from}\n{cleaned_text}")
                self.state["processed_emails"].append(message.id)

        if not emails:
            return "No emails to summarize."

        system_prompt = summarize_emails.compile(
            emails="\n\n".join(emails), user_prompt=kwargs.get("query", "")
        )

        email_summary = langfuse_model_wrapper(
            name="SummarizeSearchResults",
            system_prompt=system_prompt,
            prompt=summarize_emails,
            user_prompt="Summarize the email for the CEO.",
            model="llama3-70b-8192",
            host="groq",
            temperature=0.7,
        )

        return email_summary
