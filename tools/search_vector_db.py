from utils.langfuse_model_wrapper import langfuse_model_wrapper
from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from eezo.interface import Interface
from langfuse import Langfuse
from pinecone import Pinecone
from typing import Type
from eezo import Eezo

import openai
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
l = Langfuse()
e = Eezo()

agent = e.get_agent(os.environ["TOOL_SEARCH_VECTOR_DB"])

summarize_search_results = l.get_prompt("summarize-search-results")


class SearchVectorDB(BaseTool):
    name = agent.name
    description = agent.description
    args_schema: Type[BaseModel] = agent.input_model

    def __init__(self):
        super().__init__()

    def _run(self, **kwargs):
        embedding = (
            openai.Client()
            .embeddings.create(input=[kwargs["query"]], model="text-embedding-3-small")
            .data[0]
            .embedding
        )
        index = pc.Index("research-agent")
        result = index.query(vector=embedding, top_k=3, include_metadata=True)

        formatted_results = ""
        for result in result["matches"]:
            formatted_results += f"{result['metadata']['text']}\n\n"

        system_prompt = summarize_search_results.compile(
            search_results_str=formatted_results, user_prompt=kwargs["query"]
        )

        return langfuse_model_wrapper(
            name="DraftEmail",
            system_prompt=system_prompt,
            prompt=summarize_search_results,
            user_prompt=kwargs["query"],
            model="llama3-70b-8192",
            host="groq",
            temperature=0.7,
        )
