import dotenv
import logging

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

from langchain_core.messages import HumanMessage
from eezo import Eezo

import os


e = Eezo()

# Research Agent V1 ==============================================
# from agents.research_v1 import research_agent


# @e.on(os.environ["AGENT_RESEARCH"])
# def research_agent_v1(context, **kwargs):
#     m = context.new_message()
#     c = m.add("text", text="Researching...")
#     m.notify()

#     final_state = research_agent.invoke(
#         {"messages": [HumanMessage(content=kwargs["query"])]},
#         config={"configurable": {"thread_id": 42}},
#     )
#     result = final_state["messages"][-1].content

#     m.replace(c.id, "text", text=result)
#     m.notify()


# e.connect()


# Research Agent V2 ==============================================
from agents.research_v2 import ResearchAgent
from tools import *

tools = [YouComSearch(), SimilarWebSearch(), ExaCompanySearch(), NewsSearch()]
# Accepts only ResearchTools. ResearchTools support the necessary API
research_agent = ResearchAgent(tools)


@e.on(os.environ["AGENT_RESEARCH"])
def research_agent_handler(context, **kwargs):
    research_agent.invoke(context, **kwargs)


e.connect()

exit()

# Tools ==========================================================


@e.on(os.environ["TOOL_GENERATE_UI"])
def text_to_chart(context, **kwargs):
    m = context.new_message()
    m.add("text", text="Generate chart...")
    m.notify()

    # TODO: Problem is that the code in the sandbox doesn't have access to the self.message object
    thread = context.get_thread(to_string=True)

    gen_ui = GenerateUI(m, thread)
    gen_ui.invoke(input={"query": kwargs["query"]})
    m.notify()


# Works
@e.on(os.environ["TOOL_SEARCH_VECTOR_DB"])
def research_result_search(context, **kwargs):
    m = context.new_message()
    c = m.add("text", text="Searching past results...")
    m.notify()

    summary = SearchVectorDB().invoke(kwargs)

    m.replace(c.id, "text", text=summary)
    m.notify()


# Works
@e.on(os.environ["TOOL_YOU_COM_SEARCH"])
def search_engine(context, **kwargs):
    m = context.new_message()
    c = m.add("text", text="Searching...")
    m.notify()

    result = YouComSearch(include_summary=True).invoke(**kwargs)

    m.replace(c.id, "text", text=result.summary)
    m.notify()


# Works
@e.on(os.environ["TOOL_ASSISTENT"])
def assistent(context, **kwargs):
    m = context.new_message()
    c = m.add("text", text="Thinking...")
    m.notify()

    answer = Assistent(context).invoke(kwargs)

    m.replace(c.id, "text", text=answer)
    m.notify()


# Works
@e.on(os.environ["TOOL_EXA_COMPANY_SEARCH"])
def company_search(context, **kwargs):
    m = context.new_message()
    c = m.add("text", text="Searching for companies...")
    m.notify()

    result = ExaCompanySearch(include_summary=True).invoke(**kwargs)

    m.replace(c.id, "text", text=f"Here are the top {len(result.content)} results:")
    for content_item in result.content:
        m.add(
            "text",
            text=f"**-** [{content_item.title}]({content_item.url}) - {content_item.snippet}",
        )

    m.add("text", text=result.summary)
    m.notify()


# Works
@e.on(os.environ["TOOL_SUMMARIZE_LAST_X_EMAILS"])
def summarize_emails(context, **kwargs):
    if context:
        m = context.new_message()
    else:
        m = e.new_message(
            eezo_id=os.environ["EEZO_EEZO_ID"],
            thread_id=os.environ["THREAD_ID"],
            context="summarize_emails",
        )

    m.add("text", text="Summarizing emails...")
    m.notify()

    if context:
        email_summary = SummarizeLastXEmails().invoke(input=kwargs)
    else:
        # Use the state in cron job mode
        e.load_state()
        email_summary = SummarizeLastXEmails(e.state).invoke(input=kwargs)
        e.save_state()

    m.add("text", text=f"{email_summary}")
    m.notify()


# Works
@e.on(os.environ["TOOL_NEWS_SEARCH"])
def news_search(context, **kwargs):
    if context:
        m = context.new_message()
        m.add("text", text="Searching news...")
        m.notify()
    else:
        m = e.new_message(
            eezo_id=os.environ["EEZO_EEZO_ID"],
            thread_id=os.environ["THREAD_ID"],
            context="news_search",
        )

    result = NewsSearch(include_summary=True).invoke(**kwargs)

    m.add("text", text=f"Here are the top {len(result.content)} results:")

    for content_item in result.content:
        m.add(
            "text",
            text=f"**-** [{content_item.title}]({content_item.url}) - {content_item.snippet}",
        )

    m.add("text", text=result.summary)
    m.notify()


# Works (Except for GenerateUI)
@e.on(os.environ["TOOL_SIMILAR_WEB_SEARCH"])
def web_search(context, **kwargs):
    if context:
        m = context.new_message()
        m.add("text", text="Searching the web...")
        m.notify()
    else:
        m = e.new_message(
            eezo_id=os.environ["EEZO_EEZO_ID"],
            thread_id=os.environ["THREAD_ID"],
            context="web_search",
        )

    # include_summary not necessary. Only burnes additional tokens
    result = SimilarWebSearch(
        user_prompt=kwargs["query"], include_summary=False
    ).invoke(**kwargs)

    formatted_content = "\n".join(
        [
            f"**-** [{content_item.title}]({content_item.url}) - {content_item.content}"
            for content_item in result.content
        ]
    )

    gen_ui = GenerateUI(m, formatted_content)
    gen_ui.invoke(
        input={"query": f"Generate a UI that plots all numbers from the given text"}
    )

    m.notify()


# Cron Jobs =====================================================
# from utils.cron_manager import CronManager

# Works
# cm = CronManager()
# cm.add_task(
#     func=summarize_emails, payload={"context": None, "nr_of_emails": 3}, interval=5
# )
# cm.add_task(
#     func=news_search, payload={"context": None, "query": "Elon Musk"}, interval=60 * 6
# )
# cm.run()

# Connect to the Eezo API =======================================
e.connect()
