from langchain_openai import ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import dotenv

dotenv.load_dotenv()

from tools.research.similar_web_search import SimilarWebSearch
from tools.research.exa_company_search import ExaCompanySearch
from tools.research.you_com_search import YouComSearch
from tools.research.news_search import NewsSearch


# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Initialize the tools
tools = [YouComSearch(), SimilarWebSearch(), ExaCompanySearch(), NewsSearch()]

# Define the tool node
tool_node = ToolNode(tools)

# Bind the tools to the model
model = llm.bind_tools(tools)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
research_agent = workflow.compile(checkpointer=checkpointer)
