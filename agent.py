from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage
from gmail import tools

# add the tools from gmail.py to the list of tools.
tools = [*tools]

# the llm, the 'brain' of our agent that picks the tool.
# We will be using the gpt-4o model from OpenAI
model = ChatOpenAI(model="gpt-4o")


# create the agent executor, combing the model and the tools.
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

# invoke the agent executor to start using it.
response = agent_executor.invoke({"messages": [HumanMessage(content="""My name is 'The AI Dev'. 
    Can you create an email draft for me
    asking if Lisa is a available for a quick call on Monday? 
    Please use a dummy email address for Lisa,
    write a subject line you see fit.
    Under no circumstances may you send the email.""")]})

print(response["messages"])