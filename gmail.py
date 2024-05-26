from langchain_community.agent_toolkits import GmailToolkit

toolkit = GmailToolkit()

tools = toolkit.get_tools()
print(tools)