from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import OpenAICallbackHandler
from datetime import datetime
import chainlit as cl
import os
import sys
import constants
import openai

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI

# Cleanup function for source strings
def string_cleanup(string):
  """A function to clean up strings in the sources from unwanted symbols"""
  return string.replace("{","").replace("}","").replace("\\","").replace("/","")

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Load FAISS database
embeddings = OpenAIEmbeddings()
conceptual = FAISS.load_local("./conceptual_vectorstore/", embeddings)
empirical = FAISS.load_local("./empirical_vectorstore/", embeddings)

# Set up callback handler
handler = OpenAICallbackHandler()

# Set up source file
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"answers/answers_{timestamp}.txt"
with open(filename, 'w') as file:
  file.write(f"Answers and sources for session started on {timestamp}\n\n")

@cl.on_chat_start
async def main():
  # Set llm
#  llm = OpenAI(model="gpt-3.5-turbo")
  llm = OpenAI(temperature=0)
  
  conceptual_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=conceptual.as_retriever(search_type="mmr", search_kwargs={"k" : 10}))
  
  cl.user_session.set("conceptual_chain", conceptual_chain)

  empirical_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=empirical.as_retriever(search_type="mmr", search_kwargs={"k" : 10}))
  
  cl.user_session.set("empirical_chain", empirical_chain)

  tools = [
    Tool(
      name="Conceptual tool",
      func=conceptual_chain.run,
      description="useful for when you need to understand a concept. Input should be a fully formed question.",
      ),
    Tool(
      name="Empirical tool",
      func=empirical_chain.run,
      description="useful for when you need to find empirical information."
      ),
    ]
  
  agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
  )

  cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: str):
  agent = cl.user_session.get("agent")
  cb = cl.AsyncLangchainCallbackHandler(
       stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
  cb.answer_reached = True
  res = agent.run(message, callbacks=[cb])
  await cl.Message(content=res).send()
