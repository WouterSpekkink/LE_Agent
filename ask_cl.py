from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import OpenAICallbackHandler
from langchain.agents import AgentOutputParser
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain import LLMChain
from datetime import datetime
import chainlit as cl
import os
import sys
import constants
import openai
import json

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# Cleanup function for source strings
def string_cleanup(string):
  """A function to clean up strings in the sources from unwanted symbols"""
  return string.replace("{","").replace("}","").replace("\\","").replace("/","")

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Load FAISS databases
embeddings = OpenAIEmbeddings()
conceptual = FAISS.load_local("./conceptual_vectorstore/", embeddings)
empirical = FAISS.load_local("./empirical_vectorstore/", embeddings)

# Set up callback handler
handler = OpenAICallbackHandler()

# # Set up source file
# now = datetime.now()
# timestamp = now.strftime("%Y%m%d_%H%M%S")
# filename = f"answers/answers_{timestamp}.txt"
# with open(filename, 'w') as file:
#   file.write(f"Answers and sources for session started on {timestamp}\n\n")

@cl.on_chat_start
async def main():
  # Set llm
  llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
  

  tool_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

  system_prompt_template = (
  '''
  You are a knowledgeable professor working in academia.
  Using the provided pieces of context, you answer the questions asked by the user.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  """
  Context: {context}
  """

  Please try to give detailed answers and write your answers as an academic text, unless explicitly told otherwise.
  Use references to literature in your answer and include a bibliography for citations that you use.
  If you cannot provide appropriate references, tell me by the end of your answer.
 
  One or multiple sentences that constitutes part of your answer (APA-style reference)
  The rest of your answer
  Bibliography:
  Bulleted bibliographical entries in APA-style
  ''')
  
  system_prompt = PromptTemplate(template=system_prompt_template,
                                 input_variables=["context"])

  system_message_prompt = SystemMessagePromptTemplate(prompt = system_prompt)
  human_template = "{question}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  conceptual_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=conceptual.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': chat_prompt},
    memory=tool_memory,
  )

  system_prompt_template = (
    '''You help me to extract relevant information from a case description from news items.
    The context includes extracts from relevant new items in Dutch and English.
    You help me by answering questions about the topic I wish to write a case description on.
    Yoy also help me to write parts of my case description of I ask you to do so. 
    
    If the context doesn't provide a satisfactory answer, just tell me that and don't try to make something up.
    Please try to give detailed answers and write your answers as an academic text, unless explicitly told otherwise.

    """
    Context: {context}
    Question: {question}
    """
    If possible, consider sources from both Dutch and English language sources.
    ''')
  # Customize prompt
  system_prompt_template = (
    '''You help me to extract relevant information from a case description from news items.
    The context includes extracts from relevant new items in Dutch and English.
    You help me by answering questions about the topic I wish to write a case description on.
    Yoy also help me to write parts of my case description of I ask you to do so. 
    
    If the context doesn't provide a satisfactory answer, just tell me that and don't try to make something up.
    Please try to give detailed answers and write your answers as an academic text, unless explicitly told otherwise.
    
    """
    Context: {context}
    """

    If possible, consider sources from both Dutch and English language sources.
    ''')

  system_prompt = PromptTemplate(template=system_prompt_template,
                                 input_variables=["context"])

  system_message_prompt = SystemMessagePromptTemplate(prompt = system_prompt)
  human_template = "{question}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
  
  # Set up conversational chain
  empirical_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=empirical.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': chat_prompt},
    memory=tool_memory,
  )

  system_prompt_template = (
    '''You help me write academic texts of a length specified by the user. 
    For this, you consider the provided context and you write a coherent, essay-like text about this.

    """
    Context: {input}
    """

    ''')

  system_prompt = PromptTemplate(template=system_prompt_template,
                                 input_variables=["input"])

  writing_chain = LLMChain(
    llm=llm,
    prompt=system_prompt,
    memory=tool_memory,
  )

  # Add chains to toolbox.
  tools = [
    Tool(
      name="Conceptual tool",
      func=conceptual_chain.run,
      description="""Useful for when you need to understand a concept or theory.
      The input should be a fully formed question that also includes the full context of the conversation before.
      The input should also include the full context from the conversation before.
      """,
      ),
    Tool(
      name="Empirical tool",
      func=empirical_chain.run,
      description="""Useful for when you need empirical information on a topic.
      The input should be a fully formed question including context for the question.
      If you ask this tool to illustrate a concept, the input must include the full context of the conversation before.
      This context can be an exact copy of the output of the conceptual tool.
      """,
      ),
    Tool(
      name="Writing tool",
      func=writing_chain.run,
      description="""Useful for when you need to output texts based on input from the empirical and conceptual tool.
      The input should be a fully formed question that also includes the full context of the conversation before.
      This tool requires that other tools have been used to generate input for the writing task.
      """,
      return_direct=True,
      ),
    ]

  agent_memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", output_key="output", return_messages=True)

  # Set up agent
  agent = initialize_agent(tools,
                           llm=llm,
                           agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                           verbose=True,
                           memory=tool_memory,
                           )

  cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: str):
  agent = cl.user_session.get("agent")
  cb = cl.LangchainCallbackHandler(stream_final_answer=True)
  res = await cl.make_async(agent.run)(message, callbacks=[cb])
  await cl.Message(content=res).send()
