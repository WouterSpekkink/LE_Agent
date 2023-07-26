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
  #llm = OpenAI(temperature=0)
  
  # # Set up the conceptual chain
  # conceptual_chain = RetrievalQA.from_chain_type(
  #   llm=llm,
  #   chain_type="stuff",
  #   retriever=conceptual.as_retriever(search_type="mmr", search_kwargs={"k" : 10}))
  
  # cl.user_session.set("conceptual_chain", conceptual_chain)

  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
 
  Format your answer as follows:
  [One or multiple sentences that constitutes part of your answer (APA-style reference)]
  [The rest of your answer]
  [Bibliography:]
  [Bulleted bibliographical entries in APA-style]
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
    memory=memory,
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
    """

    If possible, consider sources from both Dutch and English language sources.
    ''')

  system_prompt = PromptTemplate(template=system_prompt_template,
                                 input_variables=["context"])

  system_message_prompt = SystemMessagePromptTemplate(prompt = system_prompt)
  human_template = "{question}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
  
  # Set up the empirical chain
  # empirical_chain = RetrievalQA.from_chain_type(
  #   llm=llm,
  #   chain_type="stuff",
  #   retriever=empirical.as_retriever(search_type="mmr", search_kwargs={"k" : 10}))
  
  # cl.user_session.set("empirical_chain", empirical_chain)
  empirical_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=empirical.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': chat_prompt},
    memory=memory,
  )

  # Add chains to toolbox.
  tools = [
    Tool(
      name="Conceptual tool",
      func=conceptual_chain.run,
      description="useful for when you need to understand a concept or theory.",
      ),
    Tool(
      name="Empirical tool",
      func=empirical_chain.run,
      description="useful for when you need empirical information."
      ),
    ]

  # Set up agent
  agent = initialize_agent(tools,
                           llm=llm,
                           agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                           memory=memory,
                           )

  cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: str):
  agent = cl.user_session.get("agent")
  cb = cl.LangchainCallbackHandler(stream_final_answer=True)
  res = await cl.make_async(agent.run)(message, callbacks=[cb])
  await cl.Message(content=res).send()
