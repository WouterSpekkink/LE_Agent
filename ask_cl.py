from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import OpenAICallbackHandler
from langchain import ConversationChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.document_transformers import LongContextReorder, EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
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
from datetime import datetime
from dotenv import load_dotenv
import chainlit as cl
import os
import sys
import constants
import openai

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

# Set up source file
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"answers/answers_{timestamp}.txt"
with open(filename, 'w') as file:
  file.write(f"Answers and sources for session started on {timestamp}\n\n")

@cl.on_chat_start
async def main():
  # Set llm
  llm = ChatOpenAI(model="gpt-3.5-turbo-16k", streaming=True)
  
  # Set up shared memory
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  readonlymemory = ReadOnlySharedMemory(memory=memory)
  
  # Set up conceptual chain prompt
  conceptual_system_prompt_template = (
  '''
  You are a knowledgeable professor working in academia, focused on clearly explaining concepts.
  Using the provided pieces of context, you answer the questions asked by the user.

  """
  Context: {context}
  """

  Please try to give detailed answers and write your answers as an academic text, unless explicitly told otherwise.
  Use references to literature in your answer and include a bibliography for citations that you use.
  If you cannot provide appropriate references, tell me by the end of your answer.
  If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.
 
  Formatting instructions:
  One or multiple sentences that constitutes part of your answer (APA-style reference)
  The rest of your answer
  Bibliography:
  Bulleted bibliographical entries in APA-style
  ''')
  
  conceptual_system_prompt = PromptTemplate(template=conceptual_system_prompt_template,
                                 input_variables=["context"])

  conceptual_system_message_prompt = SystemMessagePromptTemplate(prompt = conceptual_system_prompt)
  conceptual_human_template = "{question}"
  conceptual_human_message_prompt = HumanMessagePromptTemplate.from_template(conceptual_human_template)
  conceptual_chat_prompt = ChatPromptTemplate.from_messages([conceptual_system_message_prompt, conceptual_human_message_prompt])

  # Set up conceptual chain reordering
  redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
  reordering = LongContextReorder()
  pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, reordering])
  conceptual_compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=conceptual.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k" : 20, "score_threshold": .65}))

  # Initialize chain
  conceptual_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=conceptual_compression_retriever_reordered,
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': conceptual_chat_prompt},
    memory=readonlymemory,
    return_source_documents=True,
  )

  # Wrap chain
  def run_conceptual_chain(question):
    results = conceptual_chain({"question": question}, return_only_outputs=True)
    sources = results['source_documents']
    counter = 1
    for source in sources:
      with open(filename, 'a') as file:
        file.write(f"Tool: Conceptual tool_{counter}\n\n")
        file.write("Query:\n")
        file.write(question)
        file.write("\n\n")
        file.write("Document: ")
        file.write(os.path.basename(source.metadata['source']))
        file.write("\n\n")
        file.write("Content:\n")
        file.write(source.page_content.replace("\n", " "))
        file.write("\n\n")
      counter += 1
    return str(results['answer'])

  # Set up empirical chain prompt
  empirical_system_prompt_template = (
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

  empirical_system_prompt_template = (
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

  empirical_system_prompt = PromptTemplate(template=empirical_system_prompt_template,
                                 input_variables=["context"])

  empirical_system_message_prompt = SystemMessagePromptTemplate(prompt = empirical_system_prompt)
  empirical_human_template = "{question}"
  empirical_human_message_prompt = HumanMessagePromptTemplate.from_template(empirical_human_template)
  empirical_chat_prompt = ChatPromptTemplate.from_messages([empirical_system_message_prompt, empirical_human_message_prompt])

  # Setup empirical reorder
  empirical_compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=empirical.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k" : 20, "score_threshold": .65}))

  # Initialize empirical chain
  empirical_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=empirical_compression_retriever_reordered,
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': empirical_chat_prompt},
    memory=readonlymemory,
    return_source_documents = True,
  )

  # Wrap empirical chain
  def run_empirical_chain(question):
    results = empirical_chain({"question": question}, return_only_outputs=True)
    sources = results['source_documents']
    counter = 1
    for source in sources:
      with open(filename, 'a') as file:
        file.write(f"Tool: Empirical tool_{counter}\n\n")
        file.write("Query:\n")
        file.write(question)
        file.write("\n\n")
        file.write("Document: ")
        file.write(source.metadata['source'])
        file.write("\n\n")
        file.write("Content:\n")
        file.write(source.page_content.replace("\n", " "))
        file.write("\n\n")
      counter += 1
    return str(results['answer'])

  # Set up writing chain prompt
  writing_system_prompt_template = (
    '''You help me write academic texts of a length specified by the user. 
    For this, you consider the input, chat history and you write a coherent, essay-like text based on this information.
    Not everything in the chat history may be relevant.

    """
    Chat history: {chat_history}
    """
    """
    Input: {input}
    """

    ''')

  writing_system_prompt = PromptTemplate(template=writing_system_prompt_template,
                                 input_variables=["chat_history", "input"],
                                 )

  # Initialize writing chain
  writing_chain = ConversationChain(
    llm=llm,
    prompt=writing_system_prompt,
    memory=readonlymemory,
  )

  # Add chains to toolbox.
  tools = [
    Tool(
      name="Conceptual tool",
      func=run_conceptual_chain,
      description="""Useful for when you need to understand a concept or theory.
      The input should be a fully formed question, not referencing any obscure pronouns from the conversation before.
      The question should end with a question mark.
      """,
      return_direct=True,
      ),
    Tool(
      name="Empirical tool",
      func=run_empirical_chain,
      description="""Useful for when you need empirical information on a topic.
      The input should be a fully formed question, not referencing any obscure pronouns from the conversation before.
      The question should end with a question mark.
      The input should be a fully formed question. 
;      """,
      return_direct=True,
      ),
    Tool(
      name="Writing tool",
      func=writing_chain.run,
      description="""Useful for when you need to output texts based on input from the empirical and conceptual tool.
      The input should be a fully formed question, not referencing any obscure pronouns from the conversation before.
      The question should end with a question mark.
      """,
      return_direct=True,
      ),
    ]

  # Set up agent
  agent = initialize_agent(tools,
                           llm=llm,
                           agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                           verbose=True,
                           memory=memory,
                           handle_parsing_errors=True,
                           )

  cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: str):
  agent = cl.user_session.get("agent")
  cb = cl.LangchainCallbackHandler()
  res = await cl.make_async(agent.run)(message, callbacks=[cb])
  await cl.Message(content=res).send()
