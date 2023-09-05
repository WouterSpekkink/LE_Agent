from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import OpenAICallbackHandler
from langchain import ConversationChain
from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.document_transformers import LongContextReorder, EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.utilities import GoogleSerperAPIWrapper
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
from chainlit.input_widget import Select, Slider
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
os.environ["OPENAI_API_KEY"] = constants.OPENAIKEY
openai.api_key = constants.OPENAIKEY 

# Set Serper API Key
os.environ["SERPER_API_KEY"] = constants.SERPERKEY

# Load conceptual FAISS databases
embeddings = OpenAIEmbeddings()
conceptual_all = FAISS.load_local("./conceptual_vectorstore/Literature_All/", embeddings)
conceptual_dics = FAISS.load_local("./conceptual_vectorstore/Literature_DiCS/", embeddings)
conceptual_sturing = FAISS.load_local("./conceptual_vectorstore/Literature_Sturing/", embeddings)
conceptual_ng = FAISS.load_local("./conceptual_vectorstore/Literature_Network_Governance/", embeddings)

# Load empirical FAISS databases
empirical_ld = FAISS.load_local("./empirical_vectorstore/Empirical_Loss_and_Damage/", embeddings)
empirical_nitrogen = FAISS.load_local("./empirical_vectorstore/Empirical_Nitrogen/", embeddings)
empirical_inland_shipping = FAISS.load_local("./empirical_vectorstore/Empirical_Inland_Shipping/", embeddings)
empirical_samso = FAISS.load_local("./empirical_vectorstore/Empirical_Samso/", embeddings)
empirical_jsf = FAISS.load_local("./empirical_vectorstore/Empirical_JSF/", embeddings)

# Set up callback handler
handler = OpenAICallbackHandler()

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

# Set up writing chain prompt
writing_system_prompt_template = (
  '''You help me write academic texts of a length specified by the user. 
  For this, you consider the input and chat history and you write a coherent, essay-like text based on this information.
  
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

# Set up writing chain prompt
critical_system_prompt_template = (
  '''You are a critical academic commentator.
  You consider the input and chat history and you consider the arguments made in it and offer a critique of them.
  When offering a critique, you also offer suggestions for improvement of the argument.
  
  """
  Chat history: {chat_history}
  """
  """
  Input: {input}
  """

  ''')

critical_system_prompt = PromptTemplate(template=critical_system_prompt_template,
                                        input_variables=["chat_history", "input"],
                                        )
# Customize prompt
mc_system_prompt_template = (
  '''You are a university professor in the field of governance in the public sector.
  You are also a test expert specialized in making multiple choice exams.
  You make multiple choice exam questions based on the chat history and the input provided below.
  
  The user will ask to formulate a question about a certain topic, so make sure the answer is about that topic.
  
  """
  Chat history: {chat_history}
  """
  """
  Input: {input}
  """
  
  The multiple choice questions should adhere to a couple of rules:
  - Questions should always end with a question mark
  - Questions should not refer to the answer options, because they should be answerable without reading the answer options. For example, a question that includes the phrase "which of the following" is not allowed.
  - There should always be four answer options, of which one is correct and the others are plausible but incorrect
  - All answer options should be relevant to the topic of the question.
  - The wrong answers should really be wrong and not partially correct
  - The answers should be more or less of the same length
  - Answers are never allowed to be something like 'all of the above.'
  - Answers should always be phrased positively, rather than negatively.
  - Every question should ask only one question; do not put multiple questions together
  
  The user typically needs multiple questions on the same topic, so make sure there is some diversity in the questions you come up with.
  The questions should be appropriate for students and the bachelor level.
  
  The question must not directly refer to 'the context', as the person who will need to answer the question cannot see the context provided to you.
  The question must be stand-alone.

  Do not make the questions too easy.
  Make sure that the wrong answers are not too obviously wrong.

  Format your answers as follows:
  """
  Question
  a. Correct answer option
  b. Plausible but wrong answer option
  c. Plausible but wrong answer option
  d. Plausible but wrong answer option
  
  Eplanation of correct and incorrect answers.
  """
  
  The user may ask you to "simplify" a question.
  In that case, simplify your last generated response rather than making a new question.
  
  The user may ask you to "correct" a question and tell you what to correct.
  In that case, correct your last generated response as instructed by the user, rather than making a new question.
  ''')

mc_system_prompt = PromptTemplate(template=mc_system_prompt_template,
                                        input_variables=["chat_history", "input"],
                                        )
# Set up source file
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"answers/answers_{timestamp}.org"
with open(filename, 'w') as file:
  file.write("#+OPTIONS: toc:nil author:nil\n")
  file.write(f"#+TITLE: Answers and sources for session started on {timestamp}\n\n")

@cl.on_chat_start
async def start():
  settings = await cl.ChatSettings(
    [
      Select(
        id="Conceptual_Store",
        label="Conceptual vector store",
        values=["All", "DiCS", "Sturing", "Network Governance"],
        initial_index=0,
      ),
      Select(
        id="Empirical_Store",
        label="Empirical vector store",
        values=["Loss and Damage", "Nitrogen", "Inland Shipping", "Samso", "JSF"],
        initial_index=0,
      ),
      Select(
        id="Agent_Model",
        label="OpenAI - Agent Model",
        values=["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
        initial_index=0,
      ),
      Slider(
        id="Agent_Temperature",
        label="OpenAI - Agent Temperature",
        initial=0,
        min=0,
        max=2,
        step=0.1,
      ),
      Select(
        id="Conceptual_Model",
        label="OpenAI - Conceptual Model",
        values=["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
        initial_index=0,
      ),
      Slider(
        id="Conceptual_Temperature",
        label="OpenAI - Conceptual Temperature",
        initial=0,
        min=0,
        max=2,
        step=0.1,
      ),
      Select(
        id="Empirical_Model",
        label="OpenAI - Empirical Model",
        values=["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
        initial_index=0,
      ),
      Slider(
        id="Empirical_Temperature",
        label="OpenAI - Empirical Temperature",
        initial=0,
        min=0,
        max=2,
        step=0.1,
      ),
      Select(
        id="Writing_Model",
        label="OpenAI - Writing Model",
        values=["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
        initial_index=0,
      ),
      Slider(
        id="Writing_Temperature",
        label="OpenAI - Writing Temperature",
        initial=0,
        min=0,
        max=2,
        step=0.1,
      ),
      Select(
        id="Critical_Model",
        label="OpenAI - Critical Model",
        values=["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
        initial_index=0,
      ),
      Slider(
        id="Critical_Temperature",
        label="OpenAI - Critical Temperature",
        initial=0,
        min=0,
        max=2,
        step=0.1,
      ),
      Select(
        id="MC_Model",
        label="OpenAI - MC Model",
        values=["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
        initial_index=0,
      ),
      Slider(
        id="MC_Temperature",
        label="OpenAI - MC Temperature",
        initial=0,
        min=0,
        max=2,
        step=0.1,
      ),
    ]
  ).send()
  await setup_chain(settings)

# When settings are updated
@cl.on_settings_update
async def setup_chain(settings):
  # Set conceptual vector store
  chosen_conceptual = settings["Conceptual_Store"]
  conceptual = ""
  if (chosen_conceptual == "All"):
    conceptual = conceptual_all
  elif (chosen_conceptual == "DiCS"):
    conceptual = conceptual_dics
  elif (chosen_conceptual == "Sturing"):
    conceptual = conceptual_sturing
  elif (chosen_conceptual == "Network Governance"):
    conceptual = conceptual_ng

  # Set up conceptual chain reordering
  redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
  reordering = LongContextReorder()
  pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, reordering])
  conceptual_compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=conceptual.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k" : 20, "score_threshold": .65}))
  
  # Set empirical vector store
  chosen_empirical = settings["Empirical_Store"]
  empirical = ""
  if (chosen_empirical == "Loss and Damage"):
    empirical = empirical_ld
  elif (chosen_empirical == "Nitrogen"):
    empirical = empirical_nitrogen
  elif (chosen_empirical == "Inland Shipping"):
    empirical = empirical_inland_shipping
  elif (chosen_empirical == "Samso"):
    empirical = empirical_samso
  elif (chosen_empirical == "JSF"):
    empirical = empirical_jsf

  # Setup empirical reorder
  empirical_compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=empirical.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k" : 20, "score_threshold": .65}))

  # Set llms
  agent_llm=ChatOpenAI(
    temperature=settings["Agent_Temperature"],
    model=settings["Agent_Model"],
  )
  conceptual_llm=ChatOpenAI(
    temperature=settings["Conceptual_Temperature"],
    model=settings["Conceptual_Model"],
  )
  empirical_llm=ChatOpenAI(
    temperature=settings["Empirical_Temperature"],
    model=settings["Empirical_Model"],
  )
  writing_llm=ChatOpenAI(
    temperature=settings["Writing_Temperature"],
    model=settings["Writing_Model"],
  )
  critical_llm=ChatOpenAI(
    temperature=settings["Critical_Temperature"],
    model=settings["Critical_Model"],
  )
  mc_llm=ChatOpenAI(
    temperature=settings["MC_Temperature"],
    model=settings["MC_Model"],
  )

   # Initialize chain
  conceptual_chain = ConversationalRetrievalChain.from_llm(
    llm=conceptual_llm,
    retriever=conceptual_compression_retriever_reordered,
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': conceptual_chat_prompt},
    memory=readonlymemory,
    return_source_documents=True,
    condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k'),
  )

  # Wrap chain
  def run_conceptual_chain(question):
    results = conceptual_chain({"question": question}, return_only_outputs=True)
    sources = results['source_documents']
    with open(filename, 'a') as file:
      file.write("* Tool: Conceptual tool\n")
      file.write("* Query:\n")
      file.write(question)
      file.write("\n")
      file.write("* Answer:\n")
      file.write(results['answer'])
      file.write("\n")
      counter = 1
      for source in sources:
        file.write(f"** Document_{counter}: ")
        file.write(os.path.basename(source.metadata['source']))
        file.write("\n")
        file.write("*** Content:\n")
        file.write(source.page_content)
        file.write("\n")
      counter += 1
    return str(results['answer'])

   # Initialize empirical chain
  empirical_chain = ConversationalRetrievalChain.from_llm(
    llm=empirical_llm,
    retriever=empirical_compression_retriever_reordered,
    chain_type="stuff",
    combine_docs_chain_kwargs={'prompt': empirical_chat_prompt},
    memory=readonlymemory,
    return_source_documents = True,
    condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k'),
  )

  # Wrap empirical chain
  def run_empirical_chain(question):
    results = empirical_chain({"question": question}, return_only_outputs=True)
    sources = results['source_documents']
    with open(filename, 'a') as file:
      file.write("* Tool: Empirical tool\n\n")
      file.write("* Query:\n")
      file.write(question)
      file.write("\n")
      file.write("* Answer:\n")
      file.write(results['answer'])
      file.write("\n")
      counter = 1
      for source in sources:
        file.write(f"** Document_{counter}: ")
        file.write(source.metadata['source'])
        file.write("\n")
        file.write("*** Content:\n")
        file.write(source.page_content)
        file.write("\n")
      counter += 1
    return str(results['answer'])

 # Initialize writing chain
  writing_chain = ConversationChain(
    llm=writing_llm,
    prompt=writing_system_prompt,
    memory=readonlymemory,
  )

  # Wrap chain
  def run_writing_chain(question):
    results = writing_chain({"input": question}, return_only_outputs=True)
    with open(filename, 'a') as file:
      file.write("* Tool: Writing tool\n")
      file.write("* Query:\n")
      file.write(question)
      file.write("\n")
      file.write("* Answer:\n")
      file.write(results['response'])
      file.write("\n")
    return str(results['response'])

  # Initialize critical chain
  critical_chain = ConversationChain(
    llm=critical_llm,
    prompt=critical_system_prompt,
    memory=readonlymemory,
  )

  # Wrap chain
  def run_critical_chain(question):
    results = critical_chain({"input": question}, return_only_outputs=True)
    with open(filename, 'a') as file:
      file.write("* Tool: Critical tool\n")
      file.write("* Query:\n")
      file.write(question)
      file.write("\n")
      file.write("* Answer:\n")
      file.write(results['response'])
      file.write("\n")
    return str(results['response'])

  # Initialize MC chain
  mc_chain = ConversationChain(
    llm=mc_llm,
    prompt=mc_system_prompt,
    memory=readonlymemory,
  )

  # Wrap chain
  def run_mc_chain(question):
    results = mc_chain({"input": question}, return_only_outputs=True)
    with open(filename, 'a') as file:
      file.write("* Tool: MC tool\n")
      file.write("* Query:\n")
      file.write(question)
      file.write("\n")
      file.write("* Answer:\n")
      file.write(results['response'])
      file.write("\n")
    return str(results['response'])

  # Search API wrapper
  search = GoogleSerperAPIWrapper()

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
      func=run_writing_chain,
      description="""Useful for when you need to output texts based on input from the empirical and conceptual tool.
      The input should be a fully formed question, not referencing any obscure pronouns from the conversation before.
      The question should end with a question mark.
      """,
      return_direct=True,
      ),
    Tool(
      name="Critical tool",
      func=run_critical_chain,
      description="""Useful for when you need to offer a critique on something that was said before."
      The input should be a fully formed question, not referencing any obscure pronouns from the conversation before.
      The question should end with a question mark.
      """,
      return_direct=True,
      ),
    Tool(
      name="MC tool",
      func=run_mc_chain,
      description="""Useful for when you need to develop multiple choice questions."
      The input should be a fully formed question, not referencing any obscure pronouns from the conversation before.
      The question should end with a question mark.
      """,
      return_direct=True,
      ),
    Tool(
        name="Search tool",
        func=search.run,
        description="""Useful for when the user asks you to search for something on the internet.
        Only use this if the user explicitly asks you to search the internet."""
      ),
    ]
  
  # Set up agent
  agent = initialize_agent(tools,
                           llm=agent_llm,
                           agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                           verbose=True,
                           memory=memory,
                           handle_parsing_errors=True,
                           condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k'),
                           )

  cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: str):
  agent = cl.user_session.get("agent")
  cb = cl.LangchainCallbackHandler(stream_final_answer=True)
  res = await cl.make_async(agent.run)(message, callbacks=[cb])
  await cl.Message(content=res).send()
