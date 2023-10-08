# The LLM education and research tool

This is an agent that is useful in parts of the preparation of courses and that can simulate (with flaws) a superficial qualitative analysis. 
The agent has multiple tools at its disposal:
1. A conceptual tool that is a ConversationalRetrievalChain with access to a vectorstore that should contain theoretical knowledge, for example in the form of books or papers.
2. An empirical tool that is a ConversationalRetrievalChain with access to a vectorstore that should contain empirical information.
3. A writing tool that is prompted to focus entirely on writing tasks.
4. A critiqueing tool that can be used to critique texts and offer possible suggestions for improvement.
5. A case description tool that can be used to create case descriptions based on the discussed content.
6. A multiple choice tool that can be used to create multiple choice questions based on the discussed content.
7. An open choice tool that can be used to create open questions based on the discussed content.
8. A search tool that allows Google searches.

Some of the tools can be turned off to support workflows in which these tools should not get in the way.

It is also possible to make the agent report back outputs from the various tools directly.
This prevents the agent from taking different steps, which means that the steps have to be thought of by the user.

The tools have access to a shared memory. 
Only the agent has write access to that memory.

The agent will choose one or more of these tools as steps in answering questions, depending on how the question is formulated.
In my experience, it works best so far when questions are also asked in steps. 
For example, you ask a conceptual question first, and then you ask empirical questions, so that the answer to the conceptual question can be used as a background.
