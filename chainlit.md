# The superficial qualitative analysis simulator

This is an agent that can simulate (with flaws) a superficial qualitative analysis. 
The agent has three tools at its disposal:
1. A conceptual tool that is a ConversationalRetrievalChain with access to a vectorstore that should contain theoretical knowledge, for example in the form of books or papers.
2. An empirical tool that is a ConversationalRetrievalChain with access to a vectorstore that should contain empirical information.
3. A writing tool that is prompted to focus entirely on writing tasks.

The tools have access to a shared memory. 
The writing tool and the agent currently have read only access. 
This is for technical reasons.

The agent will choose one or more of these tools as steps in answering questions, depending on how the question is formulated.
In my experience, it works best so far when questions are also asked in steps. 
For example, you ask a conceptual question first, and then you ask empirical questions, so that the answer to the conceptual question can be used as a background.
