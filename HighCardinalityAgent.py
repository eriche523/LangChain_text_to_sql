import ast
import re
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import os
import config

class SQLQueryAgent:
    def __init__(self, database: object, llm, openrouter_api_key):
        """
        Initialize the SQLQueryAgent with a database URI.

        Args:
            db_uri (str): The URI of the SQL database.
        """
        self.llm = llm

        # Initialize the database
        self.db = database

        # Initialize the toolkit and tools
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()

        # for access to embedding vector processing
        self.openrouter_api_key = openrouter_api_key

        # Load the prompt template
        self.prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        assert len(self.prompt_template.messages) == 1
        self.prompt_template.messages[0].pretty_print()

        # Format the system message
        self.system_message = self.prompt_template.format(dialect="SQLite", top_k=5)

        # Initialize the retriever tool
        self.retriever_tool = self._initialize_retriever_tool()

        # Add the retriever tool to the tools list
        self.tools.append(self.retriever_tool)

        # Create the agent
        self.agent = create_react_agent(self.llm, self.tools, prompt=self.system_message)

    def _initialize_retriever_tool(self):
        """
        Initialize the retriever tool for proper noun lookup.

        Returns:
            The retriever tool.
        """
        # Query the database for artists and albums
        artists = self.query_as_list("SELECT Name FROM Artist")
        albums = self.query_as_list("SELECT Title FROM Album")

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=self.openrouter_api_key)
        vector_store = InMemoryVectorStore(embeddings)

        # Add texts to the vector store
        vector_store.add_texts(artists + albums)

        # Create the retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # Define the retriever tool description
        description = (
            "Use to look up values to filter on. Input is an approximate spelling "
            "of the proper noun, output is valid proper nouns. Use the noun most "
            "similar to the search."
        )

        # Create and return the retriever tool
        return create_retriever_tool(
            retriever,
            name="search_proper_nouns",
            description=description,
        )

    def query_as_list(self, query: str):
        """
        Execute a query and return the results as a cleaned list.

        Args:
            query (str): The SQL query to execute.

        Returns:
            List[str]: The cleaned results.
        """
        res = self.db.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return list(set(res))

    def run_agent(self, question: str):
        """
        Run the agent with a given question.

        Args:
            question (str): The question to ask the agent.
        """
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()


# Example usage
if __name__ == "__main__":
    # Load API Key from .env file
    load_dotenv(dotenv_path=config.api_env_path)
    api_key = os.getenv("OPENAI_API_KEY")

    # --------------------- INITIALIZE LLM AND DATABASE ---------------------

    # Initialize LLM for routing
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    # Initialize database connection
    db_uri = config.db_path
    db = SQLDatabase.from_uri(db_uri)

    # Initialize the Query Agent with an external LLM
    agent = SQLQueryAgent(
        database=db,
        llm=router_llm,
        openrouter_api_key=api_key
    )

    # Run the agent with a sample question
    agent.run_agent("what is the most sold genrel?")