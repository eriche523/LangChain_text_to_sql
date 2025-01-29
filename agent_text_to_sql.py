import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain import hub


class SQLQueryAgent:
    """A production-ready LangChain-based SQL Querying Agent."""

    def __init__(self, database: object, llm):
        """
        Initialize the SQLQueryAgent.

        Args:
        - db_path (str): Path to the SQLite database.
        - llm: Pre-initialized language model (OpenAI/GPT/Claude).
        """
        self.llm = llm

        # Initialize the Database
        self.db = database

        # Initialize the Toolkit for SQL Querying
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()

        # Load Prompt Template from LangChain Hub
        self.system_message = self._load_prompt_template()

        # Create the Agent
        self.agent_executor = create_react_agent(self.llm, self.tools, prompt=self.system_message)

    def _load_prompt_template(self):
        """Load the system prompt template from LangChain Hub."""
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        assert len(prompt_template.messages) == 1, "Prompt template structure is incorrect!"
        return prompt_template.format(dialect="SQLite", top_k=5)

    def query(self, question: str):
        """
        Process a SQL-related query using the agent.

        Args:
        - question (str): The natural language question.

        Returns:
        - The processed response from the agent.
        """
        print(f"🟢 Executing Query: {question}")

        # Stream the query execution
        response = []
        for step in self.agent_executor.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
        ):
            response.append(step["messages"][-1].content)
            print(f"Response: {step['messages'][-1].content}")  # Debugging

        return response


# --------------------- USAGE EXAMPLE ---------------------

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
        llm=router_llm
    )
    # Example Queries
    agent.query("What's the most spent category?")
    agent.query("Who are the top 5 artists with the most album sales?")
