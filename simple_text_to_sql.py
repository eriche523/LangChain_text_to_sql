import os
import config
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langchain_community.utilities import SQLDatabase
from langchain import hub
from typing_extensions import TypedDict, Annotated

class SimpleSQLAgent:
    """A production-ready LangChain-based agent for processing simple SQL-related queries."""

    def __init__(self, database: object, llm):
        """
        Initialize the SimpleSQLAgent.

        Args:
        - db_path (str): Path to the SQLite database.
        - llm: Pre-initialized language model (OpenAI/GPT/Claude).
        """
        self.llm = llm

        # Initialize the Database
        self.db = database

        # Load Prompt Template from LangChain Hub
        self.query_prompt_template = self._load_prompt_template()

        # Build the SQL Execution Graph
        self.graph = self._build_graph()

    def _load_prompt_template(self):
        """Load the system prompt template from LangChain Hub."""
        prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        assert len(prompt_template.messages) == 1, "Prompt template structure is incorrect!"
        return prompt_template

    def _build_graph(self):
        """Builds the StateGraph with the database for structured query execution."""

        class State(TypedDict):
            """State structure for query execution."""
            question: str
            query: str
            result: str
            answer: str

        class QueryOutput(TypedDict):
            """Defines structured SQL query output."""
            query: Annotated[str, ..., "Syntactically valid SQL query."]

        def write_query(state: State, db: SQLDatabase):
            """Generate a SQL query to fetch information based on user input."""
            prompt = self.query_prompt_template.invoke(
                {
                    "dialect": db.dialect,
                    "top_k": 10,
                    "table_info": db.get_table_info(),
                    "input": state["question"],
                }
            )
            structured_llm = self.llm.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            return {"query": result["query"]}

        def execute_query(state: State, db: SQLDatabase):
            """Execute the generated SQL query."""
            execute_query_tool = QuerySQLDatabaseTool(db=db)
            return {"result": execute_query_tool.invoke(state["query"])}

        def generate_answer(state: State):
            """Generate a human-readable answer from the SQL query result."""
            prompt = (
                "Given the following user question, corresponding SQL query, "
                "and SQL result, answer the user question.\n\n"
                f'Question: {state["question"]}\n'
                f'SQL Query: {state["query"]}\n'
                f'SQL Result: {state["result"]}'
            )
            response = self.llm.invoke(prompt)
            return {"answer": response.content}

        # Construct the state graph for SQL execution
        graph_builder = StateGraph(State)

        # Add nodes for query execution
        graph_builder.add_node("write_query", lambda state: write_query(state, self.db))
        graph_builder.add_node("execute_query", lambda state: execute_query(state, self.db))
        graph_builder.add_node("generate_answer", generate_answer)

        # Define execution sequence
        graph_builder.add_edge(START, "write_query")
        graph_builder.add_edge("write_query", "execute_query")
        graph_builder.add_edge("execute_query", "generate_answer")

        return graph_builder.compile()

    def query(self, question: str):
        """
        Process a schema-related query using the agent.

        Args:
        - question (str): The natural language question.

        Returns:
        - The processed response from the agent.
        """
        print(f"Executing Query: {question}")

        # Stream the query execution
        response = []
        for step in self.graph.stream({"question": question}, stream_mode="updates"):
            response.append(step)
            print(f"Response: {step}")  # Debugging

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
    agent = SimpleSQLAgent(
        database=db,
        llm=router_llm
    )

    # Example Queries
    agent.query("List all tables in the database.")
    agent.query("What columns are in the Artist table?")
