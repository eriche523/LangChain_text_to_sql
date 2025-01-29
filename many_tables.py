from typing import List
from langchain_core.output_parsers.openai_tools import PydanticToolsParser  # Parses LLM output
from langchain_core.prompts import ChatPromptTemplate  # Template for chat-based LLM prompts
from pydantic import BaseModel, Field  # Data validation & type enforcement
from operator import itemgetter  # Utility for extracting dictionary values
from langchain.chains import create_sql_query_chain  # Chain to process SQL queries
from langchain_core.runnables import RunnablePassthrough  # Passes input data between functions
from langchain_community.utilities import SQLDatabase  # Utility for handling SQL databases
from langchain_openai import ChatOpenAI  # OpenAI model integration for SQL query generation
from dotenv import load_dotenv  # Loads environment variables
import os


class SQLQueryWrapper:

    def __init__(self, database: object, llm):
        """Initialize LLM, database, and query chains."""

        self.db = database
        self.llm = llm

        self.table_names = "\n".join(self.db.get_usable_table_names())
        self.full_chain = self._setup_chain()

    def _setup_chain(self):
        """Setup the entire SQL query chain."""
        class Table(BaseModel):
            """Table in SQL database."""
            name: str = Field(description="Name of table in SQL database.")

        system_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
        The tables are:{self.table_names}

        Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        llm_with_tools = self.llm.bind_tools([Table])
        output_parser = PydanticToolsParser(tools=[Table])

        # table_chain = prompt | llm_with_tools | output_parser
        category_prompt = ChatPromptTemplate.from_messages([
            ("system", "Return the names of any SQL tables that are relevant to the user question.\n\nMusic\nBusiness"),
            ("human", "{input}")
        ])

        category_chain = category_prompt | llm_with_tools | output_parser

        def get_tables(categories: List[Table]) -> List[str]:
            tables = []
            for category in categories:
                if category.name == "Music":
                    tables.extend(["Album", "Artist", "Genre", "MediaType", "Playlist", "PlaylistTrack", "Track"])
                elif category.name == "Business":
                    tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
            return tables

        table_chain = category_chain | get_tables
        query_chain = create_sql_query_chain(self.llm, self.db)
        table_chain = {"input": itemgetter("question")} | table_chain

        return RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

    @staticmethod
    def clean_sql_query(generated_query: str) -> str:
        """Cleans the SQL query by removing unwanted prefixes, backticks, and whitespace."""
        if generated_query.startswith("SQLQuery:"):
            generated_query = generated_query[len("SQLQuery:"):].strip()

        if generated_query.startswith("```sql"):
            generated_query = generated_query[len("```sql"):].strip()
        if generated_query.endswith("```"):
            generated_query = generated_query[:-len("```")].strip()

        return generated_query.strip()

    def generate_query(self, question: str) -> str:
        """Generate a clean SQL query."""
        query = self.full_chain.invoke({"question": question})
        return self.clean_sql_query(query)

    def execute_query(self, query: str):
        """Execute the given SQL query on the database."""
        try:
            return self.db.run(query)
        except Exception as e:
            return f"Error executing query: {e}"

# Example usage
if __name__ == "__main__":
    # Load API Key from .env file
    load_dotenv(dotenv_path=r"C:\Users\Eric_\PycharmProjects\LangChain_text_to_sql\LangChain_text_to_sql\.env")
    api_key = os.getenv("OPENAI_API_KEY")

    # --------------------- INITIALIZE LLM AND DATABASE ---------------------

    # Initialize LLM for routing
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    # Initialize database connection
    db_uri = "sqlite:///C:/Users/Eric_/PycharmProjects/LangChain_text_to_sql/LangChain_text_to_sql/resources/Chinook.db"
    db = SQLDatabase.from_uri(db_uri)

    # Initialize the Query Agent with an external LLM
    agent = SQLQueryWrapper(
        database=db,
        llm=router_llm
    )

    # Generate SQL Query
    generated_query = agent.generate_query("what is the total revenue of Alanis Morissette's songs?")
    print("Generated SQL Query:", generated_query)

    # Execute SQL Query
    query_result = agent.execute_query(generated_query)
    print("Query Result:", query_result)
