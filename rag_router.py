import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

from many_tables import SQLQueryWrapper
from HighCardinalityAgent import SQLQueryAgent
from agent_text_to_sql import SQLQueryAgent as AdvancedSQLQueryAgent
from simple_text_to_sql import SimpleSQLAgent

# --------------------- LOAD ENVIRONMENT VARIABLES ---------------------

# Load API Key from .env file
load_dotenv(dotenv_path=r"C:\Users\Eric_\PycharmProjects\LangChain_text_to_sql\LangChain_text_to_sql\.env")
api_key = os.getenv("OPENAI_API_KEY")

# --------------------- INITIALIZE LLM AND DATABASE ---------------------

# Initialize LLM for routing
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# Initialize database connection
db_uri = "sqlite:///C:/Users/Eric_/PycharmProjects/LangChain_text_to_sql/LangChain_text_to_sql/resources/Chinook.db"
db = SQLDatabase.from_uri(db_uri)

# --------------------- INITIALIZE QUERY AGENTS ---------------------

sql_wrapper = SQLQueryWrapper(
    database=db, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
) # General SQL queries
sql_agent = SQLQueryAgent(
    database=db, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key), openrouter_api_key = api_key
) # High cardinality (GROUP BY, COUNT, etc.)
advanced_sql_agent = AdvancedSQLQueryAgent(
    database=db, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
)  # Complex SQL queries (nested subqueries)

simple_sql_agent = SimpleSQLAgent(
    database=db, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
)  # Schema-based queries (listing tables, columns, etc.)

# --------------------- QUERY CLASSIFICATION ---------------------

def classify_question(question: str) -> str:
    """
    Determines the correct query agent based on the question type.

    - 'general_sql': Simple queries (e.g., fetching data).
    - 'high_cardinality': Aggregations (COUNT, SUM, GROUP BY).
    - 'advanced_sql': Complex queries (nested subqueries, deep joins).
    - 'schema_query': Schema metadata queries (list tables, columns).
    """

    system_prompt = """
    Classify the user's question into one of the following categories:
    - 'general_sql': If the question involves simple SELECT queries.
    - 'high_cardinality': If the question involves COUNT, SUM, GROUP BY, or multiple joins.
    - 'advanced_sql': If the question involves nested subqueries or complex SQL logic.
    - 'schema_query': If the question involves database metadata like listing tables or columns.

    Example:
    User: "What is the total revenue of Alanis Morissette songs?"
    Output: "general_sql"

    User: "What is the most sold genre?"
    Output: "high_cardinality"

    User: "Which artist has the most albums?"
    Output: "high_cardinality"

    User: "Who are the top 5 artists with the most album sales?"
    Output: "advanced_sql"

    User: "List all tables in the database."
    Output: "schema_query"

    Always return only one of the four labels: 'general_sql', 'high_cardinality', 'advanced_sql', or 'schema_query'.
    """

    response = router_llm.invoke(f"{system_prompt}\n\nUser: {question}\nOutput:")
    return response.content.strip()

# --------------------- ROUTING FUNCTION ---------------------

def run_rag_pipeline(question: str):
    """
    Routes the question to the appropriate SQL agent and executes the query.
    """

    category = classify_question(question)
    print(f"Routing to: {category}")

    if category == "schema_query":
        # Executes schema-related queries using SimpleSQLAgent
        result = simple_sql_agent.query(question)
    elif category == "general_sql":
        # Executes general SQL queries using SQLQueryWrapper
        query = sql_wrapper.generate_query(question)
        result = sql_wrapper.execute_query(query)
    elif category == "high_cardinality":
        # Executes aggregation queries using HighCardinalityAgent
        result = sql_agent.run_agent(question)
    else:  # advanced_sql
        # Executes complex queries using AdvancedSQLAgent
        result = advanced_sql_agent.query(question)

    # print("Query Result:", result)
    return result

# --------------------- EXAMPLE USAGE ---------------------

if __name__ == "__main__":
    user_question = "list all the tables."
    run_rag_pipeline(user_question)
