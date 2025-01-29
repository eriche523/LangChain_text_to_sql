from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

db = SQLDatabase.from_uri("sqlite:///C:/Users/Eric_/PycharmProjects/LangChain_text_to_sql/LangChain_text_to_sql/resources/Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")

# step 1: Create a state class to store intermediate data

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# step 2: Import chat model, can use your own API

# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

# get key from .env the API key, Load the .env file
load_dotenv(dotenv_path=r"C:\Users\Eric_\PycharmProjects\LangChain_text_to_sql\LangChain_text_to_sql\.env")

# Get the API key
openrouter_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openrouter_api_key)

# step 3: Get prompt template from langhub
from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1
query_prompt_template.messages[0].pretty_print()

# step 4: get
from typing_extensions import Annotated

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# step 5: write query to test it out
write_query({"question": "How many Employees are there?"})

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

for step in graph.stream(
    {"question": "when were the 8 employees hired?"}, stream_mode="updates"
):
    print(step)


########################################################################################################################
# AGENT IMPLEMENTATION: SMART RETRIVAL LEVERAGING MULTIPLE DBs AND TABLES LOGIC
########################################################################################################################

from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

from langchain import hub

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1
prompt_template.messages[0].pretty_print()

system_message = prompt_template.format(dialect="SQLite", top_k=5)

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools, prompt=system_message)

question = "what's the most spent category?"

for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

########################################################################################################################
# High-Cardinality Columns: Columns with a lot of unique values or columns that are strings can be easily misspell
########################################################################################################################

import ast
import re


def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")
albums[:5]

# | output: false
# | echo: false

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key = '')

# | output: false
# | echo: false

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

from langchain.agents.agent_toolkits import create_retriever_tool

_ = vector_store.add_texts(artists + albums)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
description = (
    "Use to look up values to filter on. Input is an approximate spelling "
    "of the proper noun, output is valid proper nouns. Use the noun most "
    "similar to the search."
)
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

# Add to system message
suffix = (
    "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
    "the filter value using the 'search_proper_nouns' tool! Do not try to "
    "guess at the proper name - use this function to find similar ones."
)
system_message = prompt_template.format(dialect="SQLite", top_k=5)
system = f"{system_message}\n\n{suffix}"

tools.append(retriever_tool)

agent = create_react_agent(llm, tools, prompt=system)

question = "How many albums does alis in chain have?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

########################################################################################################################
# Many Tables: Multiple Columns and schemas and complex inner joins
########################################################################################################################

import re
from typing import List
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough

# Define the Table class
class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

# Get usable table names from the database
table_names = "\n".join(db.get_usable_table_names())

# Define the system prompt for table selection
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)

# Bind the LLM with tools and set up the output parser
llm_with_tools = llm.bind_tools([Table])
output_parser = PydanticToolsParser(tools=[Table])

# Create the table chain
table_chain = prompt | llm_with_tools | output_parser

table_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"})

# Define the system prompt for category selection
system = """Return the names of any SQL tables that are relevant to the user question.
The tables are:

Music
Business
"""

# Create the prompt template for category selection
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)

# Create the category chain
category_chain = prompt | llm_with_tools | output_parser

# Define a function to get relevant tables based on categories
def get_tables(categories: List[Table]) -> List[str]:
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables

# Update the table chain to include category-based table selection
table_chain = category_chain | get_tables

# Create the full chain for generating SQL queries
query_chain = create_sql_query_chain(llm, db)
table_chain = {"input": itemgetter("question")} | table_chain
full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

# Function to clean the SQL query
def clean_sql_query(generated_query: str) -> str:
    """
    Cleans the SQL query by removing unwanted prefixes, backticks, and whitespace.
    """
    # Remove "SQLQuery:" prefix if present
    if generated_query.startswith("SQLQuery:"):
        generated_query = generated_query[len("SQLQuery:"):].strip()

    # Remove triple backticks and "sql" (if present)
    if generated_query.startswith("```sql"):
        generated_query = generated_query[len("```sql"):].strip()
    if generated_query.endswith("```"):
        generated_query = generated_query[:-len("```")].strip()

    # Strip any leading or trailing whitespace
    cleaned_query = generated_query.strip()

    return cleaned_query

# Generate and clean the SQL query
query = full_chain.invoke({"question": "What are all the genres of Alanis Morisette songs"})
print("Generated SQL Query:", query)

# Clean the query
cleaned_query = clean_sql_query(query)
print("Cleaned SQL Query:", cleaned_query)

# Execute the cleaned query
try:
    result = db.run(cleaned_query)
    print("Query Result:", result)
except Exception as e:
    print(f"Error executing query: {e}")