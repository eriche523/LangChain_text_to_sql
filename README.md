# RAG-Based Text-to-SQL System with Modular Query Routing

## Overview
This project implements a **RAG (Retrieval-Augmented Generation) powered Text-to-SQL system**, which intelligently routes **natural language queries** to the appropriate **SQL execution module** based on query complexity.

With the increasing power of **Large Language Models (LLMs)** in natural language understanding, converting **human-like questions into structured SQL queries** has become possible. However, SQL queries can vary in complexity. This system ensures **efficient and accurate SQL generation** by:

- **Dynamically classifying the query** based on its complexity.
- **Routing it to the correct module** for execution.
- **Retrieving relevant database schema** to generate more accurate SQL statements.

---

## What is Retrieval-Augmented Generation (RAG)?
**Retrieval-Augmented Generation (RAG)** is an approach that **enhances LLM-generated responses** by incorporating **retrieved knowledge** from an external source (e.g., a database, document repository, or knowledge base). 

Traditional LLMs rely solely on their **pre-trained knowledge**, which may be **outdated** or **lacking specific details**. **RAG improves this by dynamically retrieving relevant data** before generating responses.

### **How RAG Enhances Text-to-SQL**
- Retrieves **database schema information** before query generation.
- Uses **historical queries** to refine generated SQL statements.
- Reduces **hallucination** (incorrect SQL generation) by grounding responses in **actual database structures**.

---

## Role of LangChain in the Project
LangChain is a framework designed to integrate **LLMs with external data sources** to build **intelligent applications**.

### **How LangChain is Used Here**
1. **SQL Query Generation:**
   - Uses **LangChain’s `ChatPromptTemplate`** to structure query-generation prompts.
   - Uses **LangChain’s `create_sql_query_chain`** to build a pipeline for processing SQL queries.
   
2. **Database Interaction:**
   - Uses **LangChain’s `SQLDatabase` utility** to connect to the SQLite database.
   - Uses **LangChain’s `SQLDatabaseToolkit`** to facilitate SQL query execution.
   
3. **Query Execution Pipeline:**
   - Uses **LangChain’s `RunnablePassthrough`** to pass SQL queries through a sequence of processing steps.

By combining **LangChain with LLMs and RAG**, the system ensures more **accurate SQL query generation and efficient execution**.

---

## Query Routing System
At the core of this system is the **Query Router**, which classifies and directs SQL queries to the appropriate module for execution.

### **How Query Routing Works**
1. **User inputs a natural language question** (e.g., _"What is the most sold genre?"_).
2. **The router classifies the query** based on its structure:
   - **Simple queries** → `many_tables.py`
   - **Aggregations & GROUP BY** → `HighCardinalityAgent.py`
   - **Complex subqueries** → `agent_text_to_sql.py`
   - **Schema lookups** → `simple_text_to_sql.py`
3. **Routes the query to the correct module**, executes it, and returns results.

This **prevents simple queries from being over-processed** and **ensures complex queries receive sufficient execution power**.

---

## LLMs in Text-to-SQL Conversion
LLMs such as **GPT-4o-mini** can generate **syntactically correct SQL queries** from natural language. However, LLMs have **limitations**, which is why **RAG and query routing** are critical.

### **LLM Contributions in This Project**
- **Classifies query types** (_simple, aggregate, complex, schema-related_).
- **Generates SQL queries** based on **database schema information**.
- **Reformats SQL queries** for proper syntax.
- **Provides natural language explanations** of query results.

By **combining LLMs with LangChain’s structured retrieval mechanisms**, the system ensures that queries are both **accurate and optimized**.

---

## Example Usage

To run a specific query programmatically, use the following example:

rag_router.py

```python
if __name__ == "__main__":
    user_question = "the cheapest ticket for each year by artist name."
    run_rag_pipeline(user_question)


Routing to: advanced_sql
Query Result: [('The Rolling Stones', 1998, $45.00), ('U2', 1999, $50.00), ('Coldplay', 2002, $30.00)]


![text_to_sql_demo_gif](https://github.com/user-attachments/assets/3d30c7a0-03fc-4491-9293-c35e08eaa244)

