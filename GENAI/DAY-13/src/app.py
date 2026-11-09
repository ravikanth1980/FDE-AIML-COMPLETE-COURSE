"""Streamlit interface for database-wide PostgreSQL RAG."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
load_dotenv(override=True)

from rag_core import (
    build_database_url,
    create_chat_model,
    create_db_engine,
    stream_sql_answer,
)


load_dotenv()

DB_HOST = "quantamv.c7a8gmqscnvy.ap-northeast-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "quantam"
CHAT_MODEL = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="PostgreSQL Data Assistant",
    page_icon="🧠",
    layout="wide",
)


def connect_database(password: str, groq_api_key: str) -> None:
    """Connect to PostgreSQL for read-only SQL question answering."""
    groq_api_key = groq_api_key.strip()
    if not password:
        st.error("Enter the PostgreSQL password to connect.")
        return
    if not groq_api_key:
        st.error("Enter a valid Groq API key to connect.")
        return

    database_url = build_database_url(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        username=DB_USER,
        password=password,
    )

    engine = create_db_engine(database_url)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    llm = create_chat_model(
        provider="groq",
        api_key=groq_api_key,
        model=CHAT_MODEL,
        temperature=0,
    )

    st.session_state.database_qa = {
        "engine": engine,
        "llm": llm,
        "chat_model": CHAT_MODEL,
    }


def render_sql_results(query: str, rows: list[dict]) -> None:
    """Show the exact SQL and rows used to answer a database question."""
    with st.expander("SQL query and database results"):
        st.code(query, language="sql")
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("The query returned no rows.")


def render_chat() -> None:
    """Render chat history and stream answers grounded in SQL results."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sql_query"):
                render_sql_results(
                    message["sql_query"],
                    message.get("sql_rows", []),
                )

    question = st.chat_input("Ask a question about your PostgreSQL data...")
    if not question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            database_qa = st.session_state.database_qa
            sql_query, sql_rows, token_stream = stream_sql_answer(
                question=question,
                engine=database_qa["engine"],
                llm=database_qa["llm"],
            )
            answer = st.write_stream(token_stream)
            render_sql_results(sql_query, sql_rows)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sql_query": sql_query,
                    "sql_rows": sql_rows,
                }
            )
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")


def main() -> None:
    """Render the PostgreSQL RAG application."""
    st.title("🧠 PostgreSQL Data Assistant")
    st.caption(
        "PostgreSQL | Read-only SQL answers | Groq"
    )
    st.write(
        f"Database: `{DB_NAME}` on `{DB_HOST}:{DB_PORT}` "
        f"(user `{DB_USER}`)"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.form("database_connection"):
        password = st.text_input(
            "PostgreSQL password",
            type="password",
        )
        groq_api_key = st.text_input(
            "Groq API key",
            type="password",
        )
        submitted = st.form_submit_button(
            "Connect to PostgreSQL",
            type="primary",
        )

    if submitted:
        try:
            connect_database(password, groq_api_key)
        except Exception as exc:
            st.error(f"Database connection failed: {exc}")

    if "database_qa" not in st.session_state:
        st.info(
            "Enter the PostgreSQL password and Groq API key, then connect "
            "to ask questions against live database records."
        )
        return

    st.caption(
        f"Model: Groq `{CHAT_MODEL}` · "
        "Answers are generated from live, read-only SQL query results."
    )
    render_chat()


if __name__ == "__main__":
    main()
