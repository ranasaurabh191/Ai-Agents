from typing import List, Dict, Optional, Tuple
import os
import tempfile
from datetime import datetime
import uuid
import asyncio

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed import TextEmbedding

# NEW: Gemini SDK
import google.generativeai as genai

from agents import Agent, Runner

load_dotenv()

# Constants
COLLECTION_NAME = "voice-rag-agent"

def init_session_state() -> None:
    defaults = {
        "initialized": False,
        "qdrant_url": "",
        "qdrant_api_key": "",
        "gemini_api_key": "",
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "tts_agent": None,
        "selected_voice": "coral",
        "processed_documents": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_sidebar() -> None:
    with st.sidebar:
        st.title("🔑 Configuration")
        st.markdown("---")

        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
            type="password"
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=st.session_state.qdrant_api_key,
            type="password"
        )
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password"
        )

        st.markdown("---")
        st.markdown("### 🎤 Voice Settings (Not supported in Gemini yet)")

def setup_qdrant() -> Tuple[QdrantClient, TextEmbedding]:
    if not all([st.session_state.qdrant_url, st.session_state.qdrant_api_key]):
        raise ValueError("Qdrant credentials not provided")

    client = QdrantClient(
        url=st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key
    )

    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    return client, embedding_model

def process_pdf(file) -> List:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []

def store_embeddings(client, embedding_model, documents, collection_name):
    for doc in documents:
        embedding = list(embedding_model.embed([doc.page_content]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": doc.page_content,
                        **doc.metadata
                    }
                )
            ]
        )

def setup_agents(gemini_api_key: str):
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    genai.configure(api_key=gemini_api_key)

    processor_agent = Agent(
        name="Documentation Processor",
        instructions="""You are a helpful documentation assistant...""",
        model="gemini-1.5-flash"
    )

    # Gemini does not do TTS
    tts_agent = None
    return processor_agent, tts_agent

async def process_query(query, client, embedding_model, collection_name, gemini_api_key, voice: str):
    try:
        st.info("🔄 Step 1: Generating query embedding and searching documents...")
        query_embedding = list(embedding_model.embed([query]))[0]

        search_response = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=3,
            with_payload=True
        )
        search_results = search_response.points if hasattr(search_response, 'points') else []

        if not search_results:
            raise Exception("No relevant documents found")

        context = "Based on the following documentation:\n\n"
        for result in search_results:
            payload = result.payload
            if payload:
                context += f"From {payload.get('file_name','Unknown')}:\n{payload.get('content','')}\n\n"

        context += f"\nUser Question: {query}\n\n"

        st.info("🔄 Step 2: Calling Gemini API...")
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(context)

        text_response = response.text

        st.success("✅ Query processed!")
        return {
            "status": "success",
            "text_response": text_response,
            "sources": [r.payload.get('file_name', 'Unknown Source') for r in search_results if r.payload]
        }

    except Exception as e:
        st.error(f"❌ Error during query processing: {str(e)}")
        return {"status": "error", "error": str(e)}

def main():
    st.set_page_config(page_title="Gemini RAG Agent", page_icon="🎙️", layout="wide")
    init_session_state()
    setup_sidebar()

    st.title("🤖 Gemini RAG Agent")
    st.info("Upload PDFs and ask questions. Responses powered by Google Gemini API.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_documents:
            with st.spinner("Processing PDF..."):
                if not st.session_state.client:
                    client, embedding_model = setup_qdrant()
                    st.session_state.client = client
                    st.session_state.embedding_model = embedding_model

                documents = process_pdf(uploaded_file)
                if documents:
                    store_embeddings(st.session_state.client, st.session_state.embedding_model, documents, COLLECTION_NAME)
                    st.session_state.processed_documents.append(uploaded_file.name)
                    st.success(f"✅ Added PDF: {uploaded_file.name}")
                    st.session_state.setup_complete = True

    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Documents")
        for doc in st.session_state.processed_documents:
            st.sidebar.text(f"📄 {doc}")

    query = st.text_input("Ask a question about your documents", disabled=not st.session_state.setup_complete)
    if query and st.session_state.setup_complete:
        result = asyncio.run(process_query(query, st.session_state.client, st.session_state.embedding_model, COLLECTION_NAME, st.session_state.gemini_api_key, st.session_state.selected_voice))
        if result["status"] == "success":
            st.markdown("### Response:")
            st.write(result["text_response"])
            st.markdown("### Sources:")
            for source in result["sources"]:
                st.markdown(f"- {source}")

if __name__ == "__main__":
    main()
