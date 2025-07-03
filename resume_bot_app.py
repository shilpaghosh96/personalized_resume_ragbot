# resume_bot_app.py

import streamlit as st

# Your existing code — UNCHANGED except wrapped in a function
def load_bot():
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        StorageContext,
        ServiceContext,
        load_index_from_storage
    )
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.llms.groq import Groq
    from llama_index.core import Settings
    import warnings
    warnings.filterwarnings('ignore')

    GROQ_API_KEY = "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"

    reader = SimpleDirectoryReader(input_files=["/workspaces/personalized_resume_ragbot/Resume_Shilpa_Ghosh_msds.pdf"])
    documents = reader.load_data()

    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 200

    vector_index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
        node_parser=nodes
    )

    vector_index.storage_context.persist(persist_dir="storage")

    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    return query_engine


# Streamlit UI
st.set_page_config(page_title="Resume Bot", layout="centered")
st.title("🤖 Resume Q&A Bot")

query_engine = load_bot()

query = st.text_input("Ask a question about the resume:", placeholder="e.g., What skills does Shilpa have?")

if query:
    with st.spinner("Thinking..."):
        response = query_engine.query(query)
        st.write("**Answer:**", response.response)
