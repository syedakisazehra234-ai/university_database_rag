import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "University Academic Knowledge Assistant"

GROQ_MODEL = "openai/gpt-oss-120b"

BASE_DIR = Path(__file__).resolve().parent

DATABASE_DIR = BASE_DIR / "rag_database"

FAISS_INDEX_PATH = (
    DATABASE_DIR / "faiss_index" / "index.faiss"
)

METADATA_PATH = (
    DATABASE_DIR / "metadata.json"
)

CONFIG_PATH = (
    DATABASE_DIR / "config.json"
)


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .source-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 12px;
        margin-top: 10px;
        background-color: rgba(128, 128, 128, 0.05);
    }

    .source-title {
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 University Academic Knowledge Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about the university's academic knowledge base.
    Answers are generated from the indexed academic documents and
    include source information for traceability.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data
def load_config():
    """
    Load the configuration generated during the offline
    Colab indexing process.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():
    """
    Load chunk metadata.

    Each metadata position corresponds to the same FAISS
    vector ID.

    Example:

    FAISS vector 0
        ↓
    metadata["chunks"][0]
    """

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():
    """
    Load the pre-built FAISS index.

    The index is NOT created during application runtime.
    """

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found:\n"
            f"{FAISS_INDEX_PATH}"
        )

    return faiss.read_index(
        str(FAISS_INDEX_PATH)
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(
    model_name: str,
):
    """
    Load the SAME embedding model used during indexing.

    This is critical.

    The model used for document embeddings and question
    embeddings must be the same.
    """

    return SentenceTransformer(
        model_name
    )


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Cloud Secrets."
        )

    return Groq(
        api_key=api_key
    )


# ============================================================
# RETRIEVE RELEVANT DOCUMENT CHUNKS
# ============================================================

def retrieve_chunks(
    question: str,
    index,
    metadata,
    embedding_model,
    top_k: int,
):
    """
    Embed the user's question and search the pre-built FAISS
    index.

    Returns source-aware retrieved chunks.
    """

    # --------------------------------------------------------
    # Embed QUESTION
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k,
    )


    results = []

    chunks = metadata.get(
        "chunks",
        [],
    )


    # --------------------------------------------------------
    # Map FAISS IDs back to metadata
    # --------------------------------------------------------

    for score, vector_id in zip(
        scores[0],
        indices[0],
    ):

        # FAISS uses -1 when no result exists.

        if vector_id < 0:
            continue


        vector_id = int(
            vector_id
        )


        if vector_id >= len(chunks):
            continue


        chunk = chunks[
            vector_id
        ].copy()


        # Add similarity score separately.

        chunk["similarity_score"] = float(
            score
        )


        results.append(
            chunk
        )


    return results


# ============================================================
# BUILD GROQ CONTEXT
# ============================================================

def build_context(
    retrieved_chunks,
):
    """
    Convert retrieved chunks into a structured context
    for the LLM.

    Source IDs are explicitly included so the model can
    reference them in its response.
    """

    context_parts = []


    for i, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        source_id = f"S{i}"

        context_parts.append(
            f"""
SOURCE {source_id}

File: {chunk.get("source_file", "Unknown")}
Page: {chunk.get("page_number", "Unknown")}
Chunk: {chunk.get("chunk_id", "Unknown")}

Content:
{chunk.get("text", "")}
""".strip()
        )


    return "\n\n---\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer(
    question: str,
    context: str,
):
    """
    Generate a grounded answer using Groq.

    The LLM is instructed to use only the retrieved
    academic context.
    """

    client = get_groq_client()


    system_prompt = """
You are a University Student and Academic Knowledge Assistant.

Your job is to answer student questions using ONLY the
provided knowledge-base context.

IMPORTANT RULES:

1. Use the retrieved context as your primary source of truth.

2. Do not invent facts that are not supported by the context.

3. If the answer cannot be established from the provided
   context, clearly say that the information was not found
   in the available academic knowledge base.

4. Do not pretend that general model knowledge came from
   the university documents.

5. Give a clear, useful and academically appropriate answer.

6. When the retrieved sources support the answer, cite them
   using the source IDs exactly as provided.

7. Example citation:
   [S1]

8. If multiple sources support a statement:
   [S1][S3]

9. Do not create fake source IDs.

10. Keep the answer focused on the student's question.

11. For procedures, policies, requirements or rules, preserve
    important conditions and exceptions from the source.

12. If the retrieved documents contain conflicting information,
    explicitly mention the conflict and identify the relevant
    sources.

The source material appears below.
"""


    user_prompt = f"""
KNOWLEDGE BASE CONTEXT
======================

{context}

======================
STUDENT QUESTION
======================

{question}

======================

Answer the student's question using the knowledge-base
context above.

Include source citations such as [S1] or [S2] where appropriate.
"""


    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        model=GROQ_MODEL,
        temperature=0.1,
    )


    return (
        chat_completion
        .choices[0]
        .message
        .content
    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(
    retrieved_chunks,
):
    """
    Display the actual source metadata retrieved from FAISS.

    This provides source traceability independently of what
    the LLM says.
    """

    if not retrieved_chunks:
        return


    st.markdown(
        "### 📚 Sources"
    )


    for i, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        source_file = chunk.get(
            "source_file",
            "Unknown",
        )

        page_number = chunk.get(
            "page_number",
            "Unknown",
        )

        chunk_id = chunk.get(
            "chunk_id",
            "Unknown",
        )

        score = chunk.get(
            "similarity_score",
            0.0,
        )


        with st.expander(
            f"S{i} — {source_file} — Page {page_number}"
        ):

            st.markdown(
                f"**Source file:** `{source_file}`"
            )

            st.markdown(
                f"**Page:** `{page_number}`"
            )

            st.markdown(
                f"**Chunk ID:** `{chunk_id}`"
            )

            st.markdown(
                f"**Similarity:** `{score:.4f}`"
            )

            st.markdown(
                "**Retrieved content:**"
            )

            st.write(
                chunk.get(
                    "text",
                    "",
                )
            )


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

try:

    config = load_config()

    metadata = load_metadata()

    faiss_index = load_faiss_index()


    embedding_model_name = config.get(
        "embedding_model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


    embedding_model = load_embedding_model(
        embedding_model_name
    )


except Exception as error:

    st.error(
        "❌ The academic knowledge base could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Knowledge Base"
    )


    document_count = config.get(
        "source_document_count",
        "Unknown",
    )


    vector_count = config.get(
        "vector_count",
        "Unknown",
    )


    st.metric(
        "Documents",
        document_count,
    )


    st.metric(
        "Indexed Chunks",
        vector_count,
    )


    st.caption(
        f"Embedding model:\n{embedding_model_name}"
    )


    st.caption(
        f"LLM: {GROQ_MODEL}"
    )


    st.divider()


    st.markdown(
        """
        **How it works**

        1. Your question is embedded.
        2. FAISS finds relevant academic chunks.
        3. Source metadata is recovered.
        4. Groq generates a grounded answer.
        5. Retrieved sources are displayed below.
        """
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about the academic documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # Display user question

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    top_k = int(
        config.get(
            "default_top_k",
            5,
        )
    )


    with st.spinner(
        "Searching the academic knowledge base..."
    ):

        retrieved_chunks = retrieve_chunks(
            question=question,
            index=faiss_index,
            metadata=metadata,
            embedding_model=embedding_model,
            top_k=top_k,
        )


    if not retrieved_chunks:

        answer = (
            "I could not find relevant information "
            "in the academic knowledge base."
        )

        with st.chat_message(
            "assistant"
        ):

            st.warning(
                answer
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.stop()


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        retrieved_chunks
    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question=question,
                    context=context,
                )


                st.markdown(
                    answer
                )


                # Show retrieved sources

                display_sources(
                    retrieved_chunks
                )


            except Exception as error:

                st.error(
                    "❌ Failed to generate the answer."
                )

                st.exception(
                    error
                )

                answer = (
                    "An error occurred while generating "
                    "the answer."
                )


    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
