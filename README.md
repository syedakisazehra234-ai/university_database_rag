# 🎓 University Academic Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application for university
students and academic information.

The application uses:

- Streamlit for the web interface
- FAISS for vector similarity search
- Sentence Transformers for embeddings
- Groq for LLM generation
- OpenAI GPT-OSS 120B through Groq
- Pre-built embeddings and metadata for source traceability

---

## 🧠 Architecture

The project uses an offline indexing pipeline and a lightweight
runtime RAG application.

### Offline indexing

```text
Google Drive
     │
     ▼
PDF documents
     │
     ▼
PyMuPDF
     │
     ▼
Text extraction
     │
     ▼
Page-aware chunking
     │
     ▼
Sentence Transformers
     │
     ▼
Embeddings
     │
     ▼
FAISS
     │
     ├── index.faiss
     │
     └── metadata.json
````

The indexing process is performed once in Google Colab.

The original PDF documents do NOT need to be uploaded to GitHub.

---

## 🚀 Runtime architecture

When a student uses the application:

```text
Student question
       │
       ▼
Sentence Transformer
       │
       ▼
Question embedding
       │
       ▼
FAISS similarity search
       │
       ▼
Relevant chunks
       │
       ▼
Metadata
       │
       ├── filename
       ├── page number
       ├── chunk ID
       └── source text
       │
       ▼
Groq
       │
       ▼
GPT-OSS 120B
       │
       ▼
Grounded academic answer
       │
       ▼
Source references
```

The application does NOT recreate document embeddings when it
starts.

---

## 📁 Project structure

```text
university-academic-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── rag_database/
    │
    ├── config.json
    ├── metadata.json
    │
    └── faiss_index/
        └── index.faiss
```

---

## 📚 Knowledge base

The knowledge base consists of six PDF documents.

The PDFs are processed in Google Colab before deployment.

Each document is:

1. Read page by page.
2. Converted into text.
3. Split into overlapping chunks.
4. Converted into vector embeddings.
5. Stored in FAISS.
6. Associated with source metadata.

---

## 🔎 Source traceability

Each indexed chunk contains metadata such as:

* source filename
* document number
* page number
* page index
* chunk ID
* chunk number
* SHA-256 document fingerprint
* original chunk text
* word count
* character count

FAISS vector IDs correspond directly to metadata positions.

Example:

```text
FAISS vector 0
       │
       ▼
metadata["chunks"][0]

FAISS vector 1
       │
       ▼
metadata["chunks"][1]
```

This allows the application to identify the original PDF and
page associated with retrieved information.

---

## 🤖 LLM

The application uses:

```text
openai/gpt-oss-120b
```

through the Groq API.

The LLM is used only after relevant information has been retrieved
from the FAISS knowledge base.

---

## 🔐 Environment variable

The Groq API key is NOT stored in the source code.

The application expects:

```text
GROQ_API_KEY
```

---

## ☁️ Streamlit Cloud deployment

### Step 1 — Create GitHub repository

Create a new GitHub repository, for example:

```text
university-academic-rag
```

---

### Step 2 — Upload files

Upload:

```text
app.py
requirements.txt
README.md
.gitignore
```

and the complete:

```text
rag_database/
```

directory.

Do NOT upload the original PDF files.

---

### Step 3 — Check repository

Your repository should look like:

```text
university-academic-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── rag_database/
    ├── config.json
    ├── metadata.json
    └── faiss_index/
        └── index.faiss
```

---

### Step 4 — Deploy on Streamlit Cloud

Create a new Streamlit application and select:

```text
Repository:
your GitHub repository

Main file:
app.py
```

---

### Step 5 — Add Groq API key

In Streamlit Cloud application settings, add:

```text
GROQ_API_KEY
```

with your Groq API key as the secret value.

Do NOT put the key inside:

```text
app.py
```

and do NOT upload it to GitHub.

---

## ⚙️ Configuration

The `config.json` file contains information generated during
the indexing process, including:

```json
{
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "faiss_index_type": "IndexFlatIP",
    "similarity": "cosine",
    "chunk_size_words": 350,
    "chunk_overlap_words": 60,
    "default_top_k": 5
}
```

The Streamlit application reads this configuration instead of
hard-coding the embedding dimension.

---

## 🔄 Updating the knowledge base

If the university documents change:

1. Replace/update the PDFs in Google Drive.
2. Run the Colab indexing pipeline again.
3. Generate a new FAISS database.
4. Replace:

```text
rag_database/
```

in GitHub.
5. Streamlit Cloud will redeploy the application.

The application itself does not need to download the PDFs.

---

## 🛡️ Hallucination control

The assistant is instructed to:

* use the retrieved knowledge-base context
* avoid unsupported claims
* state when information is not found
* preserve important conditions and exceptions
* identify conflicting source information
* provide source identifiers

The retrieved source metadata is also displayed separately
from the generated answer.

---

## 📦 Technology stack

| Component           | Technology            |
| ------------------- | --------------------- |
| UI                  | Streamlit             |
| Vector database     | FAISS                 |
| Embeddings          | Sentence Transformers |
| LLM                 | OpenAI GPT-OSS 120B   |
| LLM provider        | Groq                  |
| Document processing | PyMuPDF               |
| Offline indexing    | Google Colab          |
| Source storage      | GitHub                |
| Deployment          | Streamlit Cloud       |

---

## 🔒 Data architecture

The original academic PDFs are not included in the deployed
application repository.

Only the processed RAG database is deployed:

```text
PDF
 ↓
Text
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS
 +
Metadata
```

This makes the runtime application independent of the original
Google Drive folder.

---

## ⚠️ Important

This application answers questions based on the documents
available in its knowledge base.

If information is not contained in the retrieved academic
documents, the assistant should say so rather than inventing
an answer.

Always verify important academic or institutional decisions
against the university's official documents.

```
```
