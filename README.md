# Triage

An AI agent that classifies incoming support tickets by priority, using
an LLM with retrieval-augmented context (RAG) and tool/function calling.

## What it does

1. **Classifies** each ticket as Critical / High / Medium / Low using an
   LLM prompted for strict JSON output.
2. **Retrieves** similar past resolved tickets using TF-IDF and cosine
   similarity (RAG), and feeds them into the classification prompt as context,
   so decisions are grounded with context.
3. **Escalates** automatically. For Critical/High tickets, a second
   tool-enabled LLM call lets the model itself decide whether to invoke
   an `escalate_ticket()` function or not.

## Why these design choices

- **Structured JSON output** instead of free text since a classifier's
  output needs to be machine-parseable to be useful in a real pipeline.
- **TF-IDF instead of a vector DB** was used since it has the same underlying idea as RAG
  (text to vectors and then to similarity search) without the setup overhead of a
  real vector database. A real system would swap this for embeddings along with
  something like Pinecone/pgvector.
- **Tool calling only for Critical/High tickets** — calling a
  tool-enabled model for every single ticket wastes latency and cost on
  routine tickets that never need escalation. Knowing when not to call
  a model is as much a design decision as knowing when to.

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add a free Groq API key
   (console.groq.com)
3. `python src/main.py`

## What I'd improve with more time

- Swap TF-IDF for real embeddings 
- Persist tickets in a real database instead of a JSON file
- Add retries/validation for cases where the model returns malformed JSON
- Wrap this in a small FastAPI endpoint instead of a CLI script
