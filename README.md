# Code Doc Generator

Turns a 5-minute "ugh, I need to write docstrings before this PR" chore into a 10-second paste.

**[Live demo →](STREAMLIT_URL_HERE)**

## Why this exists

Docstrings are the first thing to get skipped under deadline pressure, and the first
thing a new engineer needs when ramping up on unfamiliar code. This tool takes a raw
Python function and returns a Google-style docstring, plain-English parameter notes,
a runnable usage example, and a one-line complexity note — so "we'll document it
later" stops being the default answer.

## How it works

A two-stage LLM pipeline, not one big prompt:

1. **Extract** — function calling forces the model to return a structured signature
   (name, params, types, return type) with no room to hallucinate structure.
2. **Generate** — JSON mode + a few-shot example produces the docstring and metadata,
   validated against a schema with automatic retry on malformed output.

Splitting extraction from generation means a bad signature can't silently corrupt the
docstring — each stage is checked independently before the next one runs.

## Stack

Streamlit · Groq (`openai/gpt-oss-120b`) · Pydantic for schema validation

## Run it locally

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
streamlit run app.py
```