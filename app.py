import os
import streamlit as st

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from script import document_code

st.set_page_config(page_title="Code Doc Generator", page_icon="📝", layout="centered")

st.title("📝 Code Doc Generator")
st.caption(
    "Paste a Python function, get a Google-style docstring, parameter notes, "
    "a usage example, and a complexity note -- in about 10 seconds."
)

code_input = st.text_area(
    "Paste a Python function",
    height=220,
    placeholder=(
        "def calculate_discount(price, percentage=10):\n"
        "    return price - (price * percentage / 100)"
    ),
)

generate = st.button(
    "Generate documentation", type="primary", disabled=not code_input.strip()
)

if generate:
    with st.spinner("Extracting signature, then writing docs..."):
        try:
            result = document_code(code_input)
        except Exception as e:
            st.error(f"Couldn't generate docs for that input: {e}")
            st.stop()

    docs = result["documentation"]
    sig = result["signature"]

    st.subheader(f"`{sig['name']}()`")

    st.markdown("**Docstring** — hover the block, click the copy icon")
    st.code(docs["docstring"], language="python")

    st.markdown("**Parameter notes**")
    for note in docs["parameter_notes"]:
        st.markdown(f"- {note}")

    st.markdown("**Example usage**")
    st.code(docs["example_usage"], language="python")

    st.markdown("**Complexity**")
    st.info(docs["complexity_note"])

    st.caption(f"Cost for this run: ${result['total_cost_usd']:.6f}")

st.divider()
st.caption(
    "Built with Groq (openai/gpt-oss-120b) · function calling for extraction, "
    "JSON mode for generation, Pydantic-validated end to end."
)