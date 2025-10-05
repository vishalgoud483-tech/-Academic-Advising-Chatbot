# streamlitMain.py
import streamlit as st
from pathlib import Path
from RAG_ChatBot import ChatBot, PROJECT_ROOT  # reuse resolved root

st.set_page_config(page_title="🎓 Academic Advising RAG Bot", page_icon="🎓", layout="wide")

@st.cache_resource(show_spinner=True)
def get_bot():
    materials = PROJECT_ROOT / "materials"         # always project-root/materials
    index_dir = PROJECT_ROOT / ".faiss_index"      # always project-root/.faiss_index
    return ChatBot(materials_dir=materials, index_dir=index_dir)

bot = get_bot()

st.title("🎓 Academic Advising Chatbot")
st.write(
    "Ask questions about programs, courses, prerequisites, instructors, time & location, and graduation requirements. "
)

with st.sidebar:
    st.header("Quick Prompts")
    examples = [
        "What are the core courses in [program, e.g., Marketing]?",
        "What are the graduation requirements for [degree program, e.g., Accounting]?",
        "What is the course information for [course name or ID]?",
        "What courses should a freshman in [degree program, e.g., Marketing] take in Semester 1?",
        "What is the prerequisite for [course name or ID]?",
        "What courses does Ziyi Wang teach in Fall 2025, and what is his email address?",
        "What are the time and location details for [course name or ID] in Fall 2025?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex

query = st.text_input("Your question", value=st.session_state.get("q", ""))

if st.button("Ask") or query:
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching materials and generating an answer..."):
            result = bot.ask(query)

        st.subheader("Answer")
        st.write(result["answer"] or "I couldn't find that in the provided materials.")

        st.subheader("Sources")
        if result["sources"]:
            for i, s in enumerate(result["sources"], start=1):
                with st.expander(f"Source {i}: {s['source']}"):
                    st.write(s["snippet"])
        else:
            st.write("No specific source snippets found.")

st.caption(f"Expecting .docx files in: {(PROJECT_ROOT / 'materials').resolve()}")
