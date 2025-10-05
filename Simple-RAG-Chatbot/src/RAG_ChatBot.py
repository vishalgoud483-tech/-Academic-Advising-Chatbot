# src/RAG_ChatBot.py
from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Robust imports: prefer community modules, fall back to core langchain
try:
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain_community.document_loaders import PyPDFLoader as CommunityPyPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityHuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS as CommunityFAISS
    from langchain_community.chat_models import ChatOllama
except Exception:
    Docx2txtLoader = None
    CommunityPyPDFLoader = None
    CommunityHuggingFaceEmbeddings = None
    CommunityFAISS = None
    ChatOllama = None

# Core langchain fallbacks
try:
    from langchain.document_loaders import Docx2txtLoader as CoreDocxLoader
except Exception:
    CoreDocxLoader = None

try:
    from langchain.document_loaders import PyPDFLoader as CorePyPDFLoader
except Exception:
    CorePyPDFLoader = None

try:
    from langchain.embeddings import HuggingFaceEmbeddings as CoreHuggingFaceEmbeddings
except Exception:
    CoreHuggingFaceEmbeddings = None

try:
    from langchain.vectorstores import FAISS as CoreFAISS
except Exception:
    CoreFAISS = None

# OpenAI chat wrapper fallback
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# Final selected classes
Docx2txtLoader = Docx2txtLoader or CoreDocxLoader
PDF_LOADER_CLASS = CommunityPyPDFLoader or CorePyPDFLoader
HuggingFaceEmbeddings = CommunityHuggingFaceEmbeddings or CoreHuggingFaceEmbeddings
FAISS = CommunityFAISS or CoreFAISS

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------- Paths & prompts ----------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
MATERIALS_DIR = PROJECT_ROOT / "materials"
INDEX_DIR = PROJECT_ROOT / ".faiss_index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

SYSTEM_PROMPT = """
You are an intelligent Academic Advising Assistant for a university business school. 
You have access to internal program documents, course catalogs, instructor directories, 
degree requirements, and scheduling files stored in the 'materials/' folder.

Your mission:
- Provide **accurate, concise, and directly relevant answers** about programs, courses, 
  instructors, schedules, and requirements.
- Use **ONLY** the information found in the provided context. 
  If the answer is not explicitly in the context, infer it **cautiously** from related details,
  or say that it is not available — never make up facts.
- Always identify and summarize the most relevant section(s) when a direct match is missing.

---

### 🎯 You can answer questions such as:
1. Core or required courses in a specific degree program (e.g., Marketing, Accounting, Finance)
2. Graduation or degree completion requirements
3. Course details — ID, title, credits, prerequisites, and description
4. Instructor names, emails, and which courses they teach
5. Course times, locations, and terms offered (e.g., Fall 2025)
6. Recommended courses by semester (e.g., “freshman semester 1”)
7. Any other factual details that exist in the materials folder

---

### 🧠 Reasoning Steps:
1. Read the question carefully.
2. Review the provided context and locate all directly relevant facts.
3. If multiple relevant entries exist, combine them clearly and avoid repetition.
4. If exact data is not found, respond with the **closest matching information** and 
   reference the related document or section title.

---

### 📝 Answer Format Rules:
- If the question requests a list, use bullet points.
- If about a specific course, format as:
  `CourseID — Course Title (Credits): Prerequisite(s) — Short Description`
- If about schedules, use:
  `CourseID — Day/Time — Location — Instructor`
- If about graduation or program requirements, use bullets with bold keywords:
  `• **Credits:** 120 minimum`
  `• **GPA Requirement:** 2.0 cumulative`
- End with “*(Source: <filename>)*” if possible.
- Be professional, clear, and factual — never casual or uncertain.

---

### 🚫 If the answer cannot be found:
Respond:
> "I could not find the exact answer in the provided materials. However, here is the closest related information I located: ..."

---

### ⚙️ Output Style:
Use **plain text**, **no markdown formatting** other than bullet points and bold.
Be concise: 3–6 sentences max, or a short bullet list if appropriate.

---

Now, answer the user’s question based on the following context.
"""


QA_PROMPT = PromptTemplate.from_template(
    "{system}\n\n---\n\nContext:\n{context}\n\n---\n\nUser Question: {question}\n\nAnswer:"
).partial(system=SYSTEM_PROMPT)


# ---------------- Utilities ----------------
def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.name.encode())
    h.update(str(path.stat().st_mtime_ns).encode())
    return h.hexdigest()[:16]


def _materials_signature(files: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: x.name.lower()):
        h.update(_file_fingerprint(p).encode())
    return h.hexdigest()


def _parse_filename_metadata(f: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    name = f.stem.lower()
    parts = name.split("_")
    if parts and parts[0].isalpha():
        meta["program"] = parts[0].title()
    for p in parts:
        if p.startswith("sem") and any(ch.isdigit() for ch in p):
            digits = "".join([c for c in p if c.isdigit()])
            try:
                meta["semester"] = int(digits)
            except Exception:
                pass
        if "require" in p or "gradu" in p:
            meta["type"] = "requirements"
        if "schedule" in p or "fall" in p or "spring" in p or "term" in p:
            meta["type"] = "schedule"
        if "catalog" in p or "course" in p:
            meta["type"] = "catalog"
    return meta


def _prefix_from_metadata(meta: Dict[str, Any]) -> str:
    parts = []
    if meta.get("program"):
        parts.append(f"[PROGRAM: {meta['program']}]")
    if meta.get("semester"):
        parts.append(f"[SEM: {meta['semester']}]")
    if meta.get("type"):
        parts.append(f"[TYPE: {meta['type']}]")
    return " ".join(parts) + " " if parts else ""


# ---------------- ChatBot ----------------
class ChatBot:
    def __init__(
            self,
            materials_dir: Optional[Path | str] = None,
            index_dir: Optional[Path | str] = None,
            k: int = 15,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        materials_dir: optional override path to materials folder
        index_dir: optional override path to index
        k: how many chunks to retrieve
        """
        load_dotenv()
        self.k = k
        self.embedding_model = embedding_model

        # resolve paths
        self.materials_dir = Path(materials_dir).resolve() if materials_dir else MATERIALS_DIR
        self.index_dir = Path(index_dir).resolve() if index_dir else INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # ensure embedding implementation available
        if HuggingFaceEmbeddings is None:
            raise RuntimeError("HuggingFaceEmbeddings not available. Install sentence-transformers and langchain embeddings.")
        self.embedder = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # load docs and index
        self.docs = self._load_documents()
        self.db = self._ensure_index(self.docs)
        # retriever from vectorstore
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": self.k})

        # LLM selection and LLMChain
        self.llm = self._make_llm()
        self.llm_chain = LLMChain(llm=self.llm, prompt=QA_PROMPT)

    # ---------- Public ask ----------
    def ask(self, query: str, program: Optional[str] = None, semester: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the system. Optionally bias by program and semester.
        Returns dict with 'answer' and 'sources'.
        """
        bias_prefix = ""
        if program:
            bias_prefix += f"[PROGRAM: {program.title()}] "
        if semester:
            bias_prefix += f"[SEM: {semester}] "
        full_query = bias_prefix + query

        # Retrieve top documents
        docs = self.retriever.get_relevant_documents(full_query)
        if not docs:
            return {"answer": "No relevant documents found in materials.", "sources": []}

        # Build context: include source metadata and content (limit length)
        context_pieces = []
        for d in docs[: self.k]:
            src = d.metadata.get("source") if d.metadata else None
            head = f"Source: {src}\n" if src else ""
            # keep a reasonable snippet per doc
            snippet = d.page_content or ""
            # limit snippet length to avoid too-long prompts
            if len(snippet) > 2000:
                snippet = snippet[:2000] + "..."
            context_pieces.append(head + snippet)
        context = "\n\n---\n\n".join(context_pieces)

        # call LLMChain (pass context + original question)
        try:
            answer = self.llm_chain.run({"context": context, "question": query})
        except Exception:
            # some LLMChain versions accept positional args; try fallback
            try:
                answer = self.llm_chain.run(f"Context:\n{context}\n\nQuestion: {query}")
            except Exception as e:
                answer = f"LLM error: {e}"

        # gather sources
        sources = []
        for d in docs[: self.k]:
            sources.append({"source": d.metadata.get("source") if d.metadata else None, "snippet": (d.page_content or "")[:400]})

        # debug output (console)
        print("\n🔎 Retrieved sources (top):")
        for i, s in enumerate(sources[:6], start=1):
            print(f"{i}. {s['source']}")
            print(s["snippet"][:250].replace("\n", " "), "\n")

        return {"answer": answer.strip(), "sources": sources}

    # ---------- Internal: list files ----------
    def _list_files(self) -> List[Path]:
        if not self.materials_dir.exists():
            raise FileNotFoundError(f"Materials folder not found: {self.materials_dir.resolve()}")
        patterns = ["*.docx", "*.pdf", "*.xlsx", "*.xls"]
        files: List[Path] = []
        for pat in patterns:
            files.extend([p for p in self.materials_dir.glob(pat) if not p.name.startswith("~$")])
        files = sorted(files, key=lambda p: p.name.lower())
        if not files:
            raise FileNotFoundError(f"No supported files (.docx/.pdf/.xlsx) found in {self.materials_dir.resolve()}")
        return files

    # ---------- Internal: load documents ----------
    def _load_documents(self) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        all_docs: List[Document] = []
        files = self._list_files()
        print(f"📚 Loading {len(files)} files from {self.materials_dir}")

        for f in files:
            suffix = f.suffix.lower()
            file_meta = _parse_filename_metadata(f)
            loaded: List[Document] = []
            try:
                if suffix == ".docx":
                    if Docx2txtLoader is None:
                        raise RuntimeError("Docx loader not available. Install langchain-community or python-docx.")
                    loader = Docx2txtLoader(str(f))
                    loaded = loader.load()
                elif suffix == ".pdf":
                    if PDF_LOADER_CLASS is None:
                        raise RuntimeError("PDF loader not available. Install langchain or langchain-community.")
                    loader = PDF_LOADER_CLASS(str(f))
                    loaded = loader.load()
                elif suffix in (".xlsx", ".xls"):
                    try:
                        import pandas as pd
                    except Exception:
                        raise RuntimeError("pandas required to read Excel files. Install pandas and openpyxl.")
                    sheets = pd.read_excel(str(f), sheet_name=None)
                    for sheet_name, df in sheets.items():
                        text = df.fillna("").astype(str).to_csv(index=False, sep="\t")
                        loaded.append(Document(page_content=f"Sheet: {sheet_name}\n\n{text}", metadata={"sheet": sheet_name}))
                else:
                    continue
            except Exception as e:
                print(f"⚠️ Failed to load {f.name}: {e}")
                continue

            # attach metadata and prefix content
            for d in loaded:
                if not d.metadata:
                    d.metadata = {}
                for k, v in file_meta.items():
                    if k not in d.metadata or not d.metadata.get(k):
                        d.metadata[k] = v
                d.metadata["source"] = str(f)
                prefix = _prefix_from_metadata(d.metadata)
                if prefix:
                    d = Document(page_content=prefix + (d.page_content or ""), metadata=d.metadata)
                # split into chunks
                try:
                    parts = splitter.split_documents([d])
                    all_docs.extend(parts)
                except Exception as e:
                    print(f"⚠️ Splitter failed for {f.name}: {e}")
                    all_docs.append(d)

        print(f"✅ Loaded and split into {len(all_docs)} chunks.")
        return all_docs

    # ---------- Internal: index ----------
    def _ensure_index(self, docs: List[Document]):
        files = self._list_files()
        sig = _materials_signature(files)
        if MANIFEST_PATH.exists():
            try:
                manifest = json.loads(MANIFEST_PATH.read_text())
                if manifest.get("signature") == sig:
                    print("🔍 Loading existing FAISS index...")
                    return FAISS.load_local(str(self.index_dir), self.embedder, allow_dangerous_deserialization=True)
            except Exception:
                pass  # rebuild

        print("⚙️ Building FAISS index from documents...")
        db = FAISS.from_documents(docs, self.embedder)
        db.save_local(str(self.index_dir))
        MANIFEST_PATH.write_text(json.dumps({"signature": sig}, indent=2))
        print("✅ FAISS index built and saved.")
        return db

    # ---------- Internal: LLM selection ----------
    def _make_llm(self):
        ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
        if ollama_model and ChatOllama is not None:
            print(f"🤖 Using Ollama local model: {ollama_model}")
            return ChatOllama(model=ollama_model, temperature=0)

        if ChatOpenAI is not None:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in .env")
            print("🔑 Using OpenAI model from env")
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

        raise RuntimeError("No LLM configured. Set OLLAMA_MODEL or OPENAI_API_KEY in .env")
