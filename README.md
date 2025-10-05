# 🎓 Academic Advising Chatbot (RAG-Powered with Ollama)--

An intelligent **Retrieval-Augmented Generation (RAG)**–based chatbot designed to assist students with **academic advising**, built using **LangChain**, **Ollama**, **Pinecone**, and **Streamlit**.  
This chatbot provides personalized, context-aware answers by retrieving relevant academic materials and combining them with generative reasoning — offering an efficient solution for educational institutions to enhance student support.



## 📖 Project Overview
The **Academic Advising Chatbot** acts as a virtual academic assistant for universities and colleges.  
It uses a **Retrieval-Augmented Generation (RAG)** framework to answer student queries related to courses, academic policies, deadlines, or faculty information.  

By integrating **Ollama LLM** (local language model inference) with **LangChain pipelines** and **Pinecone vector storage**, the chatbot retrieves the most relevant academic data and formulates meaningful responses.  

This system bridges the communication gap between students and advisors, providing 24/7 access to academic information in a conversational format.

---

## 🎯 Objectives
-  Enable intelligent academic query resolution using LLMs.  
-  Retrieve precise, context-aware answers from institutional data.  
-  Provide instant and accurate guidance through a Streamlit interface.  
-  Securely manage local embeddings and private documents using Pinecone.  
-  Demonstrate integration of **RAG + Ollama + LangChain + Streamlit** for real-world applications.

---

##  **System Architecture**


Explanation:
User enters a question through the Streamlit web UI.
LangChain processes the query and sends it to both:
The Retriever (Pinecone) for semantic search.
The LLM (Ollama) for context generation.
The retriever fetches the most relevant document chunks from the knowledge base (e.g., academic policy documents, FAQs).
Ollama LLM combines retrieved context with reasoning to generate a final answer.
The answer is displayed in the Streamlit chat interface.

🛠️ **Technology Stack**
Component	Technology Used	Description
 LLM	Ollama (Local Model)	Handles natural language generation
 Framework	LangChain	Connects retriever and generator through chains
 Vector Database	Pinecone	Stores and retrieves text embeddings
 Embeddings	HuggingFace Embeddings	Converts text into vector representations
 Frontend	Streamlit	Interactive web-based chat interface
 Backend	Python 3.10+	Core development language
 Configuration	dotenv	Manages environment variables and API keys


⚙️ **Setup and Installation**

- Clone the Repository
git clone https://github.com/vishalgoud483-tech/Academic-Advising-Chatbot.git
cd Academic-Advising-Chatbot

- **Create and Activate a Virtual Environment**
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux

- **Install Dependencies**
pip install -r requirements.txt

- **Set Up Environment Variables**
Create a .env file in the root folder and add:
PINECONE_API_KEY=your_pinecone_api_key
OLLAMA_MODEL=llama2      

- **Run the Streamlit App**
streamlit run src/streamlitMain.py
Then open your browser at http://localhost:8501/ 🚀

🧠 **Working Mechanism (RAG Workflow)**
Stage	Description
1. Document Loading	Loads .txt materials from the /materials folder.
2. Text Splitting	Splits large documents into manageable chunks.
3. Embedding Creation	Converts text chunks into numerical vectors using HuggingFace.
4. Vector Storage	Saves embeddings in Pinecone for quick similarity searches.
5. Query Retrieval	When a user asks something, the system retrieves the most similar chunks.
6. Contextual Response Generation	Ollama LLM combines retrieved info and generates a conversational answer.

💻 **Streamlit Interface**
A clean, minimal, and responsive chat interface that allows:
Real-time query entry
Contextual response display
Smooth scrolling and conversation history
Lightweight and fast load performance



💬 **Example Query Flow**
User:
“Can I drop a course after the registration deadline?”
Chatbot Response:
“According to the university’s academic policy, course withdrawal after the registration deadline requires approval from your academic advisor. Late withdrawal may affect your GPA and transcript record.”

🔮 **Future Enhancements**
 Add multilingual support (via additional Ollama models).
 Integrate student timetable and course registration APIs.
 Voice-to-text interaction support.
 Add analytics dashboard for admin insights.
 Deploy on cloud (AWS / Streamlit Cloud) with scalable Pinecone hosting.

👥**Contributors**
Developed by Team Academic Advisors

- Mothe Vishal Goud (111851139)
- Manohar Mahavir Gourai (111796149)
- Mahesh Babu Davala  (111869085)

🏁 **Conclusion**
The Academic Advising Chatbot demonstrates how Retrieval-Augmented Generation (RAG) can revolutionize student support systems by offering quick, reliable, and personalized academic guidance.
It’s a practical implementation of AI-driven education systems, combining Ollama’s local LLM inference with LangChain’s flexible orchestration — all wrapped in an easy-to-use Streamlit application.

⭐ If you found this project helpful, please give it a star on GitHub!
