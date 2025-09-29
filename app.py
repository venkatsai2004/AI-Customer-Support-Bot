import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd
import tempfile
import logging
from typing import Optional, List
import random

# Hugging Face Spaces compatible imports
try:
    import pdfplumber
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
    from langchain.schema import Document
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Customer Support Bot",
    page_icon="🤖",
    layout="wide"
)

# Simplified SupportBotAgent for HF Spaces
class SupportBotAgent:
    def __init__(self, document_content: str, document_name: str = "document"):
        self.document_name = document_name
        self.similarity_threshold = 0.8
        self.qa_confidence_threshold = 0.1
        self.max_context_length = 512
        try:
            # Initialize embeddings with caching for HF Spaces
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./models"  
            )
            self.vectorstore = self._process_document(document_content)
            self.qa_pipeline = self._setup_qa_chain()
            st.success(f"Bot initialized with {document_name}")
        except Exception as e:
            st.error(f"Error initializing bot: {str(e)}")
            raise

    def _process_document(self, content: str):
        # Split content into chunks
        texts = [section.strip() for section in content.split("\n\n") if section.strip()]
        if not texts:
            texts = [line.strip() for line in content.split("\n") if line.strip()]

        documents = [Document(page_content=text) for text in texts if text.strip()]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        return vectorstore

    def _setup_qa_chain(self):
        try:
            model_name = "distilbert-base-uncased-distilled-squad"
            qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                cache_dir="./models"
            )
            return qa_pipeline
        except Exception as e:
            st.warning(f"QA pipeline setup failed: {e}")
            return None

    def answer_query(self, query: str) -> dict:
        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=3)
            
            if not docs_and_scores:
                return {
                    "answer": "I don't have enough information to answer that question.",
                    "context_used": None
                }
            
            best_doc, best_score = docs_and_scores[0]
            
            if best_score > self.similarity_threshold:
                return {
                    "answer": f"I don't have specific information about '{query}' in my knowledge base.",
                    "context_used": None
                }
            
            # Prepare context
            context = best_doc.page_content
            
            # Add additional context from other relevant chunks
            if len(docs_and_scores) > 1:
                for doc, score in docs_and_scores[1:3]:
                    if score < self.similarity_threshold * 1.2:
                        context += " " + doc.page_content
            
            # Limit context length
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length]
            
            # Try QA pipeline
            if self.qa_pipeline:
                try:
                    qa_result = self.qa_pipeline(question=query, context=context)
                    if qa_result["score"] > self.qa_confidence_threshold and qa_result["answer"].strip():
                        answer = qa_result["answer"].strip()
                    else:
                        answer = context.strip()
                except:
                    answer = context.strip()
            else:
                answer = context.strip()
            
            return {
                "answer": answer,
                "context_used": context
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "context_used": None
            }

    def get_feedback(self, answer: str) -> str:
        if not isinstance(answer, str) or not answer.strip():
            return "not helpful"
        
        answer = answer.strip()
        
        if "I don't have" in answer or "Error" in answer:
            return "not helpful"
        elif len(answer) < 30:
            return "too vague"
        elif len(answer) > 200:
            return random.choice(["good", "too vague"])
        else:
            return random.choice(["good", "good", "too vague", "not helpful"])

    def adjust_response(self, query: str, response: dict, feedback: str) -> dict:
        try:
            if feedback == "too vague" and response["context_used"]:
                if "Additional Info:" not in response["answer"]:
                    docs = self.vectorstore.similarity_search(query, k=2)
                    extra_context = "\n".join([doc.page_content for doc in docs[:1]])
                    response["answer"] += f"\n\nAdditional Info:\n{extra_context[:150]}..."
            
            elif feedback == "not helpful":
                rephrased_query = f"Please explain in detail: {query}"
                return self.answer_query(rephrased_query)
            
            return response
        except:
            return response

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Default sample documents
SAMPLE_DOCUMENTS = {
"Customer Support FAQ": """
Resetting Your Password
To reset your password, go to the login page and click "Forgot Password." Enter your email address and follow the link sent to your email inbox.

Refund Policy
We offer full refunds within 30 days of purchase for any reason. To request a refund, contact our support team at support@example.com with your order number.

Contacting Support
Our support team is available to help you with any questions. Email us at support@example.com or call 1-800-555-1234 during business hours (9 AM - 5 PM EST).

Account Management
You can update your account information by logging into your account dashboard. From there, you can change your email, update billing information, and manage preferences.

Technical Support
If you're experiencing technical issues, first try clearing your browser cache and cookies. Make sure you're using a supported browser (Chrome, Firefox, Safari, or Edge).
""",
    
"API Documentation": """
API Documentation
Our REST API allows developers to integrate with our platform. Use the base URL https://api.example.com/v1/ for all requests.

Rate Limiting
API calls are limited to 1000 requests per hour per API key. Exceeded limits return a 429 status code.

Error Handling
API errors return standard HTTP status codes. 400 for bad requests, 401 for unauthorized, 404 for not found, and 500 for server errors.

SDK Support
We provide official SDKs for Python, JavaScript, and Java. Community-maintained SDKs are available for other languages.

Webhooks
Set up webhooks to receive real-time notifications about events. Configure webhook URLs in your dashboard.
""",

"Product Guide": """
Getting Started
Welcome to our platform! This guide will help you get started quickly and efficiently.

Installation
Download the software from our website and run the installer. Follow the on-screen instructions to complete the setup.

Basic Features
Our platform includes document management, collaboration tools, and automated workflows. Navigate using the main menu on the left side.

Advanced Features
Power users can access advanced features through the settings menu. This includes API access, custom integrations, and bulk operations.

Troubleshooting
Common issues include login problems, slow performance, and sync errors. Most issues can be resolved by refreshing your browser or clearing the cache.
"""
}

def main():
    st.title("🤖 AI Customer Support Bot")
    st.markdown("### Upload a document or select a sample to get started!")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Document selection
        st.subheader("Choose Document Source")
        
        option = st.radio(
            "Select input method:",
            ["Sample Documents", "Upload File", "Paste Text"]
        )
        
        document_content = None
        document_name = None
        
        if option == "Sample Documents":
            selected_doc = st.selectbox("Choose a sample:", list(SAMPLE_DOCUMENTS.keys()))
            if st.button("Load Sample Document"):
                document_content = SAMPLE_DOCUMENTS[selected_doc]
                document_name = selected_doc
        
        elif option == "Upload File":
            uploaded_file = st.file_uploader("Upload a text or PDF file", type=['txt', 'pdf'])
            if uploaded_file and st.button("Process Upload"):
                if uploaded_file.type == "application/pdf":
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            with pdfplumber.open(tmp_file.name) as pdf:
                                document_content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                        document_name = uploaded_file.name
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                else:
                    document_content = uploaded_file.read().decode('utf-8')
                    document_name = uploaded_file.name
        
        elif option == "Paste Text":
            pasted_text = st.text_area("Paste your document content here:", height=200)
            if pasted_text and st.button("Process Text"):
                document_content = pasted_text
                document_name = "Pasted Document"
        
        # Initialize bot if we have content
        if document_content and document_name:
            try:
                with st.spinner("Initializing AI models..."):
                    st.session_state.bot = SupportBotAgent(document_content, document_name)
            except Exception as e:
                st.error(f"Failed to initialize bot: {e}")
        
        # Bot status
        st.subheader("Bot Status")
        if st.session_state.bot:
            st.success("✅ Bot is ready!")
        else:
            st.warning("⚠️ Please load a document first")
        
        # Clear history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat Interface")
        
        # Sample questions
        if st.session_state.bot:
            st.subheader("Try these example questions:")
            examples = [
                "How do I reset my password?",
                "What's the refund policy?",
                "How do I contact support?",
                "What are the API rate limits?",
                "How do I get started?"
            ]
            
            cols = st.columns(len(examples))
            for i, example in enumerate(examples):
                if cols[i].button(example, key=f"example_{i}"):
                    st.session_state.current_query = example
        
        # Query input
        query = st.text_input(
            "Ask your question:",
            value=st.session_state.get('current_query', ''),
            placeholder="Type your question here...",
            disabled=st.session_state.bot is None
        )
        
        if st.button("Ask Question", disabled=not query or st.session_state.bot is None):
            with st.spinner("Processing your question..."):
                start_time = time.time()
                
                # Get response with feedback loop
                response = st.session_state.bot.answer_query(query)
                
                for _ in range(2):  # Max 2 feedback iterations
                    feedback = st.session_state.bot.get_feedback(response['answer'])
                    if feedback == "good":
                        break
                    response = st.session_state.bot.adjust_response(query, response, feedback)
                
                processing_time = time.time() - start_time
                
                # Add to history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'response': response['answer'],
                    'processing_time': processing_time
                })
                
                # Clear the input
                st.session_state.current_query = ""
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['query'][:50]}...", expanded=i==0):
                    st.write(f"**Question:** {chat['query']}")
                    st.write(f"**Answer:** {chat['response']}")
                    st.caption(f"Response time: {chat['processing_time']:.2f}s | {chat['timestamp'].strftime('%H:%M:%S')}")

    with col2:
        st.header("Statistics")
        
        if st.session_state.chat_history:
            total_queries = len(st.session_state.chat_history)
            avg_time = sum(chat['processing_time'] for chat in st.session_state.chat_history) / total_queries
            
            st.metric("Total Questions", total_queries)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            # Recent queries
            st.subheader("Recent Questions")
            for chat in st.session_state.chat_history[-3:]:
                st.write(f"• {chat['query'][:40]}...")
        else:
            st.info("No conversations yet. Ask a question to get started!")
        
        st.subheader("Model Information")
        st.write("**Embedding:** all-MiniLM-L6-v2")
        st.write("**QA Model:** DistilBERT")
        st.write("**Vector Store:** FAISS")

if __name__ == "__main__":
    main()