import logging
import os
from typing import Optional, List
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain.schema import Document  
import random

# Set up logging
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("log", "support_bot_log.txt"),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#we have used the from langchain.schema import Document  so there is no need to redefine the Document class
# class Document:
#     def __init__(self, page_content: str, metadata: Optional[dict] = None):
#         self.page_content = page_content
#         self.metadata = metadata or {}


class SupportBotAgent:
    def __init__(self, document_path: str):
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"The file {document_path} does not exist.")
        self.document_path = document_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = self._load_and_process_document()
        self.qa_pipeline = self._setup_qa_chain()
        logging.info(f"Initialized bot with document: {document_path}")

    def _load_and_process_document(self):
        try:
            if self.document_path.endswith('.pdf'):
                with pdfplumber.open(self.document_path) as pdf:
                    texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
            else:
                with open(self.document_path, 'r', encoding='utf-8') as file:
                    texts = file.read().split("\n\n")  # split by double newlines

            documents = [Document(page_content=text) for text in texts if text.strip()]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)

            vectorstore = FAISS.from_documents(texts, self.embeddings)
            logging.info(f"Processed document into {len(texts)} chunks.")
            return vectorstore
        except Exception as e:
            logging.error(f"Error loading document: {e}")
            raise
    def _setup_qa_chain(self):
        """Set up a custom QA chain using a pre-trained model."""
        try:
            model_name = "distilbert-base-uncased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
            )
            logging.info(f"Loaded QA model: {model_name}")
            return self.qa_pipeline
        except Exception as e:
            logging.error(f"Error setting up QA chain: {e}")
            raise

    def _is_query_covered(self, query: str) -> bool:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=1)
        if not docs_and_scores:
            return False
        _, score = docs_and_scores[0]
        return score < 1.0


    def answer_query(self, query: str) -> dict:
        """Retrieve relevant documents and answer the query."""
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=1)
        if not docs_and_scores:
            return {
                "answer": "I don't have enough information to answer that question.",
                "context_used": None
            }
        doc, score = docs_and_scores[0]
        if score < 1.0:
            return {
                "answer": doc.page_content.strip(),
                "context_used": doc.page_content
            }
        else:
            return {
                "answer": f"There is no information about '{query}'.",
                "context_used": None
            }


    def get_feedback(self, answer: str) -> str:
        """Simulate feedback for the response."""
        if not isinstance(answer, str):
            answer = "No answer provided."
        feedback = random.choice(["not helpful", "too vague", "good"])
        logging.debug(f"Feedback for response '{answer[:50]}...': {feedback}")
        return feedback



    def adjust_response(self, query: str, response: dict, feedback: str) -> dict:
        """Adjust the response based on feedback."""
        if feedback == "too vague":
            if "Additional Info:" in response["answer"]:
                return response  

            docs = self.vectorstore.similarity_search(query, k=3)
            extra_context = "\n".join([doc.page_content for doc in docs[1:]])
            if extra_context:
                if response["context_used"] is None:
                    response["context_used"] = extra_context
                else:
                    response["context_used"] += "\n\n" + extra_context

                response["answer"] += f"\n\nAdditional Info:\n{extra_context[:200]}..."

        elif feedback == "not helpful":
            return self.answer_query(f"Can you clarify the answer to: {query}?")

        return response


    
    def run(self, queries: List[str]):
        """Run the bot for a list of queries."""
        for query in queries:
            logging.info(f"Processing query: {query}")
            response = self.answer_query(query)
            for _ in range(2):
                feedback = self.get_feedback(response)
                if feedback == "good":
                    break
                response = self.adjust_response(query, response, feedback)
            print(f"Final Response to '{query}': {response['answer']}")


if __name__ == "__main__":
    bot = SupportBotAgent("faq.txt")
    sample_queries = [
        "How do I reset my password?",
        "What's the refund policy?",
        "How do I fly to the moon?"  # Out-of-scope query
    ]
    bot.run(sample_queries)