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

class SupportBotAgent:
    def __init__(self, document_path: str):
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"The file {document_path} does not exist.")
        
        self.document_path = document_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = self._load_and_process_document()
        self.qa_pipeline = self._setup_qa_chain()
        self.similarity_threshold = 0.75
        self.qa_confidence_threshold = 0.05  
        self.max_context_length = 800 
        
        logging.info(f"Initialized bot with document: {document_path}")

    def _load_and_process_document(self):
        try:
            if self.document_path.endswith('.pdf'):
                with pdfplumber.open(self.document_path) as pdf:
                    texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
            else:
                with open(self.document_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    texts = [section.strip() for section in content.split("\n\n") if section.strip()]
                    if len(texts) <= 1:
                        texts = [line.strip() for line in content.split("\n") if line.strip()]

            documents = [Document(page_content=text) for text in texts if text.strip()]

           
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,  
                chunk_overlap=50,  
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            split_docs = text_splitter.split_documents(documents)

            vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            logging.info(f"Processed document into {len(split_docs)} chunks.")
            return vectorstore
        except Exception as e:
            logging.error(f"Error loading document: {e}")
            raise
    
    def _setup_qa_chain(self):
        """Set up an optimized QA chain."""
        try:
            model_name = "distilbert-base-uncased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
           
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                max_answer_len=150,  
                max_seq_len=512,     
                doc_stride=128,      
                max_question_len=64  
            )
            logging.info(f"Loaded QA model: {model_name}")
            return self.qa_pipeline
        except Exception as e:
            logging.error(f"Error setting up QA chain: {e}")
            raise

    def _is_query_covered(self, query: str) -> bool:
        """Check if query is covered by the document with optimized threshold."""
        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=1)
            if not docs_and_scores:
                return False
            _, score = docs_and_scores[0]
            is_covered = score < self.similarity_threshold
            logging.debug(f"Query coverage check: {query[:50]}... | Score: {score:.3f} | Covered: {is_covered}")
            return is_covered
        except Exception as e:
            logging.error(f"Error checking query coverage: {e}")
            return False

    def answer_query(self, query: str) -> dict:
        """Retrieve relevant documents and answer the query with improved logic."""
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
            
            context = best_doc.page_content
            
            if len(docs_and_scores) > 1:
                for doc, score in docs_and_scores[1:3]: 
                    if score < self.similarity_threshold * 1.2:  
                        context += " " + doc.page_content
            
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length]

            try:
                qa_result = self.qa_pipeline(question=query, context=context)
                
            
                if qa_result["score"] > self.qa_confidence_threshold and qa_result["answer"].strip():
                    answer = qa_result["answer"].strip()
                    logging.info(f"Successfully answered query: {query[:50]}...")
                else:
                    answer = context.strip()
                    logging.info(f"Used context fallback for query: {query[:50]}...")
                    
            except Exception as qa_error:
                logging.warning(f"QA pipeline failed, using context: {qa_error}")
                answer = context.strip()
            
            return {
                "answer": answer,
                "context_used": context
            }
            
        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "context_used": None
            }

    def get_feedback(self, answer: str) -> str:
        """Simulate feedback with improved logic."""
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
            if ("I don't have specific information" in response["answer"] or 
                "I apologize, but I cannot find" in response["answer"]):
                return response
            
            if feedback == "too vague":
                if "Additional Info:" not in response["answer"]:
                    docs = self.vectorstore.similarity_search(query, k=5)
                    extra_context_parts = []
                    
                    # Smart filtering based on query context
                    for doc in docs[1:4]:
                        content = doc.page_content.strip()
                        if len(content) > 20:
                            # For password reset queries, prioritize contact info
                            if "password" in query.lower() or "reset" in query.lower():
                                if "contact" in content.lower() or "support" in content.lower():
                                    extra_context_parts.append(content)
                            else:
                                extra_context_parts.append(content)
                    
                    if extra_context_parts:
                        extra_context = "\n".join(extra_context_parts[:1])  # Only 1 relevant chunk
                        response["answer"] += f"\n\nAdditional Info:\n{extra_context[:200]}..."

                        if response["context_used"]:
                            response["context_used"] += "\n\n" + extra_context
                        else:
                            response["context_used"] = extra_context

            elif feedback == "not helpful":
                rephrased_queries = [
                    f"Please explain: {query}",
                    f"Help with: {query}",
                    f"Information about: {query}",
                    f"Details on: {query}"
                ]
                for rephrased in rephrased_queries:
                    new_response = self.answer_query(rephrased)
                    if (new_response["answer"] != response["answer"] and 
                        "I don't have" not in new_response["answer"] and
                        "I apologize, but I cannot find" not in new_response["answer"]):
                        return new_response
                response["answer"] = (
                    f"I apologize, but I cannot find specific information about '{query}' "
                    f"in my current knowledge base. Please try rephrasing your question or "
                    f"ask about topics covered in my documentation."
                )

            return response
            
        except Exception as e:
            logging.error(f"Error adjusting response: {e}")
            return response

    def run(self, queries: List[str]):
        """Run the bot for a list of queries with improved workflow."""
        results = []
        
        for query in queries:
            logging.info(f"Processing query: {query}")
            response = self.answer_query(query)
            initial_answer = response['answer']
            for iteration in range(2):
                feedback = self.get_feedback(response['answer'])
                logging.debug(f"Iteration {iteration + 1} - Feedback: {feedback}")
                
                if feedback == "good":
                    break
                    
                response = self.adjust_response(query, response, feedback)
            final_answer = response['answer']
            print(f"Query: {query}")
            print(f"Final Answer: {final_answer}")
            print("-" * 50)
            
            results.append({
                'query': query,
                'initial_answer': initial_answer,
                'final_answer': final_answer,
                'context_used': response.get('context_used')
            })
        
        return results

if __name__ == "__main__":
    try:
        bot = SupportBotAgent("faq.txt")
        sample_queries = [
            "How do I reset my password?",
            "What's the refund policy?",
            "How do I contact support?",
            "How do I fly to the moon?"  
        ]
        results = bot.run(sample_queries)
    except Exception as e:
        logging.error(f"Failed to run bot: {e}")
        print(f"Error: {e}")