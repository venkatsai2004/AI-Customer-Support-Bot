# 🤖 AI Customer Support Bot with Document Training


This project is an **agentic customer support bot** that trains on provided documents (PDF or text files) and answers user queries using advanced NLP models.It includes a feedback loop for continuous improvement, ensuring accurate and context-aware responses.

# Content in Files
- <p>
  For detailed information on <code>app.py</code>, <code>example_usage.py</code>, and <code>support_bot_main.py</code>, please refer to 
  <a href="BOT_GUIDE.md" target="_blank">BOT_GUIDE.md</a>.
</p>


  

## 🚀 Features

* 📄 **Document Processing** – Supports both PDF and text file inputs
* 🔍 **Semantic Search** – Uses FAISS vector store with HuggingFace embeddings for efficient retrieval
* 💡 **Question Answering** – Powered by a DistilBERT-based QA pipeline
* 🔄 **Feedback Loop** – Simulates user feedback to refine responses
* 📝 **Comprehensive Logging** – Tracks all decisions, iterations, and errors
* 🛡️ **Graceful Fallbacks** – Handles out-of-scope queries appropriately


## ⚙️ Setup Instructions

## Prerequisites

* Python 3.8+
* pip package manager

## Installation

```bash
git clone <your-repo-url>
cd customer-support-bot
pip install -r requirements.txt
```

### Dependencies

```bash
langchain>=0.1.0
langchain-huggingface>=0.0.3
langchain-community>=0.0.20
transformers>=4.21.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
pdfplumber>=0.8.0
torch>=1.12.0
numpy>=1.21.0
```

# 📌 Basic Usage
```bash
from bot_main import SupportBotAgent
```

# Initialize the bot with your document
```bash
bot = SupportBotAgent("faq.txt")
```
# Process queries

queries = \[
- "How do I reset my password?",
- "What's the refund policy?",
- "How do I contact support?"
\]
- bot.run(queries)

# 📌 Advanced Usage

# Process individual queries with manual feedback

- response = bot.answer_query("How do I reset my password?")
- feedback = bot.get_feedback(response\["answer"\])
- improved_response = bot.adjust_response(query, response, feedback)

# 🏃 Running Examples

python example_usage.py

- This demonstrates:
- Basic usage with FAQs
- Custom document processing
- Batch processing multiple documents
- Performance analysis
- Feedback strategy testing

# 📑 Document Requirements

- Supported Formats: .txt, .pdf
- Best Practice: Use clear section titles and structured content

# ⚙️ Configuration

- Similarity Threshold
- return score < 1.0  # Adjust threshold as needed

# Chunk Size

text_splitter = RecursiveCharacterTextSplitter(
- chunk_size=500,
- chunk_overlap=50,
- length_function=len,
- separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)

# 🐛 Troubleshooting

- “No answer provided” → Check formatting, file paths, and query relevance
- Slow performance → First run downloads models (normal delay). Reduce chunk size for large docs
- Memory issues → Reduce chunk_size, use CPU instead of GPU if needed

# Enable debug logging:

- logging.basicConfig(level=logging.DEBUG)

# 🌐 Deployment Options

- <p>
  For the Live Deployment, visit the 
  <a href="(https://huggingface.co/spaces/Vsai2004/AI_Customer_Support_Bot)" target="_blank">
    AI Customer Support Bot
  </a>.
</p>

- Local Development: python bot_main.py

# Streamlit Web App
```bash
pip install streamlit
streamlit run app.py
```
# Docker
```bash
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD \["python", "bot_main.py"\]
```
# API Integration (Flask)
```bash
from flask import Flask, request, jsonify
from bot_main import SupportBotAgent

app = Flask(__name__)
bot = SupportBotAgent("faq.txt")

@app.route('/query', methods=\['POST'\])
def handle_query():
data = request.json
response = bot.answer_query(data\['query'\])
return jsonify(response)
```
# 📊 Performance Metrics

* Response time per query
* Similarity confidence scores
* Feedback improvement rates
* Out-of-scope query detection

# 📄 License

MIT License – See LICENSE for details.

# Future Improvements for Your AI Customer Support Bot

* Integrate generative AI models (e.g., via LangChain with larger Hugging Face models like GPT-J or fine-tuned Llama) for more natural, conversational responses beyond DistilBERT's QA pipeline.
* Add sentiment analysis and emotional intelligence using libraries like Hugging Face's text-classification pipelines to detect user frustration and adapt responses (e.g., escalate to human support).
* Implement predictive analytics with scikit-learn or statsmodels to anticipate user needs based on query patterns, such as suggesting related FAQs proactively.
* Support voice input/output by integrating speech-to-text (e.g., via Hugging Face's speech-recognition models) and text-to-speech for audio-based interactions.
* Enable integration with external APIs or CRMs (e.g., Zendesk or Salesforce via REST calls) to fetch real-time data like order status, enhancing the bot's utility beyond static documents.
* Add privacy and security features, such as data encryption for logs and anonymization of user queries, to build trust in line with GDPR compliance.
* Incorporate A/B testing for response strategies using libraries like scipy to evaluate and optimize feedback loops with real or simulated user data.
* Scale for production with cloud deployment (e.g., AWS or Hugging Face Spaces enhancements) and handle high-volume queries using batch processing or distributed computing.


