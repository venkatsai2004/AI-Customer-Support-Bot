# AI Customer Support Bot with Document Training

# **Overview**

This project implements an agentic customer support bot that trains on provided documents (PDF or text files) and answers queries using advanced NLP models with a feedback loop for continuous improvement.

## Features

* **Document Processing**: Supports both PDF and text file inputs
* **Semantic Search**: Uses FAISS vector store with HuggingFace embeddings
* **Question Answering**: DistilBERT-based QA pipeline
* **Feedback Loop**: Simulated feedback system with response refinement
* **Comprehensive Logging**: Tracks all decisions and iterations
* **Graceful Fallbacks**: Handles out-of-scope queries appropriately

## Setup Instructions

### Prerequisites

* Python 3.8+
* pip package manager

### Installation


1. Clone the repository:

```bash
git clone <your-repo-url>
cd customer-support-bot
```


2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

```
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

## Usage

### Basic Usage

```python
from bot_main import SupportBotAgent

# Initialize bot with your document
bot = SupportBotAgent("faq.txt")

# Process queries
queries = [
    "How do I reset my password?",
    "What's the refund policy?",
    "How do I contact support?"
]

bot.run(queries)
```

### Advanced Usage

```python
# Process individual queries with manual feedback
response = bot.answer_query("How do I reset my password?")
feedback = bot.get_feedback(response["answer"])
improved_response = bot.adjust_response(query, response, feedback)
```

## File Structure

```
├── bot_main.py              # Main bot implementation
├── example_usage.py         # Comprehensive examples
├── faq.txt                  # Sample FAQ document
├── faq.pdf                  # Sample FAQ in PDF format
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── log/                    # Generated log files
│   └── support_bot_log.txt
└── documents/              # Document storage
    ├── sample_faq.txt
    └── tech_faq.txt
```

## Running Examples

Execute the comprehensive examples:

```bash
python example_usage.py
```

This will run multiple scenarios including:

* Basic usage with FAQ
* Custom document processing
* Batch processing multiple documents
* Performance analysis
* Feedback strategy testing

## Document Requirements

### Supported Formats

* Plain text files (.txt)
* PDF files (.pdf)

### Document Structure

For best results, structure your documents with clear sections:

```
Section Title
Content for this section...

Another Section Title  
More content here...
```

## Configuration

### Similarity Threshold

Modify the similarity threshold in `_is_query_covered()` method:

```python
return score < 1.0  # Adjust threshold as needed
```

### Chunk Size

Adjust document chunking parameters:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Increase/decrease as needed
    chunk_overlap=50,      # Overlap between chunks
    length_function=len
)
```

## Logging

All bot activities are logged to `log/support_bot_log.txt` including:

* Document loading and processing
* Query processing
* Feedback received
* Response adjustments
* Error messages

## Troubleshooting

### Common Issues

**1. "No answer provided" responses**

* Check if your document is properly formatted
* Verify file path is correct
* Ensure document content matches query topics

**2. Slow performance**

* First run downloads models (normal delay)
* Consider using smaller chunk sizes for large documents

**3. Memory issues**

* Reduce chunk_size in text splitter
* Use CPU instead of GPU if needed

### Debug Mode

Enable detailed logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Deployment Options

### Local Development

```bash
python bot_main.py
```

### Streamlit Web App

```bash
pip install streamlit
streamlit run streamlit app.py  # Create this file
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "bot_main.py"]
```

## API Integration

## This Customer Ai Bot can aslo be used for the Falsk/FastAPI

The bot can be wrapped in a Flask/FastAPI application:

```python
from flask import Flask, request, jsonify
from bot_main import SupportBotAgent

app = Flask(__name__)
bot = SupportBotAgent("faq.txt")

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    response = bot.answer_query(data['query'])
    return jsonify(response)
```

## Performance Metrics

The bot tracks several metrics:

* Response time per query
* Similarity confidence scores
* Feedback improvement rates
* Out-of-scope query detection

## Contributing


1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Known Issues

* Occasional "No answer provided" for valid queries (working on fix)
* Large PDF processing can be memory intensive
* First-time model loading takes \~30 seconds

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:

* Check the troubleshooting section
* Review the example_usage.py file
* Create an issue in the GitHub repository

## Future Improvements for Your AI Customer Support Bot

* Integrate generative AI models (e.g., via LangChain with larger Hugging Face models like GPT-J or fine-tuned Llama) for more natural, conversational responses beyond DistilBERT's QA pipeline.
* Add sentiment analysis and emotional intelligence using libraries like Hugging Face's text-classification pipelines to detect user frustration and adapt responses (e.g., escalate to human support).
* Implement predictive analytics with scikit-learn or statsmodels to anticipate user needs based on query patterns, such as suggesting related FAQs proactively.
* Support voice input/output by integrating speech-to-text (e.g., via Hugging Face's speech-recognition models) and text-to-speech for audio-based interactions.
* Enable integration with external APIs or CRMs (e.g., Zendesk or Salesforce via REST calls) to fetch real-time data like order status, enhancing the bot's utility beyond static documents.
* Add privacy and security features, such as data encryption for logs and anonymization of user queries, to build trust in line with GDPR compliance.
* Incorporate A/B testing for response strategies using libraries like scipy to evaluate and optimize feedback loops with real or simulated user data.
* Scale for production with cloud deployment (e.g., AWS or Hugging Face Spaces enhancements) and handle high-volume queries using batch processing or distributed computing.

T