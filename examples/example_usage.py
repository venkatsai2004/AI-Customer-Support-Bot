import sys
import os
import time

# Add parent directory to Python path so we can import bot_main
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from bot_main import SupportBotAgent

def basic_usage_example():
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    documents_dir = os.path.join(parent_dir, "documents")
    logs_dir = os.path.join(parent_dir, "log")
    os.makedirs(logs_dir, exist_ok=True)
    sample_faq_path = os.path.join(documents_dir, "sample_faq.txt")
    log_path = os.path.join(logs_dir, "example_usage_log.txt")

    if not os.path.exists(sample_faq_path):
        os.makedirs(documents_dir, exist_ok=True)
        with open(sample_faq_path, "w", encoding="utf-8") as f:
            f.write("""Password Reset
To reset your password, click 'Forgot Password' on the login page.
Refund Policy
Refunds are processed within 7 business days of request.
Contact
You can contact us at support@example.com.""")

    bot = SupportBotAgent(sample_faq_path)
    bot.log_file = log_path
    queries = [
        "How do I reset my password?",
        "What's the refund policy?",
        "How can I contact support?",
        "How do I fly to the moon?"
    ]
    results = bot.run(queries)
    return results

def advanced_usage_example():
    print("\n" + "=" * 60)
    print("ADVANCED USAGE EXAMPLE")
    print("=" * 60)
    documents_dir = os.path.join(parent_dir, "documents")
    logs_dir = os.path.join(parent_dir, "log")
    os.makedirs(logs_dir, exist_ok=True)
    sample_faq_path = os.path.join(documents_dir, "sample_faq.txt")
    log_path = os.path.join(logs_dir, "advanced_usage_log.txt")

    bot = SupportBotAgent(sample_faq_path)
    bot.log_file = log_path
    queries = [
        "I need help with billing",
        "Technical problems with the software",
        "How do I delete my account?"
    ]
    all_results = []
    for query in queries:
        response_info = bot.answer_query(query)
        feedback = bot.get_feedback(response_info["answer"])
        adjusted = bot.adjust_response(query, response_info, feedback)
        all_results.append({
            "query": query,
            "response": adjusted,
            "feedback": feedback
        })
        print(f"\nQuery: {query}")
        print(f"Response: {adjusted}")
        print(f"Feedback: {feedback}")
    return all_results

def custom_document_example():
    print("\n" + "=" * 60)
    print("CUSTOM DOCUMENT EXAMPLE")
    print("=" * 60)
    documents_dir = os.path.join(parent_dir, "documents")
    logs_dir = os.path.join(parent_dir, "log")
    os.makedirs(logs_dir, exist_ok=True)
    tech_faq_path = os.path.join(documents_dir, "tech_faq.txt")
    log_path = os.path.join(logs_dir, "tech_bot_log.txt")

    tech_faq = """API Documentation
Our REST API allows developers to integrate with our platform.
Use the base URL https://api.example.com/v1/ for all requests.
Authentication requires an API key in the header.
Rate Limiting
API calls are limited to 1000 requests per hour per API key.
Exceeded limits return a 429 status code. Premium accounts have higher limits.
Error Handling
API errors return standard HTTP status codes.
400 for bad requests, 401 for unauthorized, 404 for not found, and 500 for server errors.
Error messages include details in JSON format.
SDK Support
We provide official SDKs for Python, JavaScript, and Java.
Community-maintained SDKs are available for other languages.
Check our GitHub repository for the latest versions.
Webhooks
Set up webhooks to receive real-time notifications about events.
Configure webhook URLs in your dashboard.
We support retry logic for failed webhook deliveries."""

    os.makedirs(documents_dir, exist_ok=True)
    with open(tech_faq_path, "w", encoding="utf-8") as f:
        f.write(tech_faq)

    tech_bot = SupportBotAgent(tech_faq_path)
    tech_bot.log_file = log_path
    tech_queries = [
        "What's the API rate limit?",
        "How do I handle API errors?",
        "Do you have a Python SDK?",
        "How do webhooks work?"
    ]
    results = tech_bot.run(tech_queries)
    return results

def feedback_strategy_example():
    print("\n" + "=" * 60)
    print("CUSTOM FEEDBACK STRATEGY EXAMPLE")
    print("=" * 60)
    documents_dir = os.path.join(parent_dir, "documents")
    logs_dir = os.path.join(parent_dir, "log")
    os.makedirs(logs_dir, exist_ok=True)
    sample_faq_path = os.path.join(documents_dir, "sample_faq.txt")
    log_path = os.path.join(logs_dir, "strategy_bot_log.txt")

    bot = SupportBotAgent(sample_faq_path)
    bot.log_file = log_path

    def detailed_explanation_strategy(self, query, response_info, feedback):
        context = response_info.get("context_used", "")
        if context:
            detailed_response = f"""
            Here's a detailed explanation:
            {response_info['answer']}
            Background information:
            {context}
            If you need further clarification, please let me know!
            """
            return detailed_response.strip()
        return response_info['answer']

    bot.adjust_response = lambda q, r, f: detailed_explanation_strategy(bot, q, r, f)

    queries = [
        "How does the refund process work?",
        "What are the steps to contact support?"
    ]
    results = []
    for q in queries:
        response_info = bot.answer_query(q)
        feedback = bot.get_feedback(response_info["answer"])
        adjusted = bot.adjust_response(q, response_info, feedback)
        results.append({"query": q, "response": adjusted, "feedback": feedback})
        print(f"Query: {q}")
        print(f"Response: {adjusted}")
        print(f"Feedback: {feedback}")
    return results

def similarity_threshold_example():
    print("\n" + "=" * 60)
    print("SIMILARITY THRESHOLD COMPARISON")
    print("=" * 60)
    documents_dir = os.path.join(parent_dir, "documents")
    sample_faq_path = os.path.join(documents_dir, "sample_faq.txt")

    bot = SupportBotAgent(sample_faq_path)
    query = "How do I change my password?"
    if bot._is_query_covered(query):
        print(f"✓ Query is covered: {query}")
    else:
        print(f"✗ Query not covered: {query}")

def batch_processing_example():
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    documents_base = os.path.join(parent_dir, "documents", "company_faqs")
    logs_base = os.path.join(parent_dir, "log", "company_faqs")
    os.makedirs(documents_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    documents = {
        "general_faq.txt": """
        General Questions
        What is our company about? We provide innovative software solutions.
        Where are you located in India? Our branch is in Pune.
        When were you founded? We were founded in 2020.
        """,
        "technical_faq.txt": """
        Technical Support
        System requirements? Windows 10, Mac OS 10.15+, or Linux Ubuntu 18+.
        Installation help? Download from our website and run the installer.
        Troubleshooting? Check our knowledge base or contact support.
        """
    }

    document_paths = {}
    for filename, content in documents.items():
        full_path = os.path.join(documents_base, filename)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        document_paths[filename] = full_path

    all_results = {}
    for doc_name, doc_path in document_paths.items():
        print(f"\n--- Processing {doc_name} ---")
        log_filename = os.path.join(logs_base, f"batch_{doc_name}_log.txt")

        bot = SupportBotAgent(doc_path)
        bot.log_file = log_filename

        if "general" in doc_name:
            queries = [
                "What does your company do?",
                "Where are you based?",
                "When did you start?"
            ]
        else:
            queries = [
                "What are the system requirements?",
                "How do I install the software?",
                "I'm having technical issues"
            ]

        results = bot.run(queries)
        all_results[doc_name] = results if results else []

    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")

    total_queries = sum(len(results) for results in all_results.values() if results)
    print(f"Total documents processed: {len(all_results)}")
    print(f"Total queries processed: {total_queries}")

    for doc_name, results in all_results.items():
        if results:
            successful = sum(1 for r in results if not r.get("out_of_scope", True))
            print(f"{doc_name}: {successful}/{len(results)} successful queries")
        else:
            print(f"{doc_name}: 0/0 successful queries")

    return all_results

def performance_analysis_example():
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS EXAMPLE")
    print("=" * 60)
    documents_dir = os.path.join(parent_dir, "documents")
    logs_dir = os.path.join(parent_dir, "log")
    os.makedirs(logs_dir, exist_ok=True)
    sample_faq_path = os.path.join(documents_dir, "sample_faq.txt")
    log_path = os.path.join(logs_dir, "performance_analysis_log.txt")

    bot = SupportBotAgent(sample_faq_path)
    bot.log_file = log_path
    test_queries = [
        "How do I reset my password?",
        "What's your refund policy?",
        "How can I contact support?",
        "What are your shipping options?",
        "How do I update my account?",
        "I need technical support",
        "Questions about billing",
        "Privacy and security info"
    ]
    performance_data = []
    print("Running performance analysis...")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}/{len(test_queries)}: {query}")
        start_time = time.time()
        response_info = bot.answer_query(query)
        answer = response_info["answer"]
        result = {
            "initial_confidence": 1.0 if "I don't have enough" not in answer else 0.0,
            "out_of_scope": "I don't have enough" in answer,
        }
        end_time = time.time()
        processing_time = end_time - start_time
        performance_data.append({
            "query": query,
            "processing_time": processing_time,
            "confidence": result.get("initial_confidence", 0),
            "out_of_scope": result["out_of_scope"]
        })
        print(f"Processing time: {processing_time:.2f}s")

    return performance_data

def main():
    print("🤖 Customer Support Bot - Example Usage Demonstrations\n")
    try:
        basic_usage_example()
        advanced_usage_example()
        custom_document_example()
        feedback_strategy_example()
        similarity_threshold_example()
        batch_processing_example()
        performance_analysis_example()
        print(f"\n{'='*60}")
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("pip install transformers sentence-transformers PyPDF2 torch")

if __name__ == "__main__":
    main()