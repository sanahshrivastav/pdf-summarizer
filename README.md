# PDF-Summarizer

This project is a Python-based web application for summarizing PDF files using pre-trained machine learning models from Hugging Face's transformers library. The application provides an interactive interface for uploading a PDF, selecting a summarization model, and receiving a concise summary of the document.

*Features*
Upload PDF files for summarization.
Choose from multiple pre-trained summarization models:
-allenai/led-large-16384-arxiv
-philschmid/bart-large-cnn-samsum
-csebuetnlp/mT5_multilingual_XLSum
Automatically detects and uses a GPU if available for faster processing.
Easy-to-use web interface built with Gradio.

*Prerequisites*
Before running this project, ensure you have the following installed:
1. Python 3.8 or later
2. pip (Python package manager)
3. CUDA-enabled GPU (optional)

*Installation*
1. Clone the repository: gh repo clone sanahshrivastav/pdf-summarizer
2. Install the required dependencies:
   !pip install gradio langchain transformers PyPDF2 torch
