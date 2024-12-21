# PDF Summarizer

This project is a Python-based web application for summarizing PDF files using pre-trained machine learning models from Hugging Face's transformers library. The application provides an interactive interface for uploading a PDF, selecting a summarization model, and receiving a concise summary of the document.

## Features
- Upload PDF files for summarization.
- Choose from multiple pre-trained summarization models:
  - `allenai/led-large-16384-arxiv`
  - `philschmid/bart-large-cnn-samsum`
  - `csebuetnlp/mT5_multilingual_XLSum`
- Automatically detects and uses a GPU if available for faster processing.
- Easy-to-use web interface built with Gradio.

## Prerequisites
Before running this project, ensure you have the following installed:
1. Python 3.8 or later
2. pip (Python package manager)
3. CUDA-enabled GPU (optional, for faster processing)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sanahshrivastav/pdf-summarizer.git
   cd pdf-summarizer
   ```
2. Install the required dependencies:
   ```bash
   pip install gradio langchain transformers PyPDF2 torch
   ```
3. Verify that your environment is set up correctly:
   ```bash
   python -m torch.cuda.is_available
   ```
   If `True` is printed, your GPU is ready to use.

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Open the web interface:
   - By default, the app will launch in your browser.
   - If running on a remote server, use the provided shareable link.
3. Interact with the app:
   - Upload a PDF file.
   - Select a summarization model from the dropdown menu.
   - View the generated summary in the text box.

## Code Walkthrough

### Imports
```python
import gradio as gr
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from transformers import pipeline
import torch
```
- **Gradio**: A library for creating web interfaces for machine learning models or Python functions.
- **CharacterTextSplitter**: Utility for splitting text into smaller chunks (not used in the code).
- **PdfReader**: Reads and extracts text from PDF files.
- **pipeline**: High-level API from Hugging Face's transformers library for tasks like summarization.
- **torch**: Deep learning library used to detect GPU availability.

### Setting the Device
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```
- Detects if a CUDA-enabled GPU is available and configures the device accordingly.

### Summarization Function
```python
def summarize_pdf(pdf_file_path, model_name):
    summarizer = pipeline('summarization', model=model_name, min_length=100, max_length=200, device=0 if torch.cuda.is_available() else -1)
    loader = PdfReader(pdf_file_path.name)
    text = ""
    for page in loader.pages:
        text += page.extract_text()
    summary = summarizer(text)
    return summary[0]['summary_text']
```
- Extracts text from the uploaded PDF.
- Uses the specified pre-trained model to generate a summary.

### Gradio Interface
```python
def main():
    input_pdf_path = gr.File(label="Upload PDF")
    select_model = gr.Dropdown(
        choices=["allenai/led-large-16384-arxiv", "philschmid/bart-large-cnn-samsum", "csebuetnlp/mT5_multilingual_XLSum"],
        label="Select Model"
    )
    output_summary = gr.Textbox(label="Summary")

    iface = gr.Interface(
        fn=summarize_pdf,
        inputs=[input_pdf_path, select_model],
        outputs=[output_summary],
        title="PDF Summarizer",
        description="Upload a PDF file and select a model to get its summary."
    )

    iface.launch(share=True, inbrowser=True)

if __name__ == "__main__":
    main()
```
- Creates an interactive Gradio interface.
- Accepts a PDF file and model selection as input and returns the summary.

## Models
### Supported Pre-Trained Models
- **`allenai/led-large-16384-arxiv`**: Optimized for summarizing large documents like research papers.
- **`philschmid/bart-large-cnn-samsum`**: Fine-tuned for dialogue and conversational text summarization.
- **`csebuetnlp/mT5_multilingual_XLSum`**: A multilingual model for summarizing text in various languages.

## Dependencies
- `torch`
- `transformers`
- `gradio`
- `PyPDF2`
- `langchain`

## Contributing
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License. 

## Acknowledgments
- **Hugging Face**: For providing pre-trained models.
- **Gradio**: For an easy-to-use web interface library.

## Contact
For questions or support, please contact:
- **Name**: Sanah Shrivastav
- **Email**: sanahshrivastav@gmail.com.com
- **GitHub**: https://github.com/sanahshrivastav

