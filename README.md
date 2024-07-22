# Jessup Cellars Business Chatbot 
This README provides detailed instructions on how to set up and run the Jessup Cellars Business Chatbot using Streamlit. The chatbot is designed to answer questions based on a provided PDF corpus and sample question-answer pairs in JSON format.

## Table of Contents
1. Prerequisites
2. Installation
3. Project Structure
4. Setting Up the Environment
5. Running the App
6. Troubleshooting

## Prerequisites
Ensure you have the following installed on your system:

- Python 3.7+: Download from python.org.
- pip: Python package installer.
- Virtual Environment (optional but recommended): To manage dependencies.

## Installation
Install the required Python packages using pip:

pip install streamlit json PyPDF2 faiss-cpu numpy sentence-transformers torch

## Project Structure
Organize your project directory as follows:

Business-Chatbot/

├── Business_chatbot.py

├── Sample Question Answers.json

├──Corpus.pdf

└── requirements.txt

- Business_chatbot.py: Main Python script for the Streamlit app.
- Sample Question Answers.json: JSON file with sample questions and answers.
- Corpus.pdf: PDF file containing the corpus text.

## Setting Up the Environment
1. Clone the Repository:

   git clone <https://github.com/thivakaran-mnm/Business-Chatbot>

   cd Busiess-Chatbot

2. Create a Virtual Environment (optional):

   python -m venv venv

   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. Install Dependencies:

   pip install -r requirements.txt

## Running the App
1. Navigate to the Project Directory:

   cd JessupCellarsChatbot

2. Run the Streamlit App:

   streamlit run Business_chatbot.py

3. Access the App: Open a web browser and go to the URL provided by Streamlit, typically http://localhost:8501.

## Troubleshooting
- Ensure Dependencies are Installed: Double-check that all required Python packages are installed correctly.
- Check File Paths: Verify that the file paths for Sample Question Answers.json and Corpus.pdf are correct.
- Model Loading Issues: Ensure you have a stable internet connection when loading the SentenceTransformer model for the first time, as it may need to download model files.







