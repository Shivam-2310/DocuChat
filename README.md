# DocuChat

DocuChat is a document interaction platform that allows users to upload PDF documents and query their content using advanced language models. Built with Python, FastAPI, LangChain, ChromaDB, and Ollama, DocuChat offers features for document summarization and question answering.

## Key Features

- **Document Upload**: Upload PDF documents for processing.
- **Content Interaction**: Ask specific questions or request summaries based on the document content.
- **Advanced Retrieval**: Uses a MultiQueryRetriever to generate multiple question perspectives for better document retrieval.
- **Local Processing**: Ensures data privacy by processing everything locally.

## Technologies Used

- **Python**
- **FastAPI**: Web framework for building the backend API.
- **Uvicorn**: ASGI server for running FastAPI applications.
- **PyMuPDF**: Library for PDF text extraction.
- **LangChain**: Framework for chaining natural language processing operations.
- **ChromaDB**: Vector database for managing document embeddings.
- **Ollama**: Language models for generating embeddings and answering queries.

## Setup Instructions

### Prerequisites

- Python 3.7+
- Pip (Python package installer)
- Ollama needs to be installed. Obtain it from [Ollama's website](https://www.ollama.com/).

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/docuchat.git
   cd docuchat
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```

4. **Access the Application**
   Use Postman or similar tools to interact with the API endpoints.

## API Endpoints

### 1. Upload PDF Document

- **Endpoint**: `/upload/`
- **Method**: `POST`
- **Description**: Upload a PDF document to the server. The document is processed and stored for interaction.

- **Request**:
  - `file` (form-data): PDF file to upload

- **Response**:
  ```json
  {
    "message": "PDF processed successfully. You can now ask questions."
  }
  ```

### 2. Ask a Question

- **Endpoint**: `/ask/`
- **Method**: `POST`
- **Description**: Submit a question related to the uploaded document. The application generates an answer based on the document's content.

- **Request**:
  - `question` (JSON body): The question you want to ask

- **Response**:
  ```json
  {
    "answer": "The answer to your question."
  }
  ```

## Future Enhancements

1. **Support for Different File Formats**
   - Add support for additional document formats like Word, Excel, and plain text.

2. **Advanced User Interface**
   - Develop a web interface with drag-and-drop upload, real-time query results, and enhanced user experience.

3. **Enhanced Retrieval Techniques**
   - Implement advanced retrieval techniques to improve answer accuracy and relevance.

## Contributing

We welcome contributions to improve DocuChat! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your modifications and commit them.
4. Push your changes to your forked repository.
5. Open a pull request to the main repository.


## Acknowledgments

- The creators of LangChain, ChromaDB, and Ollama for providing essential libraries and tools for this project.

