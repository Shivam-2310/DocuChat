import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def initialize_pipeline(pdf_path: str):
    # Extract text directly from the PDF
    raw_text = extract_text_from_pdf(pdf_path)

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_text(raw_text)

    # Create vector store
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )

    # Setup retriever
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    local_model="llama3.1:latest"
    llm = ChatOllama(model=local_model)
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    return retriever, llm


def generate_answer(retriever, llm, question: str):
    # Define the RAG prompt template
    template = """Answer the question based ONLY on the following context.
    {context}
    Question: {question}
    """

    # Create a ChatPromptTemplate from the template
    prompt = ChatPromptTemplate.from_template(template)

    # Create a chain of operations
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with the question, expecting a dictionary as input
    inputs = {"question": question}
    return chain.invoke(inputs)