import streamlit as st
import re
import os

# Correctly setting the OpenAI API key as a string
os.environ["OPENAI_API_KEY"] = "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"

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# The rest of your code follows...


def clean_text(text):
    """Clean the text by removing excessive newlines and extra spaces."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    all_text = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if not page_text:
                    st.warning(f"Warning: No text extracted from page {page_num + 1} of {pdf.name}.")
                    page_text = ""  # Continue processing even if no text is found on this page
                pdf_text += page_text.replace("\n", " ")  # Clean up the text
            all_text.append(pdf_text)
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
    return all_text


def chunk_texts(text_list):
    """Chunk the cleaned text into manageable pieces."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    
    for idx, text in enumerate(text_list):
        cleaned_text = clean_text(text)
        
        # Log the length of the text before splitting
        st.write(f"Text length before splitting: {len(cleaned_text)} characters.")
        
        # Split the cleaned text into chunks
        chunked_text = text_splitter.split_text(cleaned_text)
        
        # Log how many chunks were generated
        st.write(f"Number of chunks generated: {len(chunked_text)}")
        
        # Ensure each chunk is stored with its metadata
        for chunk in chunked_text:
            doc = Document(page_content=chunk, metadata={"document_idx": idx})
            chunks.append(doc)
            st.write(f"Generated chunk with length {len(chunk)}: {chunk[:100]}...")  # Preview first 100 characters
            st.write(f"Metadata for chunk: {doc.metadata}")  # Debugging: Print metadata for each chunk

    return chunks


def create_embeddings(chunks):
    """Create embeddings for the document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Extract text content and metadata from the Document objects
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # Check how many texts are being indexed
    st.write(f"Indexing {len(texts)} chunks into FAISS.")
    
    # Create the FAISS vector store with the chunks and embeddings
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    return vector_store, embeddings


def search_documents(query, vector_store, embeddings):
    """Search for relevant documents based on the user's query."""
    query_embedding = embeddings.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=3)
    
    # Dynamically filter results based on query terms
    relevant_results = []
    query_terms = query.lower().split()  # Split the query into words

    for result in results:
        # Check if any of the query terms appear in the chunk
        if any(term in result.page_content.lower() for term in query_terms):
            relevant_results.append(result)
    
    return relevant_results if relevant_results else results


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon="🤖")

    # Main header and initial chat messages
    st.header("Chat with PDF(s) 🤖")
    with st.chat_message("user"):
        st.write("Hello 👋")
    with st.chat_message("assistant"):
        st.write("Hello human! Feel free to upload your PDFs and ask questions.")

    # Sidebar for file upload and processing
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF(s) here and click on 'Process'", accept_multiple_files=True)
        process_button = st.button("Process", disabled=not pdf_docs)
        
        if not pdf_docs:
            st.warning("Please upload at least one PDF file before processing.")
        elif len(pdf_docs) > 3:
            st.warning("Please upload up to 3 PDF files at a time for optimal performance.")

    # Initialize vector store and embeddings in session state if not already initialized
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.embeddings = None
        st.session_state.qa_chain = None  # Initialize QA chain in session state

    if process_button:
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # Get raw text from the uploaded PDFs
                raw_text_list = get_pdf_text(pdf_docs)
                if not raw_text_list:
                    st.error("No text extracted from PDFs.")
                    return

                # Chunking the extracted text
                chunks = chunk_texts(raw_text_list)

                # Embedding the chunks and saving to session state
                st.session_state.vector_store, st.session_state.embeddings = create_embeddings(chunks)

                # Load QA chain
                llm = OpenAI(temperature=0)  # Initialize OpenAI LLM for QA
                st.session_state.qa_chain = load_qa_chain(llm, chain_type="map_reduce")

                st.success("PDF(s) processed successfully! Embeddings created.")

    user_question = st.chat_input("Ask a question about your document(s):")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        if st.session_state.vector_store and st.session_state.embeddings and st.session_state.qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Searching for the best answer based on the documents..."):
                    # Retrieve relevant documents
                    results = search_documents(user_question, st.session_state.vector_store, st.session_state.embeddings)
                    
                    # Convert results to Document objects before passing to QA chain
                    relevant_docs = [Document(page_content=result.page_content, metadata=result.metadata) for result in results]
                    
                    # Use the QA chain to generate an answer from the documents
                    if relevant_docs:
                        answer = st.session_state.qa_chain.run(input_documents=relevant_docs, question=user_question)
                        st.write(f"Answer: {answer}")
                    else:
                        st.write("Sorry, no relevant sections found. Try rephrasing your query.")
        else:
            with st.chat_message("assistant"):
                st.write("Error: Please reprocess the documents.")
                st.button("Reprocess", on_click=main)

if __name__ == '__main__':
    main()
