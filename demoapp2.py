from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import qdrant_client
import os
import fitz

# create client

os.environ['QDRANT_HOST'] = ""
os.environ['QDRANT_API_KEY'] = ""


client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

# create collection

os.environ['QDRANT_COLLECTION_NAME'] ="my collection"

collection_config = qdrant_client.http.models.VectorParams(
       size=1536, # 768 for instructor-xl, 1536 for OpenAI
        distance=qdrant_client.http.models.Distance.COSINE
    )

#client.recreate_collection(
   # collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    #vectors_config=collection_config
#)

# create  vector store
def get_vector_store():
    os.environ['OPENAI_API_KEY'] = ""

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings
    )
    return vector_store
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():
  
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            raw_text = ""
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()

            texts = get_chunks(raw_text)
            st.write(f"Number of chunks extracted from {uploaded_file.name}: {len(texts)}")
            vector_store = get_vector_store()
            vector_store.add_texts(texts)
    
    st.header("Ask questions about PDFs 💬")
    vector_store = get_vector_store()
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Allow the user to select a PDF to ask questions about
    selected_pdf = st.selectbox("Select a PDF to ask questions about", [file.name for file in uploaded_files])
    
    if selected_pdf:
        user_question = st.text_input("Ask a question:")
        if user_question:
            st.write(f"Question: {user_question}")
            answer = qa.run(user_question)
            st.write(f"Answer: {answer}")

if __name__ == '__main__':
    main()
