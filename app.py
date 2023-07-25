from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Qdrant
import qdrant_client
from qdrant_client.http import models
from qdrant_client import QdrantClient
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("PDF READER 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings and  object out  of  it    so that  we  are able  perform  search on  it
      embeddings = OpenAIEmbeddings(
      model_name="sentence-transformers/all-mpnet-base-v2"
     )
      client = qdrant_client.QdrantClient(
    "https://0770f5b9-586a-4307-9410-550e20a69f1e.us-east-1-0.aws.cloud.qdrant.io:6333",
    api_key="jVhqP1jbL7IFqQ7cPh97gGIXj7BIfIjeicR_WbLXg5YZSpWdDualdA", 
      )
      client.recreate_collection(
    collection_name="chunks",
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
     )

      knowledge_base= Qdrant(
    client=client, collection_name="chunks", 
      embeddings =embeddings ,
     )
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
          
           
        st.write(response)
    

if __name__ == '__main__':
    main()
