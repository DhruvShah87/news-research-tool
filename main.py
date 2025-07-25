import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import pickle

import asyncio
import threading

if threading.current_thread() is not threading.main_thread():
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()

st.title('News Research Tool')
st.sidebar.title('News Articles URLs')

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

process_url_clicked = st.sidebar.button('Process URLs')
dir_path = "faiss_index_dir"

main_placeholder = st.empty()
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text('Loading Data...')
    data = loader.load()

    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=500,
        chunk_overlap = 0
    )
    main_placeholder.text('Splitting Data...')
    chunks = r_splitter.split_documents(data)

    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    main_placeholder.text('Embedding Data...')
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(dir_path)

query = main_placeholder.text_input("Question: ")
search_clicked = st.button("Search")

if search_clicked and query:
    print(query)
    if os.path.exists(dir_path):
        print(dir_path)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.load_local("faiss_index_dir", embeddings=embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=model, retriever=retriever)
        print(chain)
        response = chain({"question": query}, return_only_outputs=True)
        print(response)

        st.header('Answer')
        st.write(response['answer'])

        sources = response.get("sources", "")
        if sources:
            st.subheader('Sources')
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)