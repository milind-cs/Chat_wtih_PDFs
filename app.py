import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from transformers import pipeline

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def generate_summary(raw_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(raw_text, max_length=1000, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=vectorstore.as_retriever())
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = None
    st.header("Chat with PDFs :books:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # Get the PDFs
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.raw_text = raw_text
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                # Create vector storage
                vectorstore = get_vectorstore(text_chunks)
                # Get conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete!")
        
        if st.session_state.raw_text and st.button("Generate Summary"):
            with st.spinner("Generating summary"):
                # Generate summary
                summary = generate_summary(st.session_state.raw_text)
                # Display summary
                st.write("Summary:")
                st.write(summary)
                
        if pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            pdf_reader = PdfReader(pdf_docs[0])  # Assuming we're reading the first document
            num_pages = len(pdf_reader.pages)

            page_number = st.number_input("Enter page number", min_value=1, max_value=num_pages, value=1)
            st.write(f"Page {page_number}:")
            st.write(pdf_reader.pages[page_number - 1].extract_text())

            st.markdown("---")
        

if __name__ == '__main__':
    main()
