import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    return response["output_text"]


def main():
    st.set_page_config(page_title="🌈 Chat with your PDF, Model, or Image 💬")

    st.title("Gemini Application")

    st.sidebar.title("Select Option")
    option = st.sidebar.selectbox(
        "",
        ["Answer from PDF", "Answer from Your Model", "Answer from Image"]
    )

    if option == "Answer from PDF":
        st.header("Answer from PDF")
        pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if pdf_docs:
            if st.sidebar.button("Process", key="process_button"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Complete! 🎉")

        user_question = st.text_input("Ask a Question from the PDF Files")
        if st.button("Ask"):
            if user_question:
                response = user_input(user_question)
                st.write("Reply:", response)

    elif option == "Answer from Your Model":
        st.header("Answer from Your Model")
        input_text = st.text_input("Input text:")
        if st.button("Ask"):
            if input_text:
                model_response = get_gemini_response(input_text)
                st.write("Reply:", model_response)

    elif option == "Answer from Image":
        st.header("Answer from Image")
        st.sidebar.subheader("Choose an Image")
        uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        input_prompt = st.text_input("Input Prompt:")
        image = ""
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Tell me about the image"):
            if input_prompt:
                response = get_gemini_response(input_prompt, image)
                st.subheader("The Response is")
                st.write(response)


def get_gemini_response(input_text, image=None):
    model = genai.GenerativeModel('gemini-pro')
    if input_text != "":
        response = model.generate_content(input_text)
        return response.text


if __name__ == "__main__":
    main()
