import streamlit as st
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader


model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
st.title("Document Question Answering (Free Model)")
st.write("Upload your course materials (PDFs, DOCX) and ask questions based on the content.")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
def load_documents(files):
    """Extracts text from uploaded PDF and DOCX files."""
    text_data = ""
    for file in files:
        temp_file_path = f"temp_{file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.getvalue())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        else:
            st.warning(f"Unsupported file format: {file.name}")
            continue
        docs = loader.load()
        for doc in docs:
            text_data += doc.page_content + "\n"

    return text_data

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        document_text = load_documents(uploaded_files)

    if document_text:
        st.subheader("Extracted Document Text (Preview)")
        st.write(document_text[:1000] + "...")  
        st.subheader("Ask a question about the uploaded documents")
        user_question = st.text_input("Your Question")

        if st.button("Get Answer") and user_question:
            prompt = f"""
            You are a helpful assistant. Answer the following question based on the document content below:

            Document: {document_text}  # Limiting input length to avoid token limit issues
            Question: {user_question}

            Answer:
            """
            with st.spinner("Generating answer..."):
                response = llm.predict(prompt)
            st.subheader("Answer")
            st.write(response)
    else:
        st.warning("No valid text extracted. Please upload valid PDF or DOCX files.")
else:
    st.info("Please upload course materials to begin.")
