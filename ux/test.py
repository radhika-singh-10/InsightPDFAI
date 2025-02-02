import os
import PyPDF2
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain,LLMChain, SimpleSequentialChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


model_name = "google/flan-t5-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
hf_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

# model_name = "bigscience/bloomz-560m"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
prompt_template = PromptTemplate(template="Summarize the following text:\n{text}", input_variables=["text"])
summary_chain = LLMChain(llm=llm, prompt=prompt_template)
st.title("PDF Summarizer with LangChain")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text (Preview)")
    st.write(pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text)
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = summary_chain.run({"text": pdf_text[:3000]}) 
            st.subheader("Summary")
            st.write(summary)
