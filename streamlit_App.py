import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Form Data Analyzer", layout="wide")

st.markdown("""
## Form Data Analyzer: Analyze and gain insights from your Form Responses

This chatbot is built to help analyze form responses (Microsoft Forms or Google Forms) and provide insights based on user queries. You can upload form response data as CSV or Excel files, and the chatbot will answer your questions based on the form data.

### How It Works

1. *Enter Your API Key*: You'll need a Google API key to use Google's Generative AI models. Obtain your API key [here](https://makersuite.google.com/app/apikey).
2. *Upload Your Form Data*: The system accepts form data exported in CSV or Excel format.
3. *Ask a Question*: After processing the data, ask any question related to the content for precise insights.
""")

api_key = st.text_input("Enter your Google API Key:", type="password", key="AIzaSyDC6kvPz3YMPfvpvXqXEo2WHCb8mvkgmP8")

def read_form_data(uploaded_file):
    """Reads uploaded form data and returns a Pandas DataFrame"""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

def process_form_data(df):
    """Converts the form DataFrame into a string for text processing"""
    return df.to_string()

def get_text_chunks(text):
    """Splits the text into chunks for embedding"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Generates a vector store from the text chunks using Google's Generative AI Embeddings"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up the conversational chain with a custom prompt for analyzing form data"""
    prompt_template = """
    You are an expert at analyzing form data. Answer the user's questions based on the following form data context.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    """Handles user queries and returns answers from the processed form data"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.header("Form Data Analyzer Chatbot")

    user_question = st.text_input("Ask a question about the uploaded form data", key="user_question")

    with st.sidebar:
        st.title("Upload Form Data")
        uploaded_file = st.file_uploader("Upload your form responses (CSV or Excel)", type=["csv", "xlsx"])

        if uploaded_file and api_key:
            with st.spinner("Processing..."):
                df = read_form_data(uploaded_file)
                if df is not None:
                    # Display the data for user reference
                    st.write("Uploaded Data:")
                    st.dataframe(df)

                    # Process the data into text and create vector embeddings
                    form_text = process_form_data(df)
                    text_chunks = get_text_chunks(form_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Form data has been successfully processed and stored.")

    # Handle user query
    if user_question and api_key:
        with st.spinner("Getting the answer..."):
            answer = user_input(user_question, api_key)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
