import streamlit as st
import requests


# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF Q&A Assistant", page_icon=":books:")
st.title("PDF Question Answering Assistant")

def main():
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        try:
            # Send file to FastAPI backend
            with st.spinner("Uploading and processing file..."):
                response = requests.post(
                    f"{BACKEND_URL}/upload-pdf/",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"PDF processed successfully! Pages: {data['pages']}, Chunks: {data['chunks']}")
                else:
                    st.error(response.json().get("error", "Unknown error"))
        except Exception as e:
            st.error(f"Error uploading file: {e}")

    # Question answering
    question = st.text_input("Ask a question about the document:")
    if question:
        try:
            with st.spinner("Generating answer..."):
                response = requests.post(
                    f"{BACKEND_URL}/ask-question/",
                    data={"question": question},
                )
                if response.status_code == 200:
                    st.subheader("Answer:")
                    st.write(response.json().get("answer"))
                else:
                    st.error(response.json().get("error", "Unknown error"))
        except Exception as e:
            st.error(f"Error fetching answer: {e}")

if __name__ == "__main__":
    main()
