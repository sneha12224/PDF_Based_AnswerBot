import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="PDF AnswerBot", layout="wide")
st.title("PDF AnswerBot")

# Initialize only once and store in session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()  # by default use_openai=False (local HF model)
    st.session_state.pdf_loaded = False
    st.session_state.total_tokens = 0

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.pdf_loaded:
    with st.spinner("Processing PDF..."):
        text = st.session_state.rag.load_pdf(uploaded_file)
        total_tokens = st.session_state.rag.create_vectorstore(text)
        st.session_state.pdf_loaded = True
        st.session_state.total_tokens = total_tokens
    st.success(f"✅ PDF uploaded successfully! Total tokens: {total_tokens}")

if st.session_state.pdf_loaded:
    st.info(f"📊 Total tokens in PDF: {st.session_state.total_tokens}")

    question = st.text_input("Ask a question from the PDF:")
    if st.button("Get Answer") and question:
        with st.spinner("🔎 Searching and generating answer..."):
            result = st.session_state.rag.query_with_metrics(question)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Answer:")
            st.write(result["answer"])

            # Show metrics
            st.markdown("### 📊 Response Metrics")
            st.write(f"⏱️ Response time: **{result['elapsed_s']} seconds**")
            st.write(f"🔤 Input tokens: **{result['input_tokens']}**")
            st.write(f"🔤 Output tokens: **{result['output_tokens']}**")
            st.write(f"🔢 Total tokens: **{result['total_tokens']}**")

            # Optional: show retrieved context
            with st.expander("🔎 Show retrieved context"):
                st.write(result["context"])
