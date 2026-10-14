import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_text_from_file, clean_text

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.title("AI Resume Screening & Candidate Ranking System")
st.write("Upload candidate resumes against a Job Description to rank top applicants.")

# Sidebar: Job Description Input
st.sidebar.header("Job Description")
jd_input = st.sidebar.text_area(
    "Paste the Job Description (JD) here:",
    height=300,
    placeholder="e.g., Seeking a Python Developer with experience in SQL, Machine Learning, Flask, and REST APIs..."
)

# Main Section: Upload Resumes
st.header("Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF, DOCX, TXT):",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if st.button("Run Screening"):
    if not jd_input.strip():
        st.warning("Please provide a valid Job Description in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one candidate resume.")
    else:
        with st.spinner("Processing resumes and computing relevance scores..."):
            cleaned_jd = clean_text(jd_input)
            
            resume_texts = []
            candidate_names = []
            
            for file in uploaded_files:
                bytes_data = file.read()
                raw_text = extract_text_from_file(bytes_data, file.name)
                processed_text = clean_text(raw_text)
                
                resume_texts.append(processed_text)
                candidate_names.append(file.name)
            
            # Combine JD and Resumes into a corpus for TF-IDF Vectorization
            corpus = [cleaned_jd] + resume_texts
            
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Compute Cosine Similarity between JD (index 0) and each Resume (indices 1 to N)
            jd_vector = tfidf_matrix[0]
            resume_vectors = tfidf_matrix[1:]
            
            similarity_scores = cosine_similarity(jd_vector, resume_vectors)[0]
            
            # Format Results into DataFrame
            results = []
            for name, score in zip(candidate_names, similarity_scores):
                percentage = round(score * 100, 2)
                results.append({
                    "Candidate File": name,
                    "Match Score (%)": percentage
                })
            
            df = pd.DataFrame(results)
            df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
            df.index += 1
            
            st.success("Screening completed successfully!")
            
            # Display Rankings Table
            st.subheader("Candidate Rankings")
            st.dataframe(df, use_container_width=True)
            
            # Download Results CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Ranking Report (CSV)",
                data=csv_data,
                file_name="resume_ranking_report.csv",
                mime="text/csv"
            )
