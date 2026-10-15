import streamlit as st
import pandas as pd
import numpy as np
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

def calculate_calibrated_score(raw_similarity, jd_text, resume_text):
    """
    Calibrates raw Cosine Similarity into a realistic HR matching score (0-100%)
    by combining TF-IDF vector similarity with direct key-phrase coverage.
    """
    if raw_similarity <= 0:
        return 0.0

    # 1. Non-linear scaling curve for raw Cosine Similarity
    # Maps low raw math scores (0.15 - 0.35) smoothly to competitive HR ranges (50% - 85%)
    scaled_similarity = np.power(raw_similarity, 0.45) * 100

    # 2. Key Keyword Overlap Bonus
    jd_words = set(clean_text(jd_text).split())
    resume_words = set(clean_text(resume_text).split())
    
    # Filter out short/common words
    jd_keywords = {w for w in jd_words if len(w) > 3}
    if jd_keywords:
        matched_keywords = jd_keywords.intersection(resume_words)
        keyword_overlap_ratio = len(matched_keywords) / len(jd_keywords)
    else:
        keyword_overlap_ratio = 0.0

    # 3. Weighted Final Score: 70% TF-IDF Cosine Similarity + 30% Keyword Coverage
    final_score = (scaled_similarity * 0.70) + (keyword_overlap_ratio * 100 * 0.30)
    
    # Cap at 98.5% for realistic bounds
    return round(float(np.clip(final_score, 0.0, 98.5)), 2)


if st.button("Run Screening"):
    if not jd_input.strip():
        st.warning("Please provide a valid Job Description in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one candidate resume.")
    else:
        with st.spinner("Processing resumes and computing relevance scores..."):
            cleaned_jd = clean_text(jd_input)
            
            raw_resume_texts = []
            cleaned_resume_texts = []
            candidate_names = []
            
            for file in uploaded_files:
                bytes_data = file.read()
                raw_text = extract_text_from_file(bytes_data, file.name)
                processed_text = clean_text(raw_text)
                
                raw_resume_texts.append(raw_text)
                cleaned_resume_texts.append(processed_text)
                candidate_names.append(file.name)
            
            # Corpus for TF-IDF
            corpus = [cleaned_jd] + cleaned_resume_texts
            
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            jd_vector = tfidf_matrix[0]
            resume_vectors = tfidf_matrix[1:]
            
            raw_similarities = cosine_similarity(jd_vector, resume_vectors)[0]
            
            # Calculate calibrated HR scores
            results = []
            for name, raw_sim, raw_resume in zip(candidate_names, raw_similarities, raw_resume_texts):
                score = calculate_calibrated_score(raw_sim, jd_input, raw_resume)
                results.append({
                    "Candidate File": name,
                    "Match Score (%)": f"{score:.2f}%"
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
