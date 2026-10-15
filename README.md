# AI Resume Screening & Ranking System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-resume-screening-system.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated, lightweight decision-support tool built for recruiters to analyze candidate resumes and rank applicants based on their relevance to a given Job Description (JD).

Developed for **Edufyi Tech Solutions** Internship Project.

---

## 🚀 Live Demo

👉 **Try the Live Web App:** [https://ai-resume-screening-system.streamlit.app/](https://ai-resume-screening-system.streamlit.app/)

*(Note: Replace the link above with your exact Streamlit app URL)*

---

## 📌 Project Overview

Manually reviewing hundreds of resumes for a single job opening is time-consuming and prone to human error. This system automates preliminary candidate screening by:
- Extracting text directly from uploaded candidate resumes (`.pdf`, `.docx`, `.txt`).
- Cleaning and standardizing textual data using regular expressions.
- Vectorizing job requirements and candidate profiles using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- Evaluating relevance scores using **Cosine Similarity**.
- Displaying real-time candidate rankings and providing exportable CSV reports for HR records.

---

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3.10+
- **Frontend / UI:** Streamlit Cloud
- **NLP & ML Engine:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)
- **Document Parsers:** PyPDF, python-docx
- **Data Structuring:** Pandas

---

## 📂 Repository Structure

```text
AI-Resume-Screening-System/
├── app.py              # Main Streamlit web application & ranking UI
├── utils.py            # File extraction & text normalization helpers
├── requirements.txt    # Python dependencies for deployment
├── README.md           # Documentation & Live Demo link
└── .gitignore          # Git exclusion rules
