# AI-Resume-Screening-System
for Edufyi tech solutions intern project
# AI Resume Screening & Ranking System

An automated, lightweight decision-support tool built for recruiters to analyze resumes and rank candidates based on their relevance to a given Job Description (JD).

Developed for **Edufyi Tech Solutions** Internship Project.

---

## 📌 Project Overview

Manually reviewing hundreds of resumes for a single job opening is time-consuming, repetitive, and prone to human error. This system automates the preliminary screening process by:
- Extracting text from candidate resumes in multiple formats (`.pdf`, `.docx`, `.txt`).
- Cleaning and pre-processing textual data.
- Converting job requirements and resume text into TF-IDF vector representations.
- Calculating match scores using Cosine Similarity.
- Ranking candidates in real-time and exporting results to a CSV report.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Frontend / UI:** Streamlit
- **NLP & ML:** Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity)
- **Document Parsing:** PyPDF, python-docx
- **Data Handling:** Pandas

---

## 📂 Project Structure

```text
AI-Resume-Screening-System/
├── app.py              # Main Streamlit application UI & workflow
├── utils.py            # File parsing & text preprocessing module
├── requirements.txt    # Project dependencies
├── .gitignore          # Environment & temporary file exclusions
└── README.md           # Documentation
