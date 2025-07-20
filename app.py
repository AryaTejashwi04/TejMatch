import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
import torch
import random

# 🔒 Hide GitHub & Streamlit UI clutter
st.markdown("""
    <style>
    #MainMenu, footer, header, .viewerBadge_container__1QSob {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Load models
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# ✅ Load SBERT model
device = torch.device('cpu')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model.to(device)

# 📁 Resume cleaner
def clean_resume(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 📤 Text extractor for PDF/DOCX/TXT
def extract_text(file):
    ext = file.name.split(".")[-1]
    if ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext == "docx":
        return "\n".join([para.text for para in docx.Document(file).paragraphs])
    elif ext == "txt":
        return file.read().decode('utf-8', 'ignore')
    else:
        raise ValueError("Unsupported file type")

def predict_category(text):
    cleaned = clean_resume(text)
    vec = tfidf.transform([cleaned]).toarray()
    pred = svc_model.predict(vec)
    label = le.inverse_transform(pred)
    return label[0], vec

def get_match_score(resume_vec, jd_text):
    jd_vec = tfidf.transform([clean_resume(jd_text)]).toarray()
    score = cosine_similarity(resume_vec, jd_vec)[0][0]
    return round(score * 100, 2)

def get_sbert_score(resume, jd):
    emb1 = sbert_model.encode([resume])[0]
    emb2 = sbert_model.encode([jd])[0]
    score = cosine_similarity([emb1], [emb2])[0][0]
    return round(score * 100, 2)

def get_ats_score(resume_text, matched_skills):
    word_count = len(resume_text.split())
    score = 50
    if 300 < word_count < 1200:
        score += 20
    score += min(30, len(matched_skills))
    return min(score, 100)

def get_skill_diff(resume_text, jd_text):
    resume_words = set(clean_resume(resume_text).lower().split())
    jd_words = set(clean_resume(jd_text).lower().split())
    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words
    return list(matched), list(missing)

# 🔍 Manual feedback based on score
def get_manual_feedback(score):
    low = [
        "Add measurable achievements like 'Increased revenue by 25%'.",
        "Use action verbs such as 'Led', 'Built', 'Executed'.",
        "Tailor resume to match keywords in the job description.",
        "Include certifications, courses, or licenses.",
        "Avoid vague phrases like 'hardworking' or 'quick learner'.",
        "Add technical tools or domain-specific software.",
        "Break paragraphs into readable bullet points.",
        "Add job titles with clear dates and locations.",
    ]
    medium = [
        "Use bullet points that emphasize quantifiable impact.",
        "Mention specific tools used (e.g., Excel, Tableau).",
        "Add a short summary section at the top.",
        "Reorder items to highlight the most relevant experience.",
        "Add links to online portfolios or profiles.",
        "Clarify job titles and responsibilities clearly.",
        "Merge duplicate skills to reduce redundancy.",
        "Keep verb tense consistent across sections.",
    ]
    high = [
        "Add recent projects or accomplishments to keep resume fresh.",
        "Include leadership, mentoring or team collaboration examples.",
        "Use a clean format that’s ATS-friendly and consistent.",
        "Ensure proper font sizes and spacing throughout.",
        "Mention preferred job types or industries.",
        "Add a personal branding statement or tagline.",
        "Convert resume to PDF before submitting.",
        "Align sections visually to improve readability.",
    ]

    if score <= 40:
        tips = random.sample(low, 2)
    elif score <= 70:
        tips = random.sample(medium, 2)
    else:
        tips = random.sample(high, 2)

    return "\n".join(f"- {tip}" for tip in tips)

# 📄 PDF generator
def generate_pdf(category, score, sbert, ats, matched, missing, feedback):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="TejMatch Resume Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Category: {category}", ln=True)
    pdf.cell(200, 10, txt=f"Your Score is: {score}%", ln=True)
    pdf.cell(200, 10, txt=f"SBERT Semantic Score: {sbert}%", ln=True)
    pdf.cell(200, 10, txt=f"ATS Score: {ats}/100", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Matched Skills:", ln=True)
    for skill in matched[:15]:
        pdf.cell(200, 10, txt=f"- {skill}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Missing Skills:", ln=True)
    for skill in missing[:15]:
        pdf.cell(200, 10, txt=f"- {skill}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="Suggested Improvements:\n" + feedback)

    pdf.output("resume_feedback.pdf")

# 🚀 Main UI
def main():
    st.set_page_config("TejMatch – Smart Resume Analyzer", layout="wide")
    st.title("💼 TejMatch – AI Resume Analyzer")
    st.write("Upload your resume and job description to receive intelligent feedback and match scores.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_resume = st.file_uploader("📤 Upload Resume", type=["pdf", "docx", "txt"])
    with col2:
        uploaded_jd = st.file_uploader("📥 Upload Job Description", type=["pdf", "docx", "txt"])

    if uploaded_resume and uploaded_jd:
        try:
            resume_text = extract_text(uploaded_resume)
            jd_text = extract_text(uploaded_jd)
            st.success("✅ Files extracted successfully")

            category, resume_vec = predict_category(resume_text)
            st.markdown(f"### 🔍 Predicted Job Category: `{category}`")

            match_score = get_match_score(resume_vec, jd_text)
            sbert_score = get_sbert_score(resume_text, jd_text)
            matched, missing = get_skill_diff(resume_text, jd_text)
            ats_score = get_ats_score(resume_text, matched)
            feedback = get_manual_feedback(match_score)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Your Score is:", f"{match_score}%")
                st.progress(match_score / 100)
            with col2:
                st.metric("🧠 SBERT Semantic Score", f"{sbert_score}%")
                st.progress(sbert_score / 100)

            st.markdown(f"### 📊 ATS Score: **{ats_score}/100**")

            st.markdown("### ✅ Matched Skills")
            st.write(", ".join(matched[:15]) or "None")
            st.markdown("### ❌ Missing Skills")
            st.write(", ".join(missing[:15]) or "None")

            st.markdown("### ✨ Suggestions for Improvement")
            st.markdown(feedback)

            if st.button("📥 Download Report as PDF"):
                generate_pdf(category, match_score, sbert_score, ats_score, matched, missing, feedback)
                with open("resume_feedback.pdf", "rb") as f:
                    st.download_button("Download PDF", f, file_name="TejMatch_Resume_Report.pdf")

        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
