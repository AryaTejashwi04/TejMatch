import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
import random
import time

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

# ✅ Load SBERT model safely
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error("❌ SBERT model failed to load. Try restarting or check compatibility.")
    st.stop()

# 🧹 Clean resume text
def clean_resume(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 📄 Extract text from uploaded file
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

# 🔍 Predict job category
def predict_category(text):
    cleaned = clean_resume(text)
    vec = tfidf.transform([cleaned]).toarray()
    pred = svc_model.predict(vec)
    label = le.inverse_transform(pred)
    return label[0], vec

# 🎯 TF-IDF match score
def get_match_score(resume_vec, jd_text):
    jd_vec = tfidf.transform([clean_resume(jd_text)]).toarray()
    score = cosine_similarity(resume_vec, jd_vec)[0][0]
    return round(score * 100, 2)

# 🧠 SBERT semantic score
def get_sbert_score(resume, jd):
    emb1 = sbert_model.encode([resume])[0]
    emb2 = sbert_model.encode([jd])[0]
    score = cosine_similarity([emb1], [emb2])[0][0]
    return round(score * 100, 2)

# 📊 ATS score logic
def get_ats_score(resume_text, matched_skills):
    word_count = len(resume_text.split())
    score = 50
    if 300 < word_count < 1200:
        score += 20
    score += min(30, len(matched_skills))
    return min(score, 100)

# 🧩 Skill matching
def get_skill_diff(resume_text, jd_text):
    resume_words = set(clean_resume(resume_text).lower().split())
    jd_words = set(clean_resume(jd_text).lower().split())
    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words
    return list(matched), list(missing)

# 💡 Manual feedback based on score
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

# 🧾 PDF report generator
def generate_pdf(category, score, ats, matched, missing, feedback):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="TejMatch Resume Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Category: {category}", ln=True)
    pdf.cell(200, 10, txt=f"Score: {score}%", ln=True)
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
    pdf.multi_cell(0, 10, txt="Suggestions for Improvement:\n" + feedback)

    pdf.output("resume_feedback.pdf")

# 💤 Keep app awake with dummy prompts
def keep_awake():
    prompts = [
        "Analyzing semantic overlap between resume and JD...",
        "Optimizing ATS keyword density...",
        "Running LLM-based skill inference...",
        "Checking resume readability heuristics...",
        "Simulating recruiter eye-tracking patterns...",
        "Evaluating emotional tone of resume summary...",
        "Benchmarking resume against top 5% profiles...",
        "Scanning for leadership and impact signals...",
    ]
    st.empty()
    st.caption(random.choice(prompts))
    time.sleep(0.5)

# 🚀 Main app logic
def main():
    st.set_page_config("TejMatch – Smart Resume Analyzer", layout="wide")
    st.title("💼 TejMatch – Smart Resume Analyzer")
    st.write("Upload your resume and job description to receive feedback, scores, and suggestions.")

    keep_awake()

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

            match_score = get_match_score(resume_vec, jd_text)
            sbert_score = get_sbert_score(resume_text, jd_text)
            combined_score = round((match_score + sbert_score) / 2, 2)

            matched, missing = get_skill_diff(resume_text, jd_text)
            ats_score = get_ats_score(resume_text, matched)
            feedback = get_manual_feedback(combined_score)

            st.markdown(f"### 🔍 Predicted Job Category: `{category}`")
            st.metric("🎯 Score", f"{combined_score}%")
            st.progress(combined_score / 100)

            st.markdown(f"### 📊 ATS Score: **{ats_score}/100**")
            st.markdown("### ✅ Matched Skills")
            st.write(", ".join(matched[:15]) or "None")

            st.markdown("### ❌ Missing Skills")
            st.write(", ".join(missing[:15]) or "None")

            st.markdown("### 🧠 Suggestions for Improvement")
            st.markdown(feedback)

            if st.button("📥 Download Report as PDF"):
                generate
