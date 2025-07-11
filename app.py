import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from fpdf import FPDF
from sentence_transformers import SentenceTransformer

# ✅ Configure Gemini API
genai.configure(api_key="YOUR_API_KEY")  # Replace with your real Gemini API key

# ✅ Load files
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_resume(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text(file):
    ext = file.name.split(".")[-1]
    if ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)
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
    score = 50  # base
    if 300 < word_count < 1200:
        score += 20
    score += min(30, len(matched_skills))
    return min(score, 100)

def get_gemini_feedback(resume, jd):
    prompt = f"""You're an AI resume assistant. Here's a resume:
---
{resume}
---
And here's a job description:
---
{jd}
---
Give bullet-point suggestions to improve the resume for this job. Rewrite one weak bullet point using strong action verbs."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def get_skill_diff(resume_text, jd_text):
    resume_words = set(clean_resume(resume_text).lower().split())
    jd_words = set(clean_resume(jd_text).lower().split())
    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words
    return list(matched), list(missing)

def generate_pdf(category, score, sbert, ats, matched, missing, feedback):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="TejMatch Resume Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Category: {category}", ln=True)
    pdf.cell(200, 10, txt=f"TF-IDF Match Score: {score}%", ln=True)
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
    pdf.multi_cell(0, 10, txt="Gemini Suggestions:\n" + feedback)

    pdf.output("resume_feedback.pdf")

def main():
    st.set_page_config("TejMatch – Smart Resume Analyzer", layout="wide")
    st.title("💼 TejMatch – AI Resume Analyzer")
    st.write("Upload your resume, paste a job description, and receive intelligent feedback with scores.")

    uploaded = st.file_uploader("📤 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("📝 Paste Job Description")

    if uploaded and jd_text:
        try:
            resume_text = extract_text(uploaded)
            st.success("✅ Resume extracted successfully")

            category, resume_vec = predict_category(resume_text)
            st.markdown(f"### 🔍 Predicted Job Category: `{category}`")

            match_score = get_match_score(resume_vec, jd_text)
            sbert_score = get_sbert_score(resume_text, jd_text)
            matched, missing = get_skill_diff(resume_text, jd_text)
            ats_score = get_ats_score(resume_text, matched)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 TF-IDF Match Score", f"{match_score}%")
                st.progress(match_score)
            with col2:
                st.metric("🧠 SBERT Semantic Score", f"{sbert_score}%")
                st.progress(sbert_score)

            st.markdown(f"### 📊 ATS Score: **{ats_score}/100**")

            st.markdown("### ✅ Matched Skills")
            st.write(", ".join(matched[:15]) or "None")
            st.markdown("### ❌ Missing Skills")
            st.write(", ".join(missing[:15]) or "None")

            st.markdown("### 🤖 Gemini Suggestions")
            with st.spinner("Contacting Gemini..."):
                feedback = get_gemini_feedback(resume_text, jd_text)
            st.markdown(feedback)

            if st.button("📥 Download Report as PDF"):
                generate_pdf(category, match_score, sbert_score, ats_score, matched, missing, feedback)
                with open("resume_feedback.pdf", "rb") as f:
                    st.download_button("Download PDF", f, file_name="TejMatch_Resume_Report.pdf")

        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()