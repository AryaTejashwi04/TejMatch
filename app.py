import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import io
from PyPDF2 import PdfReader

# -------------------------
#  PDF Extractor
# -------------------------
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        text = ""
    return text

# -------------------------
#  Feedback Suggestions
# -------------------------
LOW_SCORE_FEEDBACK = [
    # 20 items for low score
    "Your resume lacks alignment with the job's core skills.",
    "Consider tailoring your resume to highlight relevant experience.",
    "The job description emphasizes skills not currently visible in your resume.",
    "Try rephrasing your achievements to match industry keywords.",
    "Your resume feels too generic for this role.",
    "Missing key technical terms that recruiters scan for.",
    "Add more role-specific accomplishments.",
    "Consider restructuring your resume to emphasize relevant sections.",
    "The resume doesn't reflect the job's seniority level.",
    "You may need to revise your summary to match the job tone.",
    "Provide more examples of relevant project work.",
    "Quantify your achievements (numbers, metrics) for impact.",
    "Address gaps between required and listed skills.",
    "Optimize formatting for ATS parsing.",
    "Highlight certifications or training relevant to the role.",
    "Simplify overly technical jargon for clarity.",
    "Prioritize critical keywords from the job post.",
    "Remove unrelated experiences to keep it focused.",
    "Add an opening summary aligned with job requirements.",
    "Emphasize transferable skills relevant to this role."
]

MEDIUM_SCORE_FEEDBACK = [
    # 20 items for medium score
    "You're on the right track—just a few tweaks needed.",
    "Some relevant skills are present, but not emphasized enough.",
    "Consider expanding on your experience with the listed tools.",
    "Your resume shows potential, but could use more specificity.",
    "Try aligning your bullet points with the job's responsibilities.",
    "Add measurable outcomes to strengthen your impact.",
    "You’ve got the foundation—now polish the presentation.",
    "Highlight your most relevant projects more clearly.",
    "Consider reordering sections to match job priorities.",
    "Your resume is decent, but lacks standout keywords.",
    "Improve clarity by reducing redundant points.",
    "Emphasize leadership or collaboration achievements.",
    "Add missing certifications or training references.",
    "Refine technical skills list for precision.",
    "Ensure consistent formatting throughout sections.",
    "Include tools or frameworks mentioned in job description.",
    "Add examples of problem-solving or innovation.",
    "Enhance summary to reflect your target role clearly.",
    "Highlight recent accomplishments more prominently.",
    "Align terminology with the employer’s language."
]

HIGH_SCORE_FEEDBACK = [
    # 20 items for high score
    "Your resume aligns strongly with the job description!",
    "Excellent keyword match—this resume is recruiter-ready.",
    "Your experience and skills are a great fit for this role.",
    "Strong ATS compatibility and relevance.",
    "Your resume reflects the job's tone and expectations well.",
    "Great use of role-specific language.",
    "Your achievements match the job's core requirements.",
    "This resume is likely to pass initial screening.",
    "You’ve nailed the alignment—just keep it updated.",
    "Your resume shows clear readiness for this position.",
    "Keep this version as a strong template for similar roles.",
    "Only minor refinements needed for perfection.",
    "Your key skills are well highlighted.",
    "Project examples directly support the job demands.",
    "Keywords and phrasing are spot on.",
    "ATS scanning should rank this highly.",
    "Formatting is clean and recruiter-friendly.",
    "Your summary effectively positions you for the role.",
    "This resume should stand out among applicants.",
    "Great job—focus next on interview preparation."
]

# -------------------------
#  Matching Function
# -------------------------
def match_resume_to_job(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_score = round(score * 100)

    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())
    matched = list(resume_words & job_words)
    missing = list(job_words - resume_words)

    if match_score < 50:
        feedback = random.sample(LOW_SCORE_FEEDBACK, 3)
    elif match_score < 75:
        feedback = random.sample(MEDIUM_SCORE_FEEDBACK, 3)
    else:
        feedback = random.sample(HIGH_SCORE_FEEDBACK, 3)

    return match_score, matched[:10], missing[:10], feedback

# -------------------------
#  Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="wide")

st.title("📄 AI Resume Match Analyzer")
st.markdown("<h4 style='text-align:center;color:#4CAF50;'>Analyze your resume against job description instantly</h4>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume Input")
    resume_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])
    resume_text = st.text_area("OR Paste Resume Text", height=200)

with col2:
    st.subheader("Job Description Input")
    job_pdf = st.file_uploader("Upload Job Description PDF", type=["pdf"])
    job_text = st.text_area("OR Paste Job Description", height=200)

# Extract text from PDFs if uploaded
if resume_pdf is not None:
    resume_text = extract_text_from_pdf(resume_pdf)
if job_pdf is not None:
    job_text = extract_text_from_pdf(job_pdf)

# Analyze Button
if st.button("Analyze Match"):
    if resume_text and job_text:
        score, matched, missing, feedback = match_resume_to_job(resume_text, job_text)

        # Show Score
        st.markdown(f"<h2 style='text-align:center;color:#2196F3;'>Your Score: {score}/100</h2>", unsafe_allow_html=True)

        # Suggestions
        st.subheader("Suggestions for Improvements:")
        for f in feedback:
            st.markdown(f"- {f}")

        # Keyword Analysis
        with st.expander("Keyword Analysis"):
            st.markdown(f"**Matched Keywords:** {', '.join(matched) if matched else 'None'}")
            st.markdown(f"**Missing Keywords:** {', '.join(missing) if missing else 'None'}")
    else:
        st.error("Please provide resume and job description (PDF or text).")
