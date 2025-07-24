import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎯 Feedback prompts
LOW_SCORE_FEEDBACK = [
    "Your resume lacks alignment with the job's core skills.",
    "Consider tailoring your resume to highlight relevant experience.",
    "The job description emphasizes skills not currently visible in your resume.",
    "Try rephrasing your achievements to match industry keywords.",
    "Your resume feels too generic for this role.",
    "Missing key technical terms that recruiters scan for.",
    "Add more role-specific accomplishments.",
    "Consider restructuring your resume to emphasize relevant sections.",
    "The resume doesn't reflect the job's seniority level.",
    "You may need to revise your summary to match the job tone."
]

MEDIUM_SCORE_FEEDBACK = [
    "You're on the right track—just a few tweaks needed.",
    "Some relevant skills are present, but not emphasized enough.",
    "Consider expanding on your experience with the listed tools.",
    "Your resume shows potential, but could use more specificity.",
    "Try aligning your bullet points with the job's responsibilities.",
    "Add measurable outcomes to strengthen your impact.",
    "You’ve got the foundation—now polish the presentation.",
    "Highlight your most relevant projects more clearly.",
    "Consider reordering sections to match job priorities.",
    "Your resume is decent, but lacks standout keywords."
]

HIGH_SCORE_FEEDBACK = [
    "Your resume aligns strongly with the job description!",
    "Excellent keyword match—this resume is recruiter-ready.",
    "Your experience and skills are a great fit for this role.",
    "Strong ATS compatibility and relevance.",
    "Your resume reflects the job's tone and expectations well.",
    "Great use of role-specific language.",
    "Your achievements match the job's core requirements.",
    "This resume is likely to pass initial screening.",
    "You’ve nailed the alignment—just keep it updated.",
    "Your resume shows clear readiness for this position."
]

def match_resume_to_job(resume_text, job_text):
    # Basic TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_score = round(score * 100)

    # Dummy category and ATS score
    category = "Software Engineering"
    ats_score = match_score - random.randint(0, 10)  # simulate ATS drop

    # Fake keyword matching
    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())
    matched = list(resume_words & job_words)
    missing = list(job_words - resume_words)

    # Select feedback based on score
    if match_score < 50:
        feedback = random.choice(LOW_SCORE_FEEDBACK)
    elif match_score < 75:
        feedback = random.choice(MEDIUM_SCORE_FEEDBACK)
    else:
        feedback = random.choice(HIGH_SCORE_FEEDBACK)

    return category, match_score, ats_score, matched[:10], missing[:10], feedback
