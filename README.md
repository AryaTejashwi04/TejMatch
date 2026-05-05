TejMatch
TejMatch is a high-performance resume-to-job-description matching platform designed for intelligent semantic alignment and relationship mapping. Utilizing a hybrid machine learning architecture, the system provides high-speed similarity searches and deep skill-gap analysis to streamline technical recruitment.

🚀 Key Features
Hybrid Rating Engine: Combines TF-IDF Keyword Matching (40%) and SBERT Semantic Embeddings (60%) for a balanced assessment of both literal keywords and conceptual context.

Dual-Index Pipeline Architecture:

L1 Local Processing: Instantaneous similarity computation using all-MiniLM-L6-v2 to minimize latency and API overhead.

L2 Dynamic LLM Analysis: Leverages Gemini 1.5 Flash for high-fidelity qualitative insights, missing skill identification, and mentorship-focused advice.

Recursive Sentiment-Tone Logic: Adjusts the AI’s feedback style (Highly Critical vs. Mentorship-focused) based on the calculated match score to provide realistic recruitment scenarios.

Research-Validated Infrastructure: Integrated benchmarks using SVC, KNN, and Random Forest, with SMOTE implementation to handle class imbalance during the training/benchmarking phase.

Multi-Format Ingestion: Supports direct PDF extraction via PyPDF2 alongside raw text input for versatile data handling.

🔬 Tech Stack
Core LLM: Google Gemini 1.5 Flash

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

Machine Learning: Scikit-learn (TF-IDF, SVC, KNN), Imbalanced-learn (SMOTE)

Interface: Streamlit with dynamic status-pill reporting

📦 Installation
Install Dependencies:

Bash
pip install streamlit pandas numpy PyPDF2 scikit-learn imbalanced-learn sentence-transformers google-generativeai
Configure Secrets:
Add your GEMINI_KEY to your Streamlit secrets management (.streamlit/secrets.toml).

Launch:

Bash
    streamlit run app.py
    ```

---

**Developer**
Tejashwi Arya
