import streamlit as st
import os
from pathlib import Path
from src.query_engine import MatchingEngine

st.title("GraphRAG Resume & Cover Letter Matcher")

# Config: where to store uploaded data
RESUME_DIR = "data/resumes"
COVER_LETTER_DIR = "data/cover_letters"
JOB_DESC_DIR = "data/job_descriptions"

for directory in [RESUME_DIR, COVER_LETTER_DIR, JOB_DESC_DIR]:
    os.makedirs(directory, exist_ok=True)

# Upload resume
st.header("1. Upload Your Resume (.txt)")
uploaded_resume = st.file_uploader("Choose resume file", type="txt", key="resume")

# Upload cover letter
st.header("2. Upload Your Cover Letter (.txt, optional)")
uploaded_cover = st.file_uploader("Choose cover letter file", type="txt", key="cover")

# Upload a job description to expand repo
st.header("3. (Optional) Upload a New Job Description to Repository")
uploaded_job = st.file_uploader("Add job description (.txt) to repo", type="txt", key="job")
if uploaded_job is not None:
    # Save job description to repo folder
    job_name = uploaded_job.name
    path = Path(JOB_DESC_DIR) / job_name
    with path.open("wb") as f:
        f.write(uploaded_job.read())
    st.success(f"Job description '{job_name}' uploaded to repository.")

# Start matching
if st.button("Match to Existing Jobs") and uploaded_resume is not None:
    resume_name = uploaded_resume.name
    resume_path = Path(RESUME_DIR) / resume_name
    with resume_path.open("wb") as f:
        f.write(uploaded_resume.read())
    st.success(f"Resume '{resume_name}' uploaded for matching.")

    cover_text = None
    if uploaded_cover is not None:
        cover_path = Path(COVER_LETTER_DIR) / uploaded_cover.name
        with cover_path.open("wb") as f:
            f.write(uploaded_cover.read())
        with cover_path.open("r", encoding="utf-8") as f:
            cover_text = f.read()

    # Load resume content
    with open(resume_path, "r", encoding="utf-8") as file:
        resume_text = file.read()
    # Load all job descriptions
    job_descs = []
    for jd_file in Path(JOB_DESC_DIR).glob("*.txt"):
        with jd_file.open("r", encoding="utf-8") as f:
            job_descs.append((jd_file.name, f.read()))

    # Create MatchingEngine and run
    engine = MatchingEngine(use_llm=False)  # set True if using LLM extraction
    # Add only this resume (with optional cover letter concatenated)
    temp_resume = resume_text
    if cover_text:
        temp_resume += "\n\n" + cover_text
    temp_resume_name = "user_resume.txt"
    with open(f"{RESUME_DIR}/{temp_resume_name}", "w", encoding="utf-8") as f:
        f.write(temp_resume)
    engine.load_candidates(RESUME_DIR)
    os.remove(f"{RESUME_DIR}/{temp_resume_name}")  # Clean up for repeat use

    results = []
    for job_name, jd_text in job_descs:
        res = engine.find_matches(jd_text, top_k=5, job_title=job_name)
        for match in res:
            if match.candidate_name == temp_resume_name or match.candidate_name == resume_name:
                results.append((job_name, match))

    if results:
        st.subheader("🔍 Match Results")
        for job, result in results:
            st.markdown(f"### {job}")
            st.write(f"**Match Score**: {result.match_score:.1%}")
            st.write("**Matched Skills:**", ', '.join(result.matched_skills))
            st.write("**Missing Skills:**", ', '.join(result.missing_skills))
            with st.expander("Explain reasoning"):
                st.write(result.reasoning)
            st.markdown("---")
    else:
        st.info("No matches found. Make sure your resume matches at least one job description in the knowledge repository.")
