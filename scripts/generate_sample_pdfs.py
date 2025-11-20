
from fpdf import FPDF
from pathlib import Path
import os

def create_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Handle unicode issues by replacing common problematic chars if strictly ascii is needed, 
    # but FPDF basic usage is fine for simple text. 
    # We'll split by newlines to keep formatting roughly.
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.output(filename)
    print(f"Created {filename}")

def main():
    base_dir = Path("Data")
    (base_dir / "resumes").mkdir(parents=True, exist_ok=True)
    (base_dir / "cover_letters").mkdir(parents=True, exist_ok=True)

    resume_text = """
    Alice Wonder
    Senior Data Scientist
    
    Experience:
    - Data Scientist at BigData Corp (2020-Present)
      - Implemented NLP models using Transformers
      - Optimized SQL queries for petabyte-scale data
    
    - Analyst at DataInc (2018-2020)
      - Python and R for statistical analysis
      - Tableau dashboards
      
    Skills: Python, SQL, NLP, TensorFlow, Tableau, Git
    Education: MS in Computer Science
    """
    
    cover_letter_text = """
    Dear Hiring Manager,
    
    I am writing to express my interest in the Senior Data Scientist position. 
    With 5 years of experience in Python and Machine Learning, I believe I am a great fit.
    I have worked extensively with SQL and NLP technologies.
    
    Sincerely,
    Alice Wonder
    """

    create_pdf(resume_text, str(base_dir / "resumes/alice_wonder.pdf"))
    create_pdf(cover_letter_text, str(base_dir / "cover_letters/alice_wonder_cl.pdf"))

if __name__ == "__main__":
    main()
