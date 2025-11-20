
from src.query_engine import MatchingEngine
from pathlib import Path

def main():
    print("Initializing Matching Engine...")
    engine = MatchingEngine()
    
    print("Loading candidates from Data...")
    engine.load_candidates("Data")
    
    print(f"Loaded {len(engine.candidate_graphs)} candidates.")
    
    # Use a sample job description
    job_description = """
    Senior Data Scientist
    
    Requirements:
    - Experience with Python and Machine Learning
    - Knowledge of NLP and Transformers
    - SQL optimization skills
    - Tableau experience
    """
    
    print("\nFinding matches for 'Senior Data Scientist'...")
    results = engine.find_matches(job_description, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.candidate_name} - Score: {result.match_score:.1%}")
        print(f"   Reason: {result.reasoning}")
        print(f"   Matched Skills: {', '.join(result.matched_skills)}")
        print(f"   Missing Skills: {', '.join(result.missing_skills)}")

if __name__ == "__main__":
    main()
