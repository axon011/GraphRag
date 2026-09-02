"""
query_engine.py

GraphRAG Query Engine
Performs intelligent matching between candidates and job requirements using knowledge graphs.

Author: Your Name
Date: 2025
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import pypdf
from .extract_entities import Entity, EntityType, RegexEntityExtractor
from .build_knowledge_graph import ResumeGraphBuilder, KnowledgeGraph, RelationType
from .config import settings


@dataclass
class MatchResult:
    """Result of a candidate-job matching operation"""
    candidate_name: str
    job_title: str
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    additional_skills: List[str]
    reasoning: str
    graph_distance: int = 0


class MatchingEngine:
    """Main engine for performing candidate-job matching using GraphRAG"""
    
    def __init__(self, use_llm: Optional[bool] = None):
        """
        Initialize the matching engine
        
        Args:
            use_llm: Use LLM for entity extraction (more accurate but slower). If None, uses settings.
        """
        self.graph_builder = ResumeGraphBuilder(use_llm=use_llm)
        self.candidate_graphs: Dict[str, KnowledgeGraph] = {}
        self.job_graph: Optional[KnowledgeGraph] = None
        self.extractor = RegexEntityExtractor()
    
    def load_candidates(self, candidates_directory: str) -> None:
        """
        Load and process all candidate resumes
        
        Args:
            candidates_directory: Path to directory containing resume files
        """
        candidates_path = Path(candidates_directory)
        
        for resume_file in candidates_path.glob("*.txt"):
            with open(resume_file, 'r') as f:
                text = f.read()
            
            candidate_name = resume_file.stem
            graph = self.graph_builder.build_from_text(text, document_type="resume")
            self.candidate_graphs[candidate_name] = graph

        for resume_file in candidates_path.glob("*.pdf"):
            text = ""
            try:
                with open(resume_file, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error reading PDF {resume_file}: {e}")
                continue
            
            if text.strip():
                candidate_name = resume_file.stem
                graph = self.graph_builder.build_from_text(text, document_type="resume")
                self.candidate_graphs[candidate_name] = graph
    
    def load_job_description(self, job_file_path: str) -> None:
        """
        Load and process a job description
        
        Args:
            job_file_path: Path to job description file
        """
        with open(job_file_path, 'r') as f:
            text = f.read()
        
        self.job_graph = self.graph_builder.build_from_text(text, document_type="job_description")
    
    def extract_job_requirements(self, job_description: str) -> Dict:
        """
        Extract skills and requirements from job description
        
        Args:
            job_description: Job description text
        
        Returns:
            Dictionary with extracted requirements
        """
        entities = self.extractor.extract(job_description)
        
        requirements = {
            'required_skills': [],
            'required_roles': [],
            'required_certifications': [],
            'nice_to_have': []
        }
        
        # Parse sections
        lines = job_description.lower().split('\n')
        in_requirements = False
        in_nice_to_have = False
        
        for line in lines:
            if 'requirement' in line:
                in_requirements = True
                in_nice_to_have = False
                continue
            elif 'nice to have' in line or 'preferred' in line:
                in_nice_to_have = True
                in_requirements = False
                continue
            
            # Extract entities
            for entity in entities:
                entity_text = entity.text.lower()
                if entity_text in line:
                    if entity.type == EntityType.SKILL:
                        target_list = requirements['nice_to_have'] if in_nice_to_have else requirements['required_skills']
                        if entity_text not in [s.lower() for s in target_list]:
                            target_list.append(entity.text)
                    elif entity.type == EntityType.ROLE:
                        requirements['required_roles'].append(entity.text)
                    elif entity.type == EntityType.CERTIFICATION:
                        requirements['required_certifications'].append(entity.text)
        
        # Deduplicate
        for key in requirements:
            requirements[key] = list(set(requirements[key]))
        
        return requirements
    
    def calculate_skill_match_score(
        self, 
        required_skills: List[str], 
        candidate_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculate skill match score between required and candidate skills
        
        Args:
            required_skills: Skills required by job
            candidate_skills: Skills candidate has
        
        Returns:
            Tuple of (match_score, matched_skills, missing_skills)
        """
        if not required_skills:
            return 1.0, [], []
        
        matched = []
        missing = []
        
        for req_skill in required_skills:
            # Check for exact match
            if any(req_skill.lower() == cand.lower() for cand in candidate_skills):
                matched.append(req_skill)
            else:
                # Check for similar skills
                found_similar = False
                for cand_skill in candidate_skills:
                    # Create temporary graph to check similarity
                    temp_graph = KnowledgeGraph()
                    distance = temp_graph.get_skill_graph_distance(req_skill, cand_skill)
                    
                    if distance < 0.3:  # Similar enough
                        matched.append(f"{req_skill} (similar: {cand_skill})")
                        found_similar = True
                        break
                
                if not found_similar:
                    missing.append(req_skill)
        
        # Calculate score
        match_score = len(matched) / len(required_skills) if required_skills else 0.0
        
        return match_score, matched, missing
    
    def match_candidate_to_job(
        self, 
        candidate_name: str, 
        job_description: str,
        job_title: str = "Position"
    ) -> MatchResult:
        """
        Match a single candidate to a job
        
        Args:
            candidate_name: Name of candidate (key in candidate_graphs)
            job_description: Full job description text
            job_title: Title of the job position
        
        Returns:
            MatchResult with detailed scoring
        """
        if candidate_name not in self.candidate_graphs:
            raise ValueError(f"Candidate '{candidate_name}' not found in loaded candidates")
        
        candidate_graph = self.candidate_graphs[candidate_name]
        candidate_profile = candidate_graph.get_candidate_profile(candidate_name)
        candidate_skills = candidate_profile.get('skills', [])
        
        # Extract job requirements
        job_requirements = self.extract_job_requirements(job_description)
        required_skills = job_requirements['required_skills']
        
        # Calculate skill match
        match_score, matched_skills, missing_skills = self.calculate_skill_match_score(
            required_skills,
            candidate_skills
        )
        
        # Identify additional skills candidate has
        additional_skills = [
            s for s in candidate_skills 
            if s not in required_skills and s not in [m.split(' (similar:')[0] for m in matched_skills]
        ]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            match_score,
            matched_skills,
            missing_skills,
            candidate_profile,
            job_requirements
        )
        
        return MatchResult(
            candidate_name=candidate_name,
            job_title=job_title,
            match_score=match_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            additional_skills=additional_skills[:5],  # Top 5
            reasoning=reasoning
        )
    
    def find_matches(
        self, 
        job_description: str,
        top_k: int = 5,
        job_title: str = "Position"
    ) -> List[MatchResult]:
        """
        Find and rank all candidates for a job
        
        Args:
            job_description: Job description text
            top_k: Return top k matches
            job_title: Job position title
        
        Returns:
            Sorted list of top matching results
        """
        results = []
        
        for candidate_name in self.candidate_graphs.keys():
            try:
                result = self.match_candidate_to_job(
                    candidate_name, 
                    job_description, 
                    job_title
                )
                results.append(result)
            except Exception as e:
                print(f"Error matching {candidate_name}: {e}")
                continue
        
        # Sort by match score descending
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        return results[:top_k]
    
    def explain_match(
        self, 
        candidate_name: str, 
        job_description: str
    ) -> Dict:
        """
        Generate detailed explanation for why a candidate matches a job
        
        Args:
            candidate_name: Name of candidate
            job_description: Job description
        
        Returns:
            Detailed explanation dictionary
        """
        result = self.match_candidate_to_job(candidate_name, job_description)
        
        explanation = {
            'candidate': candidate_name,
            'overall_score': f"{result.match_score:.1%}",
            'summary': result.reasoning,
            'matched_skills': result.matched_skills,
            'missing_skills': result.missing_skills,
            'additional_strengths': result.additional_skills,
            'recommendation': self._get_recommendation(result.match_score),
            'advice': self._get_improvement_advice(result.missing_skills)
        }
        
        return explanation
    
    @staticmethod
    def _generate_reasoning(
        match_score: float,
        matched_skills: List[str],
        missing_skills: List[str],
        candidate_profile: Dict,
        job_requirements: Dict
    ) -> str:
        """Generate human-readable reasoning for match"""
        
        if match_score >= 0.9:
            reason = "Strong match. Candidate has most required skills."
        elif match_score >= 0.7:
            reason = "Good match. Candidate has majority of required skills."
        elif match_score >= 0.5:
            reason = "Moderate match. Candidate has half of required skills."
        else:
            reason = "Weak match. Candidate missing several key skills."
        
        if missing_skills:
            reason += f" Missing: {', '.join(missing_skills[:3])}."
        
        return reason
    
    @staticmethod
    def _get_recommendation(score: float) -> str:
        """Get recommendation based on score"""
        if score >= 0.9:
            return "STRONG RECOMMEND - Move to interview"
        elif score >= 0.75:
            return "RECOMMEND - Qualified candidate"
        elif score >= 0.5:
            return "CONSIDER - May need training in some areas"
        else:
            return "NOT RECOMMENDED - Significant skill gaps"
    
    @staticmethod
    def _get_improvement_advice(missing_skills: List[str]) -> str:
        """Get advice for candidate improvement"""
        if not missing_skills:
            return "No skill gaps identified."
        
        return f"Focus on developing: {', '.join(missing_skills[:3])} to strengthen candidacy."


if __name__ == "__main__":
    # Example usage
    print("=== Matching Engine Example ===\n")
    
    engine = MatchingEngine()
    
    # Example job description
    job_description = """
    Senior Python Developer
    
    Requirements:
    - 5+ years Python development experience
    - Docker and Kubernetes expertise
    - AWS cloud experience
    - REST API design
    - PostgreSQL database knowledge
    
    Nice to have:
    - Machine learning frameworks (PyTorch, TensorFlow)
    - GitHub Actions CI/CD
    - Microservices architecture
    """
    
    # Example candidate profile
    candidate_profile = """
    John Doe - Senior Developer
    
    Experience:
    Senior Python Developer at TechCorp (2021-2023)
    - Led Python microservices team
    - Built services with Docker and Kubernetes
    - AWS infrastructure management
    - RESTful API development
    
    Skills: Python, Docker, Kubernetes, AWS, PostgreSQL, PyTorch, GitHub Actions
    """
    
    # This would work with actual files loaded via load_candidates()
    # For demonstration, showing API structure
    
    print("Matching Engine initialized successfully!")
    print("Features:")
    print("  - load_candidates(directory): Load resumes")
    print("  - find_matches(job_desc): Find top candidates")
    print("  - explain_match(candidate, job_desc): Detailed explanation")
