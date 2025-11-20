"""
tests/test_matching.py

Unit tests for candidate-job matching engine
Tests entity extraction, graph building, and matching logic.

Author: Your Name
Date: 2025
"""

import unittest
from pathlib import Path
import tempfile

from src.extract_entities import Entity, EntityType, RegexEntityExtractor
from src.build_knowledge_graph import ResumeGraphBuilder, KnowledgeGraph, RelationType
from src.query_engine import MatchingEngine, MatchResult


class TestEntityExtraction(unittest.TestCase):
    """Test entity extraction functionality"""
    
    def setUp(self):
        self.extractor = RegexEntityExtractor()
    
    def test_extract_skills(self):
        """Test extraction of programming skills"""
        text = "Experienced with Python, Docker, and Kubernetes"
        entities = self.extractor.extract(text, [EntityType.SKILL])
        
        self.assertGreater(len(entities), 0)
        skill_texts = [e.text.lower() for e in entities]
        self.assertIn('python', skill_texts)
        self.assertIn('docker', skill_texts)
    
    def test_extract_roles(self):
        """Test extraction of job roles"""
        text = "Senior Software Engineer with 5 years experience"
        entities = self.extractor.extract(text, [EntityType.ROLE])
        
        self.assertGreater(len(entities), 0)
    
    def test_extract_duration(self):
        """Test extraction of time durations"""
        text = "Worked from 2020-2023 at TechCorp"
        entities = self.extractor.extract(text, [EntityType.DURATION])
        
        self.assertGreater(len(entities), 0)
    
    def test_extract_certifications(self):
        """Test extraction of certifications"""
        text = "AWS Certified Solutions Architect and CKA holder"
        entities = self.extractor.extract(text, [EntityType.CERTIFICATION])
        
        self.assertGreater(len(entities), 0)
    
    def test_deduplication(self):
        """Test that duplicate entities are removed"""
        text = "Python Python Python"
        entities = self.extractor.extract(text, [EntityType.SKILL])
        
        python_entities = [e for e in entities if 'python' in e.text.lower()]
        # Should not have multiple identical Python entities
        self.assertEqual(len(python_entities), 1)


class TestKnowledgeGraph(unittest.TestCase):
    """Test knowledge graph construction"""
    
    def setUp(self):
        self.graph = KnowledgeGraph()
    
    def test_add_entity(self):
        """Test adding entities to graph"""
        entity = Entity(
            text="Python",
            type=EntityType.SKILL,
            confidence=0.95
        )
        self.graph.add_entity(entity)
        
        self.assertEqual(len(self.graph.entities), 1)
        self.assertEqual(self.graph.graph.number_of_nodes(), 1)
    
    def test_skill_similarity(self):
        """Test skill similarity calculation"""
        # Exact match should be 0.0
        distance = self.graph.get_skill_graph_distance("Python", "Python")
        self.assertEqual(distance, 0.0)
        
        # Alias match should be < 0.5
        distance = self.graph.get_skill_graph_distance("k8s", "kubernetes")
        self.assertLess(distance, 0.5)
        
        # Different skills should be >= 0.5
        distance = self.graph.get_skill_graph_distance("Python", "Docker")
        self.assertGreaterEqual(distance, 0.5)
    
    def test_candidate_profile(self):
        """Test candidate profile retrieval"""
        # Add test entities
        skill1 = Entity("Python", EntityType.SKILL, 0.95)
        skill2 = Entity("Docker", EntityType.SKILL, 0.90)
        
        self.graph.add_entity(skill1)
        self.graph.add_entity(skill2)
        
        # Should have 2 skills
        self.assertEqual(len(self.graph.entities), 2)


class TestResumeGraphBuilder(unittest.TestCase):
    """Test resume graph building"""
    
    def setUp(self):
        self.builder = ResumeGraphBuilder(use_llm=False)
    
    def test_build_from_text(self):
        """Test building graph from resume text"""
        resume = """
        Senior Python Developer at TechCorp (2021-2023)
        - Led Python development team
        - Built Docker containers
        - AWS infrastructure
        
        Skills: Python, Docker, Kubernetes, AWS
        """
        
        graph = self.builder.build_from_text(resume, "resume")
        
        # Should have extracted entities
        self.assertGreater(len(graph.entities), 0)
        self.assertGreater(len(graph.relationships), 0)
    
    def test_graph_summary(self):
        """Test graph summary statistics"""
        resume = "Python Docker Kubernetes AWS"
        graph = self.builder.build_from_text(resume, "resume")
        
        summary = graph.visualize_summary()
        
        self.assertIn('total_nodes', summary)
        self.assertIn('total_edges', summary)
        self.assertIn('entity_types', summary)
        self.assertGreater(summary['total_nodes'], 0)


class TestMatchingEngine(unittest.TestCase):
    """Test candidate-job matching functionality"""
    
    def setUp(self):
        self.engine = MatchingEngine(use_llm=False)
    
    def test_extract_job_requirements(self):
        """Test extraction of job requirements"""
        job_desc = """
        Senior Python Developer
        
        Requirements:
        - 5+ years Python experience
        - Docker and Kubernetes
        - AWS knowledge
        """
        
        reqs = self.engine.extract_job_requirements(job_desc)
        
        self.assertIn('required_skills', reqs)
        self.assertGreater(len(reqs['required_skills']), 0)
    
    def test_skill_match_score_exact(self):
        """Test exact skill match scoring"""
        required = ["Python", "Docker"]
        candidate = ["Python", "Docker", "Kubernetes"]
        
        score, matched, missing = self.engine.calculate_skill_match_score(required, candidate)
        
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(missing), 0)
        self.assertEqual(score, 1.0)
    
    def test_skill_match_score_partial(self):
        """Test partial skill match scoring"""
        required = ["Python", "Docker", "Kubernetes"]
        candidate = ["Python", "Kubernetes"]
        
        score, matched, missing = self.engine.calculate_skill_match_score(required, candidate)
        
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(missing), 1)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.5)
    
    def test_skill_match_score_none(self):
        """Test when candidate has no required skills"""
        required = ["Python", "Docker"]
        candidate = ["Java", "Kubernetes"]
        
        score, matched, missing = self.engine.calculate_skill_match_score(required, candidate)
        
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(missing), 2)
        self.assertEqual(score, 0.0)
    
    def test_match_result_structure(self):
        """Test MatchResult data structure"""
        result = MatchResult(
            candidate_name="John Doe",
            job_title="Senior Python Developer",
            match_score=0.85,
            matched_skills=["Python", "Docker"],
            missing_skills=["Kubernetes"],
            additional_skills=["AWS"],
            reasoning="Strong match with minor gaps"
        )
        
        self.assertEqual(result.candidate_name, "John Doe")
        self.assertEqual(result.match_score, 0.85)
        self.assertEqual(len(result.matched_skills), 2)
    
    def test_recommendation_logic(self):
        """Test recommendation generation"""
        # Strong match
        rec = self.engine._get_recommendation(0.92)
        self.assertIn("STRONG", rec)
        
        # Good match
        rec = self.engine._get_recommendation(0.80)
        self.assertIn("RECOMMEND", rec)
        
        # Weak match
        rec = self.engine._get_recommendation(0.40)
        self.assertIn("NOT RECOMMENDED", rec)


class TestIntegration(unittest.TestCase):
    """Integration tests with real files"""
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create test resumes
        resumes_dir = self.temp_path / "resumes"
        resumes_dir.mkdir()
        
        resume1 = resumes_dir / "john_doe.txt"
        resume1.write_text("""
        Senior Python Developer
        
        Experience:
        Senior Developer at TechCorp (2021-2023)
        - Python microservices
        - Docker and Kubernetes
        - AWS infrastructure
        
        Skills: Python, Docker, Kubernetes, AWS, PostgreSQL
        """)
        
        resume2 = resumes_dir / "jane_smith.txt"
        resume2.write_text("""
        Junior Java Developer
        
        Experience:
        Java Developer at StartupX (2022-2023)
        - Backend Java services
        - MySQL database
        
        Skills: Java, MySQL, JavaScript
        """)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_end_to_end_matching(self):
        """Test complete matching workflow"""
        engine = MatchingEngine(use_llm=False)
        
        # Load candidates
        resumes_dir = self.temp_path / "resumes"
        engine.load_candidates(str(resumes_dir))
        
        self.assertEqual(len(engine.candidate_graphs), 2)
        
        # Define job
        job_desc = """
        Senior Python Developer
        
        Requirements:
        - Python expertise
        - Docker and Kubernetes
        - AWS experience
        """
        
        # Find matches
        results = engine.find_matches(job_desc, top_k=2, job_title="Senior Python Developer")
        
        # Should return results (even if scores are 0)
        self.assertEqual(len(results), 2)
        
        # Verify result structure
        for result in results:
            self.assertIsInstance(result.candidate_name, str)
            self.assertIsInstance(result.match_score, float)
            self.assertIsInstance(result.matched_skills, list)
            self.assertIsInstance(result.missing_skills, list)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.engine = MatchingEngine(use_llm=False)
    
    def test_empty_job_description(self):
        """Test with empty job description"""
        reqs = self.engine.extract_job_requirements("")
        
        # Should return structure with empty lists
        self.assertIn('required_skills', reqs)
    
    def test_no_matches_found(self):
        """Test when no candidates match"""
        required = ["Cobol", "Fortran"]
        candidate = ["Python", "Java"]
        
        score, matched, missing = self.engine.calculate_skill_match_score(required, candidate)
        
        self.assertEqual(score, 0.0)
        self.assertEqual(len(matched), 0)
    
    def test_case_insensitivity(self):
        """Test that matching is case-insensitive"""
        required = ["python"]
        candidate = ["Python", "PYTHON"]
        
        score, matched, missing = self.engine.calculate_skill_match_score(required, candidate)
        
        # Should match despite case differences
        self.assertGreater(len(matched), 0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
