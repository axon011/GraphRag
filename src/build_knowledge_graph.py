"""
build_knowledge_graph.py

Knowledge Graph Construction Pipeline
Builds entity-relationship graphs from extracted entities for GraphRAG processing.

Author: Your Name
Date: 2025
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import networkx as nx
from pathlib import Path
import pypdf
from .extract_entities import Entity, EntityType, RegexEntityExtractor, LLMEntityExtractor
from .config import settings



class RelationType(str, Enum):
    """Supported relationship types in the knowledge graph"""
    HAS_SKILL = "HAS_SKILL"
    WORKED_AT = "WORKED_AT"
    CERTIFIED_IN = "CERTIFIED_IN"
    WORKED_FROM = "WORKED_FROM"
    PROGRESSED_TO = "PROGRESSED_TO"  # Skill progression
    SIMILAR_TO = "SIMILAR_TO"  # Skill similarity
    REQUIRES = "REQUIRES"  # Job requires skill


@dataclass
class Relationship:
    """Represents a relationship between two entities"""
    source: Entity
    target: Entity
    relation_type: RelationType
    confidence: float
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Relationship({self.source.text} --{self.relation_type.value}--> {self.target.text})"


class KnowledgeGraph:
    """Graph-based knowledge representation of resumes and job descriptions"""
    
    def __init__(self):
        """Initialize the knowledge graph"""
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        
        # Skill aliases and synonyms for semantic matching
        self.skill_aliases = {
            'k8s': 'kubernetes',
            'k8': 'kubernetes',
            'js': 'javascript',
            'ts': 'typescript',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'devops': 'devops',
            'ci/cd': 'continuous integration',
            'rest': 'rest api',
            'sql': 'sql',
            'nosql': 'nosql',
        }
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the graph
        
        Args:
            entity: Entity to add
        """
        entity_id = f"{entity.type.value}:{entity.text.lower()}"
        
        if entity_id not in self.entities:
            self.entities[entity_id] = entity
            self.graph.add_node(entity_id, **{
                'text': entity.text,
                'type': entity.type.value,
                'confidence': entity.confidence,
                'context': entity.context
            })
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the graph
        
        Args:
            relationship: Relationship to add
        """
        source_id = f"{relationship.source.type.value}:{relationship.source.text.lower()}"
        target_id = f"{relationship.target.type.value}:{relationship.target.text.lower()}"
        
        self.add_entity(relationship.source)
        self.add_entity(relationship.target)
        
        self.graph.add_edge(source_id, target_id, 
                           relation_type=relationship.relation_type.value,
                           confidence=relationship.confidence,
                           **relationship.metadata)
        
        self.relationships.append(relationship)
    
    def find_skill_synonyms(self, skill: str) -> List[str]:
        """
        Find synonym skills using predefined aliases
        
        Args:
            skill: Skill to find synonyms for
        
        Returns:
            List of synonymous skills
        """
        skill_lower = skill.lower()
        synonyms = [skill]
        
        # Check if skill is an alias
        if skill_lower in self.skill_aliases:
            base_skill = self.skill_aliases[skill_lower]
            synonyms.append(base_skill)
        
        # Check if any aliases point to this skill
        for alias, base in self.skill_aliases.items():
            if base == skill_lower:
                synonyms.append(alias)
        
        return list(set(synonyms))
    
    def get_skill_graph_distance(self, skill1: str, skill2: str) -> float:
        """
        Calculate semantic distance between two skills
        
        Args:
            skill1: First skill
            skill2: Second skill
        
        Returns:
            Distance score (0.0 = identical, 1.0 = very different)
        """
        # Normalize
        skill1_lower = skill1.lower()
        skill2_lower = skill2.lower()
        
        # Exact match
        if skill1_lower == skill2_lower:
            return 0.0
        
        # Alias match
        syn1 = self.find_skill_synonyms(skill1_lower)
        syn2 = self.find_skill_synonyms(skill2_lower)
        
        if any(s in syn2 for s in syn1):
            return 0.2  # Very similar
        
        # Similar prefix/suffix (e.g., Python, Python3, PyPy)
        if skill1_lower.startswith(skill2_lower) or skill2_lower.startswith(skill1_lower):
            return 0.4
        
        return 1.0  # Different
    
    def find_matching_candidates(self, job_skills: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find candidates that match job requirements
        
        Args:
            job_skills: Required skills for job
            top_k: Return top k candidates
        
        Returns:
            List of (candidate_id, match_score) tuples
        """
        candidate_scores = {}
        
        for job_skill in job_skills:
            job_skill_nodes = [
                node for node in self.graph.nodes()
                if self.graph.nodes[node].get('type') == 'SKILL' and 
                   self.get_skill_graph_distance(job_skill, self.graph.nodes[node].get('text', '')) < 0.5
            ]
            
            # Find candidates with these skills
            for skill_node in job_skill_nodes:
                # Get incoming edges (candidates that have this skill)
                sources = list(self.graph.predecessors(skill_node))
                for source in sources:
                    if 'CANDIDATE' in source or 'RESUME' in source:
                        if source not in candidate_scores:
                            candidate_scores[source] = 0.0
                        candidate_scores[source] += 1.0
        
        # Score candidates by match percentage
        total_required = len(job_skills)
        scored_candidates = [
            (cand, score / total_required if total_required > 0 else 0.0)
            for cand, score in candidate_scores.items()
        ]
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[:top_k]
    
    def get_candidate_profile(self, candidate_id: str) -> Dict:
        """
        Get complete profile of a candidate
        
        Args:
            candidate_id: Candidate identifier
        
        Returns:
            Dictionary with candidate information
        """
        profile = {
            'id': candidate_id,
            'skills': [],
            'roles': [],
            'companies': [],
            'certifications': [],
            'timeline': []
        }
        
        # Find all nodes connected to this candidate
        if candidate_id in self.graph:
            for neighbor in self.graph.successors(candidate_id):
                node_type = self.graph.nodes[neighbor].get('type')
                node_text = self.graph.nodes[neighbor].get('text')
                
                if node_type == 'SKILL':
                    profile['skills'].append(node_text)
                elif node_type == 'ROLE':
                    profile['roles'].append(node_text)
                elif node_type == 'COMPANY':
                    profile['companies'].append(node_text)
                elif node_type == 'CERTIFICATION':
                    profile['certifications'].append(node_text)
                elif node_type == 'DURATION':
                    profile['timeline'].append(node_text)
        
        return profile
    
    def to_json(self) -> str:
        """Export graph to JSON format"""
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            graph_data['nodes'].append({
                'id': node_id,
                **node_data
            })
        
        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            graph_data['edges'].append({
                'source': source,
                'target': target,
                **edge_data
            })
        
        return json.dumps(graph_data, indent=2)
    
    def save(self, filepath: str) -> None:
        """Save graph to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    def visualize_summary(self) -> Dict:
        """Get summary statistics of the graph"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_types': {
                entity_type.value: sum(
                    1 for node in self.graph.nodes()
                    if self.graph.nodes[node].get('type') == entity_type.value
                )
                for entity_type in EntityType
            },
            'relationship_types': {
                rel_type.value: sum(
                    1 for _, _, data in self.graph.edges(data=True)
                    if data.get('relation_type') == rel_type.value
                )
                for rel_type in RelationType
            }
        }


class ResumeGraphBuilder:
    """Builder for constructing knowledge graphs from resume documents"""
    
    def __init__(self, use_llm: Optional[bool] = None):
        """
        Initialize the graph builder
        
        Args:
            use_llm: Use LLM for entity extraction. If None, checks settings.
        """
        if use_llm is None:
            # Check settings, default to False if not specified
            use_llm = settings.get('use_llm', False)
            
        print(f"DEBUG: ResumeGraphBuilder initialized with use_llm={use_llm}")
        self.extractor = LLMEntityExtractor() if use_llm else RegexEntityExtractor()
        self.graph = KnowledgeGraph()
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """
        Extract text from a PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text

    def build_from_text(self, text: str, document_type: str = "resume") -> KnowledgeGraph:
        """
        Build knowledge graph from text document
        
        Args:
            text: Document text (resume or job description)
            document_type: Type of document ('resume' or 'job_description')
        
        Returns:
            Populated KnowledgeGraph
        """
        # Extract entities
        entities = self.extractor.extract(text)
        
        # Create document node
        doc_id = f"{document_type.upper()}:doc_{len(self.graph.graph.nodes())}"
        self.graph.graph.add_node(doc_id, text='document', type=document_type)
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)
        
        # Add entities and relationships
        for entity_type, entity_list in entities_by_type.items():
            for entity in entity_list:
                self.graph.add_entity(entity)
                
                # Connect to document
                entity_id = f"{entity.type.value}:{entity.text.lower()}"
                relation_type = RelationType.REQUIRES if "job" in document_type.lower() else RelationType.HAS_SKILL
                rel = Relationship(
                    source=Entity(text=doc_id.split(':')[1], type=EntityType.ROLE, confidence=1.0),
                    target=entity,
                    relation_type=relation_type,
                    confidence=entity.confidence
                )
                self.graph.add_relationship(rel)
        
        # Add semantic relationships between skills
        if EntityType.SKILL in entities_by_type:
            skills = entities_by_type[EntityType.SKILL]
            for i, skill1 in enumerate(skills):
                for skill2 in skills[i+1:]:
                    distance = self.graph.get_skill_graph_distance(skill1.text, skill2.text)
                    if distance < 0.5:  # Skills are similar
                        rel = Relationship(
                            source=skill1,
                            target=skill2,
                            relation_type=RelationType.SIMILAR_TO,
                            confidence=1.0 - distance
                        )
                        self.graph.add_relationship(rel)
        
        return self.graph
    
    def build_from_directory(self, directory: str) -> KnowledgeGraph:
        """
        Build knowledge graph from all documents in a directory
        
        Args:
            directory: Directory containing resume/job description files
        
        Returns:
            Unified KnowledgeGraph
        """
        path = Path(directory)
        
        # Process Resumes (txt and pdf)
        for resume_file in path.glob("resumes/*.txt"):
            with open(resume_file, 'r') as f:
                self.build_from_text(f.read(), document_type="resume")
                
        for resume_file in path.glob("resumes/*.pdf"):
            text = self._extract_text_from_pdf(resume_file)
            if text.strip():
                self.build_from_text(text, document_type="resume")
        
        # Process Cover Letters (txt and pdf) - treating as resume/profile info for now
        for cover_file in path.glob("cover_letters/*.txt"):
            with open(cover_file, 'r') as f:
                self.build_from_text(f.read(), document_type="resume")

        for cover_file in path.glob("cover_letters/*.pdf"):
            text = self._extract_text_from_pdf(cover_file)
            if text.strip():
                self.build_from_text(text, document_type="resume")
        
        # Process Job Descriptions (txt and pdf)
        for job_file in path.glob("job_descriptions/*.txt"):
            with open(job_file, 'r') as f:
                self.build_from_text(f.read(), document_type="job_description")

        for job_file in path.glob("job_descriptions/*.pdf"):
            text = self._extract_text_from_pdf(job_file)
            if text.strip():
                self.build_from_text(text, document_type="job_description")
        
        return self.graph


if __name__ == "__main__":
    # Example usage
    print("=== Knowledge Graph Building Example ===\n")
    
    # Check if we should use LLM (you can make this configurable)
    # use_llm = False 
    # Now uses settings by default if not specified
    
    builder = ResumeGraphBuilder()
    
    # Build from Data directory
    data_dir = Path("Data")
    if data_dir.exists():
        print(f"Building graph from {data_dir}...")
        graph = builder.build_from_directory(str(data_dir))
        
        print("\nGraph Summary:")
        summary = graph.visualize_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print(f"\nTotal entities: {len(graph.entities)}")
        print(f"Total relationships: {len(graph.relationships)}")
        
        # Save the graph
        output_file = "output/knowledge_graph.json"
        graph.save(output_file)
        print(f"\nGraph saved to {output_file}")
    else:
        print("Data directory not found.")

