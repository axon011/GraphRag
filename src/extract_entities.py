"""
extract_entities.py

Entity Extraction Pipeline
Extracts SKILL, ROLE, COMPANY, DURATION, and CERTIFICATION entities from resumes and job descriptions.

Author: Your Name
Date: 2025
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field
import re
from abc import ABC, abstractmethod
from .config import settings


class EntityType(str, Enum):
    """Supported entity types for extraction"""
    SKILL = "SKILL"
    ROLE = "ROLE"
    COMPANY = "COMPANY"
    DURATION = "DURATION"
    CERTIFICATION = "CERTIFICATION"


@dataclass
class Entity:
    """Represents an extracted entity with metadata"""
    text: str
    type: EntityType
    confidence: float
    context: Optional[str] = None
    position: Optional[Tuple[int, int]] = None  # (start, end) character positions
    
    def __repr__(self) -> str:
        return f"Entity(text='{self.text}', type={self.type.value}, confidence={self.confidence:.2f})"


class EntityExtractor(ABC):
    """Abstract base class for entity extractors"""
    
    @abstractmethod
    def extract(self, text: str, entity_types: Optional[List[EntityType]] = None) -> List[Entity]:
        """Extract entities from text"""
        pass


class RegexEntityExtractor(EntityExtractor):
    """Rule-based entity extraction using regex patterns"""
    
    # Predefined patterns for common skills and roles
    SKILL_PATTERNS = {
        'programming_languages': r'\b(python|java|javascript|golang|c\+\+|rust|scala|kotlin|typescript|php|ruby)\b',
        'cloud_platforms': r'\b(aws|azure|gcp|google cloud|kubernetes|k8s|docker|terraform)\b',
        'databases': r'\b(postgresql|mysql|mongodb|redis|elasticsearch|dynamodb|cassandra)\b',
        'ml_frameworks': r'\b(tensorflow|pytorch|scikit-learn|huggingface|transformers|pandas|numpy)\b',
        'devops_tools': r'\b(jenkins|gitlab-ci|github actions|circleci|datadog|prometheus|grafana)\b',
    }
    
    ROLE_PATTERNS = {
        'seniority': r'\b(junior|mid|senior|lead|principal|staff|architect)\b',
        'position': r'\b(engineer|developer|scientist|analyst|architect|manager|lead)\b',
    }
    
    DURATION_PATTERN = r'(\d{4})\s*[-–]\s*(\d{4}|present|current)'
    CERTIFICATION_PATTERN = r'\b(aws certified|gcp certified|cka|ckad|pmp|scrum master)\b'
    
    def __init__(self, use_llm: bool = False, llm_model: Optional[str] = None):
        """
        Initialize extractor
        
        Args:
            use_llm: Whether to use LLM-based extraction (more accurate but slower)
            llm_model: LLM model name if using LLM extraction
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
    
    def extract(
        self, 
        text: str, 
        entity_types: Optional[List[EntityType]] = None
    ) -> List[Entity]:
        """
        Extract entities from text using regex patterns
        
        Args:
            text: Input text to extract entities from
            entity_types: Filter to specific entity types (None = all types)
        
        Returns:
            List of extracted entities
        """
        if entity_types is None:
            entity_types = list(EntityType)
        
        entities = []
        text_lower = text.lower()
        
        # Extract SKILL entities
        if EntityType.SKILL in entity_types:
            for category, pattern in self.SKILL_PATTERNS.items():
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entities.append(Entity(
                        text=match.group(0).title(),
                        type=EntityType.SKILL,
                        confidence=0.85,
                        context=category,
                        position=match.span()
                    ))
        
        # Extract ROLE entities
        if EntityType.ROLE in entity_types:
            for category, pattern in self.ROLE_PATTERNS.items():
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entities.append(Entity(
                        text=match.group(0).title(),
                        type=EntityType.ROLE,
                        confidence=0.80,
                        context=category,
                        position=match.span()
                    ))
        
        # Extract DURATION entities
        if EntityType.DURATION in entity_types:
            matches = re.finditer(self.DURATION_PATTERN, text_lower, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(0),
                    type=EntityType.DURATION,
                    confidence=0.90,
                    position=match.span()
                ))
        
        # Extract CERTIFICATION entities
        if EntityType.CERTIFICATION in entity_types:
            matches = re.finditer(self.CERTIFICATION_PATTERN, text_lower, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(0),
                    type=EntityType.CERTIFICATION,
                    confidence=0.88,
                    position=match.span()
                ))
        
        # Deduplicate entities
        return self._deduplicate_entities(entities)
    
    @staticmethod
    def _deduplicate_entities(entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence"""
        seen = {}
        for entity in sorted(entities, key=lambda e: e.confidence, reverse=True):
            key = (entity.text.lower(), entity.type)
            if key not in seen:
                seen[key] = entity
        return list(seen.values())


class LLMEntityExtractor(EntityExtractor):
    """LLM-based entity extraction using language models"""
    
    def __init__(self, llm_model: Optional[str] = None):
        """
        Initialize LLM-based extractor
        
        Args:
            llm_model: Model name (e.g., 'mistral', 'llama2'). If None, uses settings.
        """
        self.llm_model = llm_model or settings.llm_config.get('model', 'mistral')
        self.api_base = settings.llm_config.get('api_base', 'http://localhost:11434/v1')
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM connection"""
        try:
            from langchain_community.llms import Ollama
            from langchain_community.llms import Ollama
            # LangChain's Ollama class expects base_url without /v1
            base_url = self.api_base.replace("/v1", "")
            self.llm = Ollama(model=self.llm_model, base_url=base_url, temperature=0.3)
        except ImportError:
            print("Warning: LangChain not available. LLM extraction will fall back to regex.")
            self.llm = None
    
    def extract(
        self, 
        text: str, 
        entity_types: Optional[List[EntityType]] = None
    ) -> List[Entity]:
        """
        Extract entities using LLM
        
        Args:
            text: Input text to extract entities from
            entity_types: Filter to specific entity types
        
        Returns:
            List of extracted entities with higher accuracy
        """
        if entity_types is None:
            entity_types = list(EntityType)
        
        entity_type_str = ", ".join([et.value for et in entity_types])
        
        prompt = f"""Extract the following entity types from the text: {entity_type_str}

For each entity found, provide:
1. Text: the exact text from the document
2. Type: the entity type
3. Confidence: a score from 0.0 to 1.0

Text to analyze:
{text}

Return as JSON array with objects like {{"text": "...", "type": "...", "confidence": 0.0}}
"""
        
        if self.llm is None:
            print("LLM not available. Using regex extraction.")
            regex_extractor = RegexEntityExtractor()
            return regex_extractor.extract(text, entity_types)
            
        try:
            print(f"DEBUG: Invoking LLM with text length: {len(text)}")
            response = self.llm.invoke(prompt)
            print(f"DEBUG: LLM Response: {response[:200]}...")
            # Parse JSON response and create Entity objects
            import json
            entities_data = json.loads(response)
            entities = [
                Entity(
                    text=e['text'],
                    type=EntityType[e['type']],
                    confidence=float(e.get('confidence', 0.80))
                )
                for e in entities_data
            ]
            return entities
        except Exception as e:
            print(f"LLM extraction error: {e}. Falling back to regex extraction.")
            # Fallback to regex if LLM fails
            regex_extractor = RegexEntityExtractor()
            return regex_extractor.extract(text, entity_types)


def extract_resume_entities(resume_path: str, use_llm: bool = False) -> Dict[EntityType, List[Entity]]:
    """
    Convenience function to extract entities from a resume file
    
    Args:
        resume_path: Path to resume file
        use_llm: Use LLM extraction for higher accuracy
    
    Returns:
        Dictionary mapping entity types to extracted entities
    """
    with open(resume_path, 'r') as f:
        text = f.read()
    
    extractor = LLMEntityExtractor() if use_llm else RegexEntityExtractor()
    entities = extractor.extract(text)
    
    # Group entities by type
    grouped = {}
    for entity_type in EntityType:
        grouped[entity_type] = [e for e in entities if e.type == entity_type]
    
    return grouped


if __name__ == "__main__":
    # Example usage
    sample_resume = """
    John Doe
    john@example.com | Senior Python Developer
    
    Experience:
    Senior Developer at TechCorp (2021-2023)
    - Led Python development team of 5 engineers
    - Built microservices using Docker and Kubernetes
    - Implemented machine learning pipeline with PyTorch and TensorFlow
    - AWS certified architect
    
    Junior Developer at StartupX (2019-2021)
    - Developed REST APIs in Python
    - Worked with PostgreSQL and MongoDB
    - Automated CI/CD using GitHub Actions
    
    Skills: Python, JavaScript, AWS, Docker, Kubernetes, PyTorch, TensorFlow, PostgreSQL
    Certifications: AWS Certified Solutions Architect, CKA (Kubernetes)
    """
    
    print("=== Entity Extraction Example ===\n")
    
    # Regex-based extraction
    print("1. Regex-Based Extraction:")
    regex_extractor = RegexEntityExtractor()
    entities = regex_extractor.extract(sample_resume)
    for entity in sorted(entities, key=lambda e: (e.type.value, e.text)):
        print(f"  {entity}")
    
    print(f"\nTotal entities extracted: {len(entities)}")
