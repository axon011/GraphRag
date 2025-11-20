# __init__.py for src package

"""
Resume GraphRAG - AI-Powered Resume-Job Matching System

This package implements a Graph RAG system for intelligent candidate-job matching.
It combines entity extraction, knowledge graph construction, and semantic matching.

Key Modules:
- extract_entities: Extract skills, roles, companies, durations from documents
- build_knowledge_graph: Construct entity-relationship graphs
- query_engine: Match candidates to jobs using graph reasoning
"""

from .extract_entities import Entity, EntityType, RegexEntityExtractor, LLMEntityExtractor, extract_resume_entities
from .build_knowledge_graph import KnowledgeGraph, ResumeGraphBuilder, RelationType, Relationship
from .query_engine import MatchingEngine, MatchResult

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    'Entity',
    'EntityType',
    'RegexEntityExtractor',
    'LLMEntityExtractor',
    'extract_resume_entities',
    'KnowledgeGraph',
    'ResumeGraphBuilder',
    'RelationType',
    'Relationship',
    'MatchingEngine',
    'MatchResult',
]
