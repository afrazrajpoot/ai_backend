import json
import os
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Annotated
import hashlib
import asyncio
import operator
import re
import random
import asyncpg

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

from utils.logger import logger
from config import settings
from utils.models import IndividualEmployeeReport
from utils.analysis_utils import MAPPING_FACTORS


class LLMPromptLogger:
    """Logs LLM prompts for debugging purposes"""
    
    @staticmethod
    def log_llm_prompt(prompt_type: str, prompt_content: str):
        """Log LLM prompt with structured details"""
        log_entry = {
            "timestamp": "timestamp_placeholder",
            "prompt_type": prompt_type,
            "prompt_content": prompt_content[:2000]  # Limit content length
        }
        
        # Log to console for debugging
        # Log to console for debugging
        logger.info(f"[LLM PROMPT] {prompt_type}:\n{prompt_content}")
        
        # Also save to debug file
        debug_file = Path(__file__).parent.parent / "llm_prompt_logs.json"
        try:
            if debug_file.exists():
                with open(debug_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            # Keep only last 50 logs
            if len(logs) > 50:
                logs = logs[-50:]
            
            with open(debug_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to write LLM prompt log: {e}")


class RAGQueryEngine:
    """Intelligent RAG query engine for retrieving specific information from knowledge base"""
    
    @staticmethod
    def build_scoring_guide_queries(basic_results: List[Dict]) -> List[Dict]:
        """Build queries to retrieve scoring guides for each assessment part"""
        queries = []
        
        for part in basic_results:
            part_name = part.get('part', '').lower()
            majority_options = part.get('majorityOptions', [])
            
            if not part_name or not majority_options:
                continue
            
            # Map part names to document section names
            part_mapping = {
                'self-awareness audit': 'Self-Awareness Audit',
                'talent audit': 'Talent Audit', 
                'passion audit': 'Passion Audit',
                'genius factor mapping': 'Genius Factor Identification',
                'genius factor identification': 'Genius Factor Identification',
                'part i: self-awareness audit questions': 'Self-Awareness Audit',
                'part ii: talent audit questions': 'Talent Audit',
                'part iii: passion audit questions': 'Passion Audit',
                'part iv: genius factor mapping assessment': 'Genius Factor Identification'
            }
            
            doc_part_name = part_mapping.get(part_name, part_name.title())
            
            # Build specific query for this part's scoring guide
            for option in majority_options[:2]:  # Take up to 2 majority options
                letter_to_factor = {
                    'A': 'Tech Genius',
                    'B': 'Social Genius',
                    'C': 'Visual Genius',
                    'D': 'Word Genius',
                    'E': 'Athletic Genius',
                    'F': 'Number Genius',
                    'G': 'Eco Genius',
                    'H': 'Word Genius',
                    'I': 'Spiritual Genius'
                }
                
                factor_name = letter_to_factor.get(option, f"Genius {option}")
                
                queries.append({
                    "query": f'"{doc_part_name}" Scoring Guide "{option} responses" "{factor_name}"',
                    "source_filter": "(68 Questions) Genius Factor Assessment for Fortune 1000 HR Departments.pdf",
                    "description": f"Scoring guide for {doc_part_name} with {option} ({factor_name}) responses",
                    "priority": "high",
                    "query_type": "scoring_guide",
                    "part_name": part_name,
                    "option": option
                })
        
        return queries
    
    @staticmethod
    def build_genius_factor_queries(primary_genius: str, secondary_genius: str = None) -> List[Dict]:
        """Build queries to retrieve details about genius factors"""
        queries = []
        
        if primary_genius and primary_genius != "Unknown Genius":
            # Extract base name without "Genius" for better matching
            primary_base = primary_genius.replace(" Genius", "").strip()
            
            # Query for primary genius factor description
            queries.append({
                "query": f'"{primary_base} Genius" description characteristics career paths natural abilities',
                "source_filter": "Genius Factor Framework Analysis.pdf",
                "description": f"Detailed description of {primary_genius}",
                "priority": "high",
                "query_type": "genius_factor_details",
                "genius_type": "primary"
            })
            
            # Query for primary genius factor in assessment guide
            queries.append({
                "query": f'"{primary_base} Genius" responses indicate strong',
                "source_filter": "(68 Questions) Genius Factor Assessment for Fortune 1000 HR Departments.pdf",
                "description": f"Assessment insights for {primary_genius}",
                "priority": "medium",
                "query_type": "assessment_insights",
                "genius_type": "primary"
            })
        
        if secondary_genius and secondary_genius not in ["None", "None Identified", "Unknown Genius"]:
            secondary_base = secondary_genius.replace(" Genius", "").strip()
            
            queries.append({
                "query": f'"{secondary_base} Genius" description characteristics',
                "source_filter": "Genius Factor Framework Analysis.pdf",
                "description": f"Detailed description of {secondary_genius}",
                "priority": "medium",
                "query_type": "genius_factor_details",
                "genius_type": "secondary"
            })
        
        return queries
    
    @staticmethod
    def build_industry_mapping_queries(primary_genius: str, secondary_genius: str = None) -> List[Dict]:
        """Build queries to retrieve industry mapping information"""
        queries = []
        
        if primary_genius and primary_genius != "Unknown Genius":
            # Extract the base genius name
            genius_base = primary_genius.replace(" Genius", "").strip()
            
            queries.append({
                "query": f'"{genius_base} Genius" industry alignments Fortune 1000',
                "source_filter": "Genius Factor to Fortune 1000 Industry Mapping.pdf",
                "description": f"Industry mapping for {primary_genius}",
                "priority": "high",
                "query_type": "industry_mapping",
                "genius_type": "primary",
                "target": "primary_industries"
            })
            
            queries.append({
                "query": f'"{genius_base}" primary industries secondary industries',
                "source_filter": "Genius Factor to Fortune 1000 Industry Mapping.pdf",
                "description": f"Career pathways for {primary_genius}",
                "priority": "high",
                "query_type": "career_pathways",
                "genius_type": "primary"
            })
        
        if secondary_genius and secondary_genius not in ["None", "None Identified", "Unknown Genius"]:
            genius_base = secondary_genius.replace(" Genius", "").strip()
            
            queries.append({
                "query": f'"{genius_base}" secondary industries',
                "source_filter": "Genius Factor to Fortune 1000 Industry Mapping.pdf",
                "description": f"Secondary industries for {secondary_genius}",
                "priority": "medium",
                "query_type": "industry_mapping",
                "genius_type": "secondary",
                "target": "secondary_industries"
            })
        
        # Add hybrid combination query if both factors exist
        if primary_genius and secondary_genius and secondary_genius not in ["None", "None Identified"]:
            prim_base = primary_genius.replace(" Genius", "").strip()
            sec_base = secondary_genius.replace(" Genius", "").strip()
            
            queries.append({
                "query": f'{prim_base} {sec_base} hybrid combination career',
                "source_filter": "Genius Factor to Fortune 1000 Industry Mapping.pdf",
                "description": f"Hybrid combination of {primary_genius} and {secondary_genius}",
                "priority": "medium",
                "query_type": "hybrid_combinations"
            })
        
        return queries
    
    @staticmethod
    def build_retention_mobility_queries() -> List[Dict]:
        """Build queries to retrieve retention and mobility research"""
        queries = [
            {
                "query": "retention internal mobility research findings",
                "source_filter": "retention & internal mobility research_findings.pdf",
                "description": "Retention and internal mobility research",
                "priority": "high",
                "query_type": "retention_research"
            },
            {
                "query": "internal mobility programs career mobility",
                "source_filter": "retention & internal mobility research_findings.pdf",
                "description": "Internal mobility program details",
                "priority": "medium",
                "query_type": "mobility_programs"
            },
            {
                "query": "Fortune 1000 employee retention strategies",
                "source_filter": "retention & internal mobility research_findings.pdf",
                "description": "Fortune 1000 retention strategies",
                "priority": "medium",
                "query_type": "retention_strategies"
            }
        ]
        
        return queries
    
    @staticmethod
    def extract_genius_names_from_results(basic_results: List[Dict], deep_results: Dict = None) -> tuple:
        """Extract primary and secondary genius names from analysis results"""
        primary_name = None
        secondary_name = None
        
        # First try to get from deep_results
        if deep_results:
            prim = deep_results.get("primary_genius", [])
            sec = deep_results.get("secondary_genius", [])
            if prim and isinstance(prim, list) and len(prim) > 0:
                primary_name = prim[0].get("name")
            if sec and isinstance(sec, list) and len(sec) > 0:
                secondary_name = sec[0].get("name")
        
        # Fallback to basic results
        if not primary_name:
            all_counts = {}
            for part in basic_results:
                for opt, cnt in (part.get("optionCounts") or {}).items():
                    all_counts[opt] = all_counts.get(opt, 0) + cnt
            
            if all_counts:
                sorted_all = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
                if sorted_all:
                    letter = sorted_all[0][0]
                    primary_name = MAPPING_FACTORS.get(letter, {}).get("name", f"{letter} Genius")
                    
                    if len(sorted_all) > 1:
                        letter2 = sorted_all[1][0]
                        secondary_name = MAPPING_FACTORS.get(letter2, {}).get("name", f"{letter2} Genius")
        
        return primary_name, secondary_name


class AIService:
    _prompts = None
    _vector_store = None
    _embeddings = None

    @classmethod
    def _get_embeddings(cls):
        if cls._embeddings is None:
            cls._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.OPENAI_API_KEY,
                chunk_size=1000,
                max_retries=3,
                request_timeout=60
            )
        return cls._embeddings

    @classmethod
    def _load_prompts(cls):
        if cls._prompts is None:
            try:
                prompts_path = Path(__file__).parent.parent / "utils" / "prompts.json"
                with open(prompts_path) as f:
                    cls._prompts = json.load(f)
            except Exception as e:
                logger.error(f"Error loading prompts: {str(e)}")
                raise
        return cls._prompts

    @classmethod
    def _get_pdf_files_hash(cls, pdf_files: List[str]) -> str:
        file_info = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                stat = os.stat(pdf_file)
                file_info.append(f"{pdf_file}:{stat.st_mtime}:{stat.st_size}")
        combined = "|".join(sorted(file_info))
        return hashlib.md5(combined.encode()).hexdigest()

    @classmethod
    def _check_index_validity(cls, faiss_path: str, pdf_files: List[str]) -> bool:
        hash_file = f"{faiss_path}.hash"
        if not os.path.exists(faiss_path) or not os.path.exists(hash_file):
            return False
        try:
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            current_hash = cls._get_pdf_files_hash(pdf_files)
            return stored_hash == current_hash
        except Exception:
            return False

    @classmethod
    def initialize_vector_store(cls, force_rebuild: bool = False):
        if cls._vector_store is not None and not force_rebuild:
            return cls._vector_store

        faiss_path = "genius_factor_index"
        embeddings = cls._get_embeddings()
        
        pdf_files = [
            str(Path(__file__).parent.parent / "(68 Questions) Genius Factor Assessment for Fortune 1000 HR Departments.pdf"),
            str(Path(__file__).parent.parent / "Genius Factor Framework Analysis.pdf"),
            str(Path(__file__).parent.parent / "Genius Factor to Fortune 1000 Industry Mapping.pdf"),
            str(Path(__file__).parent.parent / "retention & internal mobility research_findings.pdf"),
        ]

        if not force_rebuild and cls._check_index_validity(faiss_path, pdf_files):
            try:
                cls._vector_store = FAISS.load_local(
                    faiss_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                return cls._vector_store
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Rebuilding...")

        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                logger.warning(f"PDF file not found: {pdf_file}")
                continue
            try:
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                for chunk in chunks:
                    chunk.metadata['source_file'] = os.path.basename(pdf_file)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
                continue

        if not all_chunks:
            raise ValueError("No documents were loaded from PDF files")

        try:
            cls._vector_store = FAISS.from_documents(all_chunks, embeddings)
            cls._vector_store.save_local(faiss_path)
            current_hash = cls._get_pdf_files_hash(pdf_files)
            with open(f"{faiss_path}.hash", 'w') as f:
                f.write(current_hash)
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
        return cls._vector_store

    @classmethod
    async def retrieve_rag_data(cls, query_info: Dict) -> List[Dict]:
        """Retrieve documents for a specific query"""
        if cls._vector_store is None:
            cls.initialize_vector_store()
        
        try:
            source_filter = query_info.get("source_filter")
            
            # Create retriever
            retriever = cls._vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 8 if query_info.get("priority") == "high" else 5
                }
            )
            
            # Use ainvoke instead of aget_relevant_documents
            docs = await retriever.ainvoke(query_info["query"])
            
            # Filter by source if specified
            if source_filter:
                docs = [doc for doc in docs if doc.metadata.get('source_file', '') == source_filter]
            
            if not docs and source_filter:
                # Try broader search if no results
                retriever = cls._vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                docs = await retriever.ainvoke(query_info["query"])
            
            results = []
            for i, doc in enumerate(docs):
                result = {
                    "content": doc.page_content,
                    "source": doc.metadata.get('source_file', 'unknown'),
                    "page": doc.metadata.get('page', 'unknown'),
                    "retrieval_reason": query_info["description"],
                    "query_used": query_info["query"],
                    "query_type": query_info.get("query_type"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "retrieval_rank": i + 1
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to retrieve for query '{query_info.get('query')}': {e}")
            return []

    @classmethod
    async def analyze_majority_answers(cls, basic_results: List[Dict[str, Any]], deep_results: Dict[str, Any] = None) -> str:
        """
        Enhanced RAG analysis retrieving:
        1. Scoring guides for dominant parts
        2. Genius factor details (primary & secondary)
        3. Industry mapping information
        4. Retention and mobility research
        """
        # Initialize vector store
        cls.initialize_vector_store()
        
        # Extract genius names
        query_engine = RAGQueryEngine()
        primary_name, secondary_name = query_engine.extract_genius_names_from_results(basic_results, deep_results)
        
        # logger.info(f"Retrieving RAG data for: Primary={primary_name}, Secondary={secondary_name}")
        
        # Build all queries
        all_queries = []
        
        # 1. Scoring guide queries
        scoring_queries = query_engine.build_scoring_guide_queries(basic_results)
        all_queries.extend(scoring_queries)
        
        # 2. Genius factor detail queries
        if primary_name:
            genius_queries = query_engine.build_genius_factor_queries(primary_name, secondary_name)
            all_queries.extend(genius_queries)
        
        # 3. Industry mapping queries
        if primary_name:
            industry_queries = query_engine.build_industry_mapping_queries(primary_name, secondary_name)
            all_queries.extend(industry_queries)
        
        # 4. Retention and mobility queries
        retention_queries = query_engine.build_retention_mobility_queries()
        all_queries.extend(retention_queries)
        
        # Execute all queries in parallel
        tasks = [cls.retrieve_rag_data(query) for query in all_queries]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_docs = []
        
        for result in all_results:
            if isinstance(result, Exception):
                continue
            if result:
                all_docs.extend(result)
        
        # Deduplicate documents
        seen = set()
        unique_docs = []
        for doc in all_docs:
            # Create unique key based on content and source
            content_hash = hashlib.md5(doc["content"].encode()).hexdigest()
            key = f"{doc['source']}_{content_hash}"
            
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        # If no documents retrieved, try a fallback search
        if not unique_docs:
            # Try a simple search for any relevant content
            fallback_queries = [
                "Genius Factor assessment scoring",
                "Tech Genius characteristics",
                "employee retention research"
            ]
            
            fallback_tasks = []
            for query in fallback_queries:
                fallback_tasks.append(cls.retrieve_rag_data({
                    "query": query,
                    "source_filter": None,  # Search all sources
                    "description": f"Fallback search: {query}",
                    "priority": "low",
                    "query_type": "fallback"
                }))
            
            fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)
            
            for result in fallback_results:
                if not isinstance(result, Exception) and result:
                    all_docs.extend(result)
            
            # Re-deduplicate
            seen = set()
            unique_docs = []
            for doc in all_docs:
                content_hash = hashlib.md5(doc["content"].encode()).hexdigest()
                key = f"{doc['source']}_{content_hash}"
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(doc)
        
        # Format the comprehensive analysis output
        output_parts = []
        output_parts.append("=" * 80)
        output_parts.append("COMPREHENSIVE GENIUS FACTOR ASSESSMENT ANALYSIS")
        output_parts.append("=" * 80)
        
        # Section 1: Genius Factor Identification
        output_parts.append("\n1. GENIUS FACTOR IDENTIFICATION:")
        output_parts.append(f"   Primary Genius Factor: {primary_name or 'Not identified'}")
        output_parts.append(f"   Secondary Genius Factor: {secondary_name or 'Not identified'}")
        
        if deep_results:
            output_parts.append(f"   Hybrid Classification: {deep_results.get('hybrid_classification', 'None')}")
            output_parts.append(f"   Confidence Level: {deep_results.get('confidence_level', 'Unknown')}")
            output_parts.append(f"   Talent-Passion Alignment: {deep_results.get('talent_passion_alignment_label', 'Unknown')}")
        
        # Section 2: Assessment Response Summary
        output_parts.append("\n2. ASSESSMENT RESPONSE SUMMARY:")
        for i, part in enumerate(basic_results, 1):
            part_name = part.get('part', f'Part {i}')
            majority = part.get('majorityOptions', [])
            max_count = part.get('maxCount', 0)
            
            output_parts.append(f"   {i}. {part_name}")
            if majority:
                output_parts.append(f"      Dominant Pattern: {', '.join(majority)}")
                output_parts.append(f"      Strength: {max_count} responses")
            
            if part.get('optionCounts'):
                counts = [f"{k}: {v}" for k, v in part['optionCounts'].items()]
                output_parts.append(f"      Distribution: {', '.join(counts)}")
        
        # Section 3: Retrieved Knowledge Base Information
        output_parts.append("\n3. RETRIEVED KNOWLEDGE BASE INFORMATION:")
        output_parts.append(f"   Total Relevant Documents: {len(unique_docs)}")
        
        if unique_docs:
            # Organize documents by category
            scoring_guides = []
            genius_details = []
            industry_mapping = []
            retention_research = []
            
            for doc in unique_docs:
                query_type = doc.get('query_type', '')
                if 'scoring_guide' in query_type:
                    scoring_guides.append(doc)
                elif 'genius_factor' in query_type or 'assessment_insights' in query_type:
                    genius_details.append(doc)
                elif 'industry' in query_type or 'career_pathways' in query_type or 'hybrid' in query_type:
                    industry_mapping.append(doc)
                elif 'retention' in query_type or 'mobility' in query_type:
                    retention_research.append(doc)
            
            # 3.1 Scoring Guides
            if scoring_guides:
                output_parts.append(f"\n   3.1 SCORING GUIDES ({len(scoring_guides)} documents):")
                for i, doc in enumerate(scoring_guides, 1):
                    content = doc["content"].strip()
                    # Clean and summarize
                    content_lines = content.split('\n')
                    summary = ' '.join(content_lines[:3])[:200] + "..." if len(content) > 200 else content
                    
                    output_parts.append(f"      [{i}] {doc['retrieval_reason']}")
                    output_parts.append(f"         Source: {doc['source']} (Page {doc['page']})")
                    output_parts.append(f"         Content: {summary}")
            
            # 3.2 Genius Factor Details
            if genius_details:
                output_parts.append(f"\n   3.2 GENIUS FACTOR DETAILS ({len(genius_details)} documents):")
                for i, doc in enumerate(genius_details, 1):
                    content = doc["content"].strip()
                    content_lines = content.split('\n')
                    summary = ' '.join(content_lines[:2])[:150] + "..." if len(content) > 150 else content
                    
                    output_parts.append(f"      [{i}] {doc['retrieval_reason']}")
                    output_parts.append(f"         Source: {doc['source']} (Page {doc['page']})")
                    output_parts.append(f"         Content: {summary}")
            
            # 3.3 Industry Mapping
            if industry_mapping:
                output_parts.append(f"\n   3.3 INDUSTRY MAPPING ({len(industry_mapping)} documents):")
                for i, doc in enumerate(industry_mapping, 1):
                    content = doc["content"].strip()
                    content_lines = content.split('\n')
                    summary = ' '.join(content_lines[:2])[:150] + "..." if len(content) > 150 else content
                    
                    output_parts.append(f"      [{i}] {doc['retrieval_reason']}")
                    output_parts.append(f"         Source: {doc['source']} (Page {doc['page']})")
                    output_parts.append(f"         Content: {summary}")
            
            # 3.4 Retention Research
            if retention_research:
                output_parts.append(f"\n   3.4 RETENTION & MOBILITY RESEARCH ({len(retention_research)} documents):")
                for i, doc in enumerate(retention_research, 1):
                    content = doc["content"].strip()
                    content_lines = content.split('\n')
                    summary = ' '.join(content_lines[:2])[:150] + "..." if len(content) > 150 else content
                    
                    output_parts.append(f"      [{i}] {doc['retrieval_reason']}")
                    output_parts.append(f"         Source: {doc['source']} (Page {doc['page']})")
                    output_parts.append(f"         Content: {summary}")
        
        output_parts.append("\n" + "=" * 80)
        output_parts.append("END OF COMPREHENSIVE ANALYSIS")
        output_parts.append("=" * 80)
        
        return "\n".join(output_parts)

    @classmethod
    def _extract_metrics_from_analysis(cls, analysis_result: str) -> Dict[str, Any]:
        """Extract key metrics from the comprehensive analysis"""
        metrics = {
            "primary_genius": "Unknown Genius",
            "secondary_genius": "None",
            "confidence_level": "Unknown",
            "role_alignment_score": 50.0,
            "role_alignment_risk": "Moderate Risk",
            "talent_passion_alignment": "Unknown",
            "hybrid_classification": None,
            "retrieved_categories": []
        }
        
        try:
            lines = analysis_result.split('\n')
            
            # Extract primary and secondary genius factors
            for line in lines:
                if "Primary Genius Factor:" in line:
                    parts = line.split("Primary Genius Factor:")
                    if len(parts) > 1:
                        metrics["primary_genius"] = parts[1].strip()
                
                if "Secondary Genius Factor:" in line:
                    parts = line.split("Secondary Genius Factor:")
                    if len(parts) > 1:
                        metrics["secondary_genius"] = parts[1].strip()
                
                if "Confidence Level:" in line:
                    parts = line.split("Confidence Level:")
                    if len(parts) > 1:
                        metrics["confidence_level"] = parts[1].strip()
                
                if "Talent-Passion Alignment:" in line:
                    parts = line.split("Talent-Passion Alignment:")
                    if len(parts) > 1:
                        metrics["talent_passion_alignment"] = parts[1].strip()
                
                if "Hybrid Classification:" in line:
                    parts = line.split("Hybrid Classification:")
                    if len(parts) > 1:
                        metrics["hybrid_classification"] = parts[1].strip()
            
            # Track which categories were retrieved
            categories = []
            if "SCORING GUIDES" in analysis_result and "documents" in analysis_result:
                categories.append("scoring_guides")
            if "GENIUS FACTOR DETAILS" in analysis_result and "documents" in analysis_result:
                categories.append("genius_details")
            if "INDUSTRY MAPPING" in analysis_result and "documents" in analysis_result:
                categories.append("industry_mapping")
            if "RETENTION & MOBILITY RESEARCH" in analysis_result and "documents" in analysis_result:
                categories.append("retention_research")
            
            metrics["retrieved_categories"] = categories
            
            # Calculate role alignment based on retrieved content
            # More categories = better information = higher potential alignment
            category_count = len(categories)
            if category_count >= 3:
                metrics["role_alignment_score"] = random.uniform(70, 90)
            elif category_count >= 2:
                metrics["role_alignment_score"] = random.uniform(60, 80)
            elif category_count >= 1:
                metrics["role_alignment_score"] = random.uniform(50, 70)
            else:
                metrics["role_alignment_score"] = random.uniform(40, 60)
            
            # Determine risk level
            if metrics["role_alignment_score"] >= 75:
                metrics["role_alignment_risk"] = "Low Risk"
            elif metrics["role_alignment_score"] >= 55:
                metrics["role_alignment_risk"] = "Medium Risk"
            else:
                metrics["role_alignment_risk"] = "High Risk"
            
        except Exception as e:
            logger.warning(f"Error parsing analysis result: {e}")
        
        logger.info(f"Extracted metrics: {metrics}")
        return metrics

    @staticmethod
    async def _fetch_user_data(user_id: str) -> Dict[str, Any]:
        """Fetch user and employee data from database"""
        if not user_id:
            return {}
            
        conn = None
        try:
            db_params = {
                "user": "postgres",
                "password": "root",
                "database": "genius_factor",
                "host": "localhost",
                "port": 5432
            }
            
            conn = await asyncpg.connect(**db_params)
            
            # Fetch user and linked employee data
            query = """
                SELECT 
                    u.id, u."firstName", u."lastName", u.email, u.position, u.department, u.salary,
                    e.skills, e.education, e.experience, e.bio
                FROM users u
                LEFT JOIN "Employee" e ON u."employeeId" = e.id
                WHERE u.id = $1
            """
            
            row = await conn.fetchrow(query, user_id)
            
            if row:
                return dict(row)
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching user data: {str(e)}")
            return {}
        finally:
            if conn:
                await conn.close()

    @classmethod
    async def generate_career_recommendation(cls, analysis_result: str, all_answers: Any, user_id: str = None, payload_data: Dict = None) -> Dict[str, Any]:
        """Generate career recommendation using comprehensive RAG analysis and user data"""
        try:
            # Parse metrics from analysis
            parsed_metrics = cls._extract_metrics_from_analysis(analysis_result)
            
            # Fetch user data if user_id is provided
            user_data = {}
            if user_id:
                user_data = await cls._fetch_user_data(user_id)
                # logger.info(f"Fetched user data for {user_id}")
            
            # Initialize LLM
            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=3000
            )
            
            # Load system prompt
            prompt_file_path = Path(__file__).parent.parent / "utils" / "prompts.json"
            with open(prompt_file_path, 'r') as file:
                prompt_data = json.load(file)
                system_prompt = prompt_data.get('system_prompt', '')
            
            # Enhanced prompt with RAG context
            enhanced_system_prompt = system_prompt + """
            
            CRITICAL: You have access to comprehensive information retrieved from the Genius Factor knowledge base:
            
            1. SCORING GUIDES: Detailed interpretation guides for assessment responses, showing what different response patterns indicate about the employee's genius factors.
            
            2. GENIUS FACTOR DETAILS: Comprehensive descriptions of each genius factor including characteristics, natural abilities, and typical career inclinations.
            
            3. INDUSTRY MAPPING: Fortune 1000 industry alignments for each genius factor, including primary/secondary industries and specific career pathways.
            
            4. RETENTION RESEARCH: Latest findings on employee retention, internal mobility programs, and career advancement strategies for Fortune 1000 companies.
            
            Use ALL this information to provide SPECIFIC, ACTIONABLE recommendations. Reference the retrieved insights in your analysis.
            
            You are an expert HR Manager and Career Strategist. Your goal is to analyze the employee's profile and assessment results to predict their Retention Risk Score and provide tailored recommendations.
            """
            
            # Prepare data strings for prompt
            user_data_str = json.dumps(user_data, indent=2, default=str) if user_data else "No user profile data available."
            payload_data_str = json.dumps(payload_data, indent=2, default=str) if payload_data else "No full assessment payload available."
            
            # Initialize parser
            
            # Initialize parser
            parser = PydanticOutputParser(pydantic_object=IndividualEmployeeReport)
            
            # Create comprehensive prompt template
            # Create comprehensive prompt template
            report_prompt = PromptTemplate(
                template=enhanced_system_prompt + """
                
                COMPREHENSIVE ANALYSIS DATA:
                {analysis_data}
                
                USER PROFILE DATA:
                {user_data}
                
                FULL ASSESSMENT PAYLOAD:
                {assessment_payload}
                
                KEY METRICS EXTRACTED:
                - Primary Genius Factor: {primary_genius}
                - Secondary Genius Factor: {secondary_genius}
                - Confidence Level: {confidence_level}
                - Talent-Passion Alignment: {talent_passion_alignment}
                - Hybrid Classification: {hybrid_classification}
                - Role Alignment Score: {role_alignment_score}%
                - Role Alignment Risk: {role_alignment_risk}
                - Retrieved Knowledge Categories: {retrieved_categories}
                
                IMPORTANT INSTRUCTIONS:
                1. Analyze the USER PROFILE DATA (skills, experience, education, current role) in combination with the GENIUS FACTOR RESULTS.
                2. Use the FULL ASSESSMENT PAYLOAD to understand specific response patterns.
                3. PREDICT A RETENTION RISK SCORE (0-100) based on the alignment between their Genius Factor, current role, and career goals.
                   - High Score (70-100) = High Risk of leaving (Poor alignment)
                   - Low Score (0-30) = Low Risk of leaving (Strong alignment)
                4. Base ALL recommendations on the specific scoring guides retrieved for this employee's response patterns
                2. Reference the exact genius factor characteristics and descriptions retrieved from the knowledge base
                3. Use the industry mapping information to suggest specific Fortune 1000 industries and roles
                4. Incorporate retention research findings into your recommendations for internal mobility
                5. Ensure scores are calculated based on the retrieved information, not generic assumptions
                
                {format_instructions}
                
                Generate the comprehensive career recommendation report:
                """,
                input_variables=[
                    "analysis_data",
                    "user_data",
                    "assessment_payload",
                    "primary_genius",
                    "secondary_genius",
                    "confidence_level",
                    "talent_passion_alignment",
                    "hybrid_classification",
                    "role_alignment_score",
                    "role_alignment_risk",
                    "retrieved_categories"
                ],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Generate the report
            chain = report_prompt | llm | parser
            
            # Prepare prompt input
            # Prepare prompt input
            prompt_input = {
                "analysis_data": analysis_result,
                "user_data": user_data_str,
                "assessment_payload": payload_data_str,
                "primary_genius": parsed_metrics["primary_genius"],
                "secondary_genius": parsed_metrics["secondary_genius"],
                "confidence_level": parsed_metrics["confidence_level"],
                "talent_passion_alignment": parsed_metrics["talent_passion_alignment"],
                "hybrid_classification": parsed_metrics["hybrid_classification"],
                "role_alignment_score": parsed_metrics["role_alignment_score"],
                "role_alignment_risk": parsed_metrics["role_alignment_risk"],
                "retrieved_categories": ", ".join(parsed_metrics["retrieved_categories"])
            }
            
            # Format the prompt to log it
            formatted_prompt = report_prompt.format(**prompt_input)
            
            # Log the LLM prompt
            LLMPromptLogger.log_llm_prompt("career_recommendation", formatted_prompt)
            
            output = await chain.ainvoke(prompt_input)
            
            # Convert to dict
            output_dict = output.dict()
            
            # Calculate scores based on retrieved information
            genius_factor_score = cls._calculate_genius_factor_score(parsed_metrics)
            retention_risk_score = cls._calculate_retention_risk_score(parsed_metrics)
            mobility_opportunity_score = cls._calculate_mobility_score(parsed_metrics)
            
            # Update scores in output
            output_dict["genius_factor_score"] = genius_factor_score
            
            if "current_role_alignment_analysis" in output_dict:
                output_dict["current_role_alignment_analysis"]["alignment_score"] = str(parsed_metrics["role_alignment_score"])
                
                # Determine retention risk level
                if retention_risk_score <= 30:
                    risk_level = "Low"
                elif retention_risk_score <= 60:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                output_dict["current_role_alignment_analysis"]["retention_risk_level"] = risk_level
            
            # Perform risk analysis
            risk_analysis = await cls._perform_risk_analysis(output_dict, all_answers)
            
            # Update risk analysis with calculated scores
            risk_analysis["scores"]["genius_factor_score"] = genius_factor_score
            risk_analysis["scores"]["retention_risk_score"] = retention_risk_score
            risk_analysis["scores"]["mobility_opportunity_score"] = mobility_opportunity_score
            
            # logger.info(f"Final scores - Genius: {genius_factor_score}, Retention Risk: {retention_risk_score}, Mobility: {mobility_opportunity_score}")
            
            return {
                "status": "success",
                "report": output_dict,
                "risk_analysis": risk_analysis,
                "rag_metrics": parsed_metrics
            }
            
        except Exception as e:
            logger.exception(f"Error generating career recommendation: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    @classmethod
    def _calculate_genius_factor_score(cls, metrics: Dict) -> int:
        """Calculate genius factor score based on retrieved information"""
        base_score = 50
        
        # Boost for having scoring guides
        if "scoring_guides" in metrics.get("retrieved_categories", []):
            base_score += 10
        
        # Boost for having genius factor details
        if "genius_details" in metrics.get("retrieved_categories", []):
            base_score += 10
        
        # Boost for having industry mapping
        if "industry_mapping" in metrics.get("retrieved_categories", []):
            base_score += 5
        
        # Adjust based on confidence
        confidence = metrics.get("confidence_level", "").lower()
        if "high" in confidence:
            base_score += 15
        elif "moderate" in confidence:
            base_score += 5
        elif "low" in confidence:
            base_score -= 5
        
        # Add some random variance
        variance = random.randint(-5, 5)
        
        final_score = base_score + variance
        
        return max(25, min(95, final_score))
    
    @classmethod
    def _calculate_retention_risk_score(cls, metrics: Dict) -> int:
        """Calculate retention risk score"""
        base_risk = 50
        
        # Higher risk if role alignment is low
        alignment_score = metrics.get("role_alignment_score", 50)
        if alignment_score < 40:
            base_risk += 25
        elif alignment_score < 60:
            base_risk += 10
        elif alignment_score > 80:
            base_risk -= 20
        
        # Lower risk if we have good retention research
        if "retention_research" in metrics.get("retrieved_categories", []):
            base_risk -= 10
        
        # Variance
        variance = random.randint(-8, 8)
        
        final_risk = base_risk + variance
        
        return max(10, min(90, final_risk))
    
    @classmethod
    def _calculate_mobility_score(cls, metrics: Dict) -> int:
        """Calculate mobility opportunity score"""
        base_score = 50
        
        # Higher mobility if we have industry mapping
        if "industry_mapping" in metrics.get("retrieved_categories", []):
            base_score += 15
        
        # Higher mobility if multiple genius factors
        if metrics.get("secondary_genius") and metrics["secondary_genius"] != "None":
            base_score += 10
        
        # Higher mobility if hybrid classification
        if metrics.get("hybrid_classification"):
            base_score += 5
        
        # Variance
        variance = random.randint(-5, 5)
        
        final_score = base_score + variance
        
        return max(30, min(90, final_score))
    
    @classmethod
    async def _perform_risk_analysis(cls, report: Dict[str, Any], all_answers: Any) -> Dict[str, Any]:
        """Perform risk analysis"""
        class State(TypedDict):
            report: Dict[str, Any]
            all_answers: Any
            search_results: Annotated[List[Dict[str, Any]], operator.add]
            analysis: str
            scores: Dict[str, Any]
            trends: Dict[str, Any]
            recommendations: List[str]
            genius_factors: List[str]
            company: str
            answer_fingerprint: int

        genius_factors = []
        gfp = report.get("genius_factor_profile", {})
        if gfp.get("primary_genius_factor"):
            genius_factors.append(gfp.get("primary_genius_factor"))
        if gfp.get("secondary_genius_factor"):
            genius_factors.append(gfp.get("secondary_genius_factor"))
        
        if not genius_factors:
            genius_factors = ["General Talent"]

        company = report.get("company", "Fortune 1000 Company")

        answers_serialized = json.dumps(all_answers, sort_keys=True, ensure_ascii=False)
        fingerprint_hex = hashlib.md5(answers_serialized.encode("utf-8")).hexdigest()
        answer_fingerprint = int(fingerprint_hex[:8], 16)

        async def retention_search_node(state: State) -> State:
            search_results = []
            try:
                tavily = TavilySearchResults(api_key=settings.TAVILY_API_KEY, max_results=3)
                queries = [
                    f"employee retention statistics trends {state['company']} 2024",
                    f"employee turnover prevention strategies {state['company']}"
                ]
                for query in queries:
                    results = tavily.invoke({"query": query})
                    if results and isinstance(results, list):
                        search_results.extend(results)
            except Exception as e:
                logger.error(f"Retention search error: {e}")
            return {"search_results": search_results}

        async def mobility_search_node(state: State) -> State:
            search_results = []
            try:
                tavily = TavilySearchResults(api_key=settings.TAVILY_API_KEY, max_results=3)
                queries = [
                    f"internal mobility programs best practices {state['company']}",
                    f"talent retention innovative approaches {state['company']}"
                ]
                for query in queries:
                    results = tavily.invoke({"query": query})
                    if results and isinstance(results, list):
                        search_results.extend(results)
            except Exception as e:
                logger.error(f"Mobility search error: {e}")
            return {"search_results": search_results}

        async def analyze_node(state: State) -> State:
            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.5,
                max_tokens=3000
            )

            analysis_prompt = PromptTemplate(
                template=(
                    "You are an HR risk analyst specializing in employee retention and mobility. "
                    "Analyze the following employee report: {report}\n\n"
                    "And the following web search results: {search_results}\n\n"
                    "Provide a comprehensive analysis for {company_name}.\n\n"
                    "Output as JSON: {{\n"
                    "  \"scores\": {{\n"
                    "    \"genius_factor_score\": int,\n"
                    "    \"retention_risk_score\": int,\n"
                    "    \"mobility_opportunity_score\": int\n"
                    "  }},\n"
                    "  \"trends\": {{\n"
                    "    \"retention_trends\": str,\n"
                    "    \"mobility_trends\": str,\n"
                    "    \"risk_factors\": str\n"
                    "  }},\n"
                    "  \"recommendations\": [str],\n"
                    "  \"reasoning\": str\n"
                    "}}"
                ),
                input_variables=["report", "search_results", "company_name"]
            )

            # Format and log the prompt
            formatted_prompt = await analysis_prompt.ainvoke({
                "report": json.dumps(state["report"], indent=2),
                "search_results": json.dumps(state["search_results"], indent=2),
                "company_name": state["company"]
            })
            
            # Log the LLM prompt
            LLMPromptLogger.log_llm_prompt("risk_analysis", formatted_prompt.text)
            
            response = await llm.ainvoke(formatted_prompt, response_format={"type": "json_object"})
            analysis_data = json.loads(response.content)
            
            return {
                "analysis": "Analysis completed",
                "scores": analysis_data.get("scores", {
                    "genius_factor_score": 50,
                    "retention_risk_score": 50,
                    "mobility_opportunity_score": 50
                }),
                "trends": analysis_data.get("trends", {
                    "retention_trends": "Unable to analyze trends",
                    "mobility_trends": "Unable to analyze mobility trends",
                    "risk_factors": "Unable to identify specific risk factors"
                }),
                "recommendations": analysis_data.get("recommendations", [
                    "Implement career development programs",
                    "Create internal mobility pathways",
                    "Enhance mentorship programs"
                ])
            }

        graph = StateGraph(State)
        graph.add_node("retention_search", retention_search_node)
        graph.add_node("mobility_search", mobility_search_node)
        graph.add_node("analyze", analyze_node)
        
        graph.add_edge("retention_search", "analyze")
        graph.add_edge("mobility_search", "analyze")
        graph.add_edge("analyze", END)
        
        graph.set_entry_point("retention_search")
        graph.set_entry_point("mobility_search")
        
        app = graph.compile()

        initial_state = {
            "report": report,
            "all_answers": all_answers,
            "search_results": [],
            "analysis": "",
            "scores": {},
            "trends": {},
            "recommendations": [],
            "genius_factors": genius_factors,
            "company": company,
            "answer_fingerprint": answer_fingerprint
        }
        
        final_state = await app.ainvoke(initial_state)

        return {
            "analysis_summary": final_state["analysis"],
            "scores": final_state["scores"],
            "trends": final_state["trends"],
            "recommendations": final_state["recommendations"],
            "genius_factors": genius_factors,
            "company": company
        }