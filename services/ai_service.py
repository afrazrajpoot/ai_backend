import json
import os
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Annotated
import hashlib

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
import operator
from utils.logger import logger
from config import settings
from utils.models import IndividualEmployeeReport
from utils.analysis_utils import MAPPING_FACTORS
import asyncio
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

        faiss_path = "faiss_index"
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

        for i, pdf_file in enumerate(pdf_files, 1):
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
    def get_vector_store_stats(cls) -> Dict[str, Any]:
        if cls._vector_store is None:
            return {"status": "not_initialized"}
        try:
            index_info = {
                "total_vectors": cls._vector_store.index.ntotal,
                "vector_dimension": cls._vector_store.index.d,
                "is_trained": cls._vector_store.index.is_trained
            }
            return {
                "status": "initialized",
                "index_info": index_info,
                "docstore_size": len(cls._vector_store.docstore._dict) if hasattr(cls._vector_store.docstore, '_dict') else 0
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    # inside your ai_service class (replace existing analyze_majority_answers)
    @classmethod
    async def analyze_majority_answers(cls, basic_results: List[Dict[str, Any]], deep_results: Dict[str,Any] = None) -> str:
        """
        Enhanced RAG step: prefer deep_results primary/secondary; fallback to basic_results.
        Returns formatted text (or any structure you prefer).
        """
        vector_store = cls.initialize_vector_store()
        # Determine primary/secondary names
        primary_name = None
        secondary_name = None
        hybrid_classification = None
        confidence_level = None
        alignment_label = None
        current_role_alignment_label = None

        if deep_results:
            prim = deep_results.get("primary_genius", [])
            sec = deep_results.get("secondary_genius", [])
            if prim:
                primary_name = prim[0].get("name")
                primary_qualities = prim[0].get("qualities", [])
            else:
                primary_qualities = []
            if sec:
                secondary_name = sec[0].get("name")
                secondary_qualities = sec[0].get("qualities", [])
            else:
                secondary_qualities = []
            hybrid_classification = deep_results.get("hybrid_classification")
            hybrid_qualities = deep_results.get("hybrid_qualities", [])
            confidence_level = deep_results.get("confidence_level")
            alignment_label = deep_results.get("talent_passion_alignment_label")
            # --- Current Role Alignment Calculation (Dynamic) ---
            import logging
            import random
            logger = logging.getLogger("ai_service")
            
            # Get user's current role from multiple sources
            current_role = (deep_results.get("departement") or 
                          deep_results.get("department") or 
                          deep_results.get("current_role") or
                          "Unknown Role")
            logger.info(f"[RoleAlign] Analyzing alignment for role: {current_role}")
            
            # Get user's genius factor profile from mapping counts
            mapping_counts = deep_results.get("section_counts", {}).get("Mapping", {})
            mapping_total = sum(mapping_counts.values()) or 1
            user_profile = {k: v / mapping_total for k, v in mapping_counts.items()}
            logger.info(f"[RoleAlign] User genius profile: {user_profile}")
            
            # Enhanced role-to-genius mapping that includes secondary factors
            def get_expected_genius_profile(role_text):
                """Dynamically determine expected genius factors for a role, including secondary factors"""
                role_lower = role_text.lower()
                
                # Tech roles - expect high A (Tech Genius), with F (Number Genius) as valuable secondary
                if any(word in role_lower for word in ['engineer', 'developer', 'programmer', 'software', 'data', 'analyst', 'tech']):
                    return {
                        "A": 0.45,  # Primary Tech Genius
                        "F": 0.25,  # Secondary Number Genius (important for technical roles)
                        "B": 0.12,  # Some business acumen
                        "C": 0.08,  # Some creative thinking
                        "D": 0.06,  # Communication skills
                        "E": 0.04   # Operations understanding
                    }
                
                # Business/Management roles - expect high B (Business Genius)
                elif any(word in role_lower for word in ['manager', 'director', 'executive', 'business', 'lead', 'supervisor']):
                    return {
                        "B": 0.40,  # Primary Business Genius
                        "D": 0.20,  # Communication crucial
                        "A": 0.15,  # Tech understanding
                        "E": 0.15,  # Operations/Strategic
                        "C": 0.10   # Some creativity
                    }
                
                # Creative roles - expect high C (Creative Genius)
                elif any(word in role_lower for word in ['designer', 'artist', 'creative', 'marketing', 'brand']):
                    return {
                        "C": 0.45,  # Primary Creative Genius
                        "D": 0.20,  # Communication important
                        "A": 0.15,  # Tech skills valuable
                        "B": 0.15,  # Business understanding
                        "E": 0.05   # Operations
                    }
                
                # Service/Support roles - expect high D (Service Genius)
                elif any(word in role_lower for word in ['support', 'customer', 'service', 'hr', 'human']):
                    return {
                        "D": 0.40,  # Primary Service Genius
                        "B": 0.25,  # Business skills important
                        "A": 0.15,  # Tech understanding helpful
                        "E": 0.12,  # Process orientation
                        "C": 0.08   # Some creativity
                    }
                
                # Operations/Strategic roles - expect high E (Strategic Genius)
                elif any(word in role_lower for word in ['operations', 'strategy', 'planning', 'consultant']):
                    return {
                        "E": 0.40,  # Primary Strategic Genius
                        "B": 0.25,  # Business acumen crucial
                        "A": 0.15,  # Analytical skills
                        "D": 0.12,  # Communication
                        "C": 0.08   # Innovation thinking
                    }
                
                # Sales roles - mix of B and D with communication focus
                elif any(word in role_lower for word in ['sales', 'account']):
                    return {
                        "D": 0.35,  # Primary Communication
                        "B": 0.30,  # Business skills
                        "E": 0.15,  # Strategic thinking
                        "A": 0.12,  # Tech understanding
                        "C": 0.08   # Creative approach
                    }
                
                # Healthcare roles - mix of D and B with service focus
                elif any(word in role_lower for word in ['healthcare', 'medical', 'nurse', 'doctor']):
                    return {
                        "D": 0.35,  # Service/Communication
                        "B": 0.25,  # Management/Business
                        "A": 0.20,  # Analytical/Systems thinking
                        "E": 0.15,  # Process optimization
                        "C": 0.05   # Creative problem solving
                    }
                
                # Default balanced profile for unknown roles
                else:
                    return {"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20, "E": 0.20}
            
            expected_profile = get_expected_genius_profile(current_role)
            logger.info(f"[RoleAlign] Expected genius profile for '{current_role}': {expected_profile}")
            
            # Enhanced alignment calculation considering primary, secondary, and hybrid factors
            current_role_alignment_pct = None
            current_role_alignment_risk = "Unknown"
            
            if user_profile and expected_profile:
                # Get user's primary and secondary genius factors
                prim = deep_results.get("primary_genius", [])
                sec = deep_results.get("secondary_genius", [])
                hybrid_classification = deep_results.get("hybrid_classification")
                
                # Method 1: Traditional weighted overlap (base score)
                base_alignment_score = 0.0
                total_expected_weight = 0.0
                
                for genius_letter, expected_weight in expected_profile.items():
                    user_weight = user_profile.get(genius_letter, 0)
                    alignment_contribution = min(user_weight, expected_weight)
                    base_alignment_score += alignment_contribution
                    total_expected_weight += expected_weight
                
                base_alignment_pct = (base_alignment_score / total_expected_weight) * 100 if total_expected_weight > 0 else 0
                logger.info(f"[RoleAlign] Base alignment score: {base_alignment_pct:.1f}%")
                
                # Method 2: Bonus for primary/secondary genius alignment
                genius_alignment_bonus = 0.0
                
                if prim:
                    primary_letter = prim[0].get("letter", "")
                    primary_strength = prim[0].get("percentage", 0)
                    expected_primary_weight = expected_profile.get(primary_letter, 0)
                    
                    if expected_primary_weight > 0:
                        # Bonus based on how well primary genius aligns with role expectations
                        # Scaled to avoid over-scoring: max 15% bonus for perfect primary alignment
                        primary_bonus = min(0.15, (expected_primary_weight * 0.4) * (primary_strength / 100))
                        genius_alignment_bonus += primary_bonus
                        logger.info(f"[RoleAlign] Primary genius bonus: {primary_bonus:.3f} (letter: {primary_letter}, strength: {primary_strength}%, expected: {expected_primary_weight:.2f})")
                
                if sec:
                    secondary_letter = sec[0].get("letter", "")
                    secondary_strength = sec[0].get("percentage", 0)
                    expected_secondary_weight = expected_profile.get(secondary_letter, 0)
                    
                    if expected_secondary_weight > 0:
                        # Bonus for secondary genius alignment (weighted lower than primary)
                        # Scaled to avoid over-scoring: max 10% bonus for perfect secondary alignment
                        secondary_bonus = min(0.10, (expected_secondary_weight * 0.3) * (secondary_strength / 100))
                        genius_alignment_bonus += secondary_bonus
                        logger.info(f"[RoleAlign] Secondary genius bonus: {secondary_bonus:.3f} (letter: {secondary_letter}, strength: {secondary_strength}%, expected: {expected_secondary_weight:.2f})")
                
                # Method 3: Hybrid bonus if applicable
                hybrid_bonus = 0.0
                if hybrid_classification and prim and sec:
                    # Additional bonus for balanced hybrid profile
                    primary_letter = prim[0].get("letter", "")
                    secondary_letter = sec[0].get("letter", "")
                    
                    both_expected = (expected_profile.get(primary_letter, 0) > 0.1 and 
                                   expected_profile.get(secondary_letter, 0) > 0.1)
                    
                    if both_expected:
                        hybrid_bonus = 0.05  # 5% bonus for having both expected factors
                        logger.info(f"[RoleAlign] Hybrid bonus applied: {hybrid_bonus:.3f} for balanced {primary_letter}+{secondary_letter} profile")
                
                # Combine all scores with cap to prevent over-scoring
                final_alignment_score = min(95, base_alignment_pct + (genius_alignment_bonus * 100) + (hybrid_bonus * 100))
                
                # Apply realistic variation
                variation = random.uniform(-3, 6)  # Slightly optimistic bias, smaller range
                current_role_alignment_pct = max(0, min(100, final_alignment_score + variation))
                current_role_alignment_pct = round(current_role_alignment_pct, 1)
                
                logger.info(f"[RoleAlign] Final calculation: base={base_alignment_pct:.1f}% + genius_bonus={genius_alignment_bonus*100:.1f}% + hybrid_bonus={hybrid_bonus*100:.1f}% + variation={variation:.1f}% = {current_role_alignment_pct}%")
                
                # Determine risk level
                if current_role_alignment_pct >= 71:
                    current_role_alignment_risk = "Low"
                elif current_role_alignment_pct >= 41:
                    current_role_alignment_risk = "Medium"
                else:
                    current_role_alignment_risk = "High"
            else:
                logger.info(f"[RoleAlign] Missing user profile or expected profile for alignment calculation")
            current_role_alignment_label = f"{current_role_alignment_risk} ({current_role_alignment_pct if current_role_alignment_pct is not None else '?'}%)"
        else:
            primary_qualities = []
            secondary_qualities = []
            hybrid_qualities = []

        # Final fallback check - only if both primary and secondary are still None
        if not primary_name:
            # Use the most common genius factor across all sections as final fallback
            all_counts = {}
            for part in basic_results:
                for opt, cnt in (part.get("optionCounts") or {}).items():
                    all_counts[opt] = all_counts.get(opt, 0) + cnt
            sorted_all = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
            if sorted_all:
                letter = sorted_all[0][0]
                primary_name = MAPPING_FACTORS.get(letter, {}).get("name", f"Unknown Genius ({letter})")
                primary_qualities = MAPPING_FACTORS.get(letter, {}).get("qualities", [])
                logger.info(f"[AI Service] Final fallback: Using most common letter '{letter}' -> {primary_name}")
                
                if len(sorted_all) > 1:
                    letter2 = sorted_all[1][0]
                    secondary_name = MAPPING_FACTORS.get(letter2, {}).get("name", f"Unknown Genius ({letter2})")
                    secondary_qualities = MAPPING_FACTORS.get(letter2, {}).get("qualities", [])
            else:
                primary_name = "Unidentified Genius"
                primary_qualities = []
                logger.warning("[AI Service] No genius factor could be determined from assessment data")
        
        if not secondary_name:
            secondary_name = "None Identified" if primary_name != "Unidentified Genius" else primary_name
            if primary_name != "Unidentified Genius":
                secondary_qualities = []

        logger.info(f"RAG primary={primary_name} secondary={secondary_name}")

        queries = [
            f'"{primary_name}" OR "{secondary_name}"',
            f'"{primary_name}" OR "{secondary_name}"',
            f'"{primary_name}" OR "{secondary_name}"',
            f'"{primary_name}" OR "{secondary_name}"'
        ]

        source_files = [
            "(68 Questions) Genius Factor Assessment for Fortune 1000 HR Departments.pdf",
            "Genius Factor to Fortune 1000 Industry Mapping.pdf",
            "Genius Factor Framework Analysis.pdf",
            "retention & internal mobility research_findings.pdf"
        ]

        all_docs = []
        for query, source_basename in zip(queries, source_files):
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 500, "filter": {"source_file": source_basename}}
            )
            try:
                docs = await retriever.aget_relevant_documents(query)
                if docs:
                    all_docs.extend(docs)
                    logger.info(f"Retrieved {len(docs)} docs from {source_basename}")
            except Exception as e:
                logger.exception(f"Error retrieving documents for {source_basename}: {e}")

        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = (doc.metadata.get("source_file", ""), doc.metadata.get("page", -1), doc.page_content[:400])
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # Format output
        output_parts = []
        output_parts.append(f"Genius Factors Identified: Primary = {primary_name}, Secondary = {secondary_name}")
        if primary_qualities:
            output_parts.append(f"Primary Qualities: {', '.join(primary_qualities)}")
        if secondary_qualities:
            output_parts.append(f"Secondary Qualities: {', '.join(secondary_qualities)}")
        if hybrid_classification:
            output_parts.append(f"Hybrid Classification: {hybrid_classification}")
        if hybrid_qualities:
            output_parts.append(f"Hybrid Qualities: {', '.join(hybrid_qualities)}")
        if confidence_level:
            output_parts.append(f"Confidence Level: {confidence_level}")
        if alignment_label:
            output_parts.append(f"Talent-Passion Alignment: {alignment_label}")
        if current_role_alignment_label:
            output_parts.append(f"Current Role Alignment: {current_role_alignment_label}")
        output_parts.append("")
        output_parts.append("Responses snapshot (basic majority):")
        for pr in basic_results:
            output_parts.append(f"Part: {pr.get('part')} | Majority: {pr.get('majorityOptions')} | MaxCount: {pr.get('maxCount')}")

        output_parts.append(f"\nDocuments Retrieved: {len(unique_docs)}")
        for i, doc in enumerate(unique_docs, 1):
            src = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", -1)
            content = doc.page_content.replace("\n", " ").strip()[:2000]
            output_parts.append(f"=== Document {i} ===\nSource: {src} (Page {page})\n{content}\n")

        result_text = "\n".join(output_parts)
        return result_text

    @classmethod
    def _extract_metrics_from_analysis(cls, analysis_result: str) -> Dict[str, Any]:
        """
        Extract key metrics from the analysis result string for accurate report generation
        """
        import re
        
        metrics = {
            "primary_genius": "Unknown Genius",
            "secondary_genius": "None",
            "confidence_level": "Unknown",
            "role_alignment_score": 50.0,
            "role_alignment_risk": "Moderate Risk",
            "talent_passion_alignment": "Unknown",
            "hybrid_classification": None
        }
        
        try:
            # Extract Primary and Secondary Genius Factors
            primary_match = re.search(r"Primary = ([^,\n]+)", analysis_result)
            if primary_match:
                metrics["primary_genius"] = primary_match.group(1).strip()
                
            secondary_match = re.search(r"Secondary = ([^,\n]+)", analysis_result)
            if secondary_match:
                metrics["secondary_genius"] = secondary_match.group(1).strip()
            
            # Extract Confidence Level
            confidence_match = re.search(r"Confidence Level:\s*([^\n]+)", analysis_result)
            if confidence_match:
                metrics["confidence_level"] = confidence_match.group(1).strip()
            
            # Extract Role Alignment with score and risk level
            role_align_match = re.search(r"Current Role Alignment:\s*([^(]+)\(([0-9.]+)%\)", analysis_result)
            if role_align_match:
                metrics["role_alignment_risk"] = role_align_match.group(1).strip()
                metrics["role_alignment_score"] = float(role_align_match.group(2))
            
            # Extract Talent-Passion Alignment
            talent_passion_match = re.search(r"Talent-Passion Alignment:\s*([^\n]+)", analysis_result)
            if talent_passion_match:
                metrics["talent_passion_alignment"] = talent_passion_match.group(1).strip()
                
            # Extract Hybrid Classification
            hybrid_match = re.search(r"Hybrid Classification:\s*([^\n]+)", analysis_result)
            if hybrid_match:
                metrics["hybrid_classification"] = hybrid_match.group(1).strip()
                
        except Exception as e:
            logger.warning(f"Error parsing analysis result: {e}")
            
        logger.info(f"Extracted metrics: {metrics}")
        return metrics



    @classmethod
    async def generate_career_recommendation(cls, analysis_result: str, all_answers: Any) -> Dict[str, Any]:
        try:
            # Check if analysis_result is empty or invalid
            logger.info(f"Analysis result received: {analysis_result[:500]}...")
            if not analysis_result or not isinstance(analysis_result, str):
                return {"status": "error", "message": "Invalid or empty analysis results provided"}

            # Parse key metrics from analysis_result
            parsed_metrics = cls._extract_metrics_from_analysis(analysis_result)
            logger.info(f"Parsed metrics: {parsed_metrics}")

            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=3000
            )

            # Load system prompt with enhanced context
            prompt_file_path = Path(__file__).parent.parent / "utils" / "prompts.json"
            with open(prompt_file_path, 'r') as file:
                prompt_data = json.load(file)
                system_prompt = prompt_data.get('system_prompt', '')
                if not system_prompt:
                    raise ValueError("System prompt not found in JSON file")

            # Enhanced system prompt with specific metric integration
            enhanced_system_prompt = system_prompt + f"""

            CRITICAL SCORING INSTRUCTIONS:
            Use these EXACT values from the analysis results:
            - Primary Genius Factor: {parsed_metrics['primary_genius']} 
            - Secondary Genius Factor: {parsed_metrics['secondary_genius']}
            - Confidence Level: {parsed_metrics['confidence_level']}
            - Current Role Alignment: {parsed_metrics['role_alignment_score']}% ({parsed_metrics['role_alignment_risk']})
            - Talent-Passion Alignment: {parsed_metrics['talent_passion_alignment']}
            - Hybrid Classification: {parsed_metrics['hybrid_classification']}

            IMPORTANT: 
            1. Use the role_alignment_score ({parsed_metrics['role_alignment_score']}%) EXACTLY as provided in current_role_alignment_analysis.alignment_score
            2. Calculate genius_factor_score based on confidence level: 
            - High confidence: 80-95
            - Moderate confidence: 65-80  
            - Low confidence: 45-65
            - Very Low confidence: 25-45
            3. For retention risk assessment, consider role alignment and confidence:
            - Role alignment <40%: High retention risk (70-90)
            - Role alignment 40-70%: Moderate retention risk (40-70)
            - Role alignment >70%: Low retention risk (10-40)
            4. Mobility scores should reflect realistic opportunities based on genius factors and current role fit"""

            # Initialize Pydantic output parser
            parser = PydanticOutputParser(pydantic_object=IndividualEmployeeReport)

            # Enhanced prompt template with parsed metrics
            report_prompt = PromptTemplate(
                template=enhanced_system_prompt + "\n\nAnalysis Data:\n{analysis_data}\n\nParsed Metrics:\n{parsed_metrics}\n\n{format_instructions}\nGenerate the report:",
                input_variables=["analysis_data", "parsed_metrics"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            # Log the analysis data
            logger.debug(f"Analysis data: {analysis_result}")

            # Render and log the full prompt
            full_prompt_str = report_prompt.format(
                analysis_data=analysis_result,
                parsed_metrics=json.dumps(parsed_metrics, indent=2)
            )
            logger.debug(f"Full prompt:\n{full_prompt_str}")
        
            chain = report_prompt | llm | parser
            output = await chain.ainvoke({
                "analysis_data": analysis_result,
                "parsed_metrics": json.dumps(parsed_metrics, indent=2)
            })
            
            # Override certain fields with parsed values to ensure accuracy
            output_dict = output.dict()
            
            # Clean confidence level from genius factor names (remove "(Confidence: ...)" patterns)
            import re
            if "genius_factor_profile" in output_dict:
                gfp = output_dict["genius_factor_profile"]
                if "primary_genius_factor" in gfp:
                    # Remove confidence level pattern like "(Confidence: Moderate)" or "(Confidence: High)" etc.
                    cleaned_primary = re.sub(r'\s*\(Confidence:\s*[^)]+\)', '', gfp["primary_genius_factor"]).strip()
                    output_dict["genius_factor_profile"]["primary_genius_factor"] = cleaned_primary
                    
                if "secondary_genius_factor" in gfp and gfp["secondary_genius_factor"]:
                    # Remove confidence level pattern from secondary factor too
                    cleaned_secondary = re.sub(r'\s*\(Confidence:\s*[^)]+\)', '', gfp["secondary_genius_factor"]).strip()
                    output_dict["genius_factor_profile"]["secondary_genius_factor"] = cleaned_secondary
            
            # Ensure role alignment score matches exactly
            if "current_role_alignment_analysis" in output_dict:
                output_dict["current_role_alignment_analysis"]["alignment_score"] = str(parsed_metrics["role_alignment_score"])
            
            # Calculate realistic genius factor score based on confidence
            confidence_level = parsed_metrics["confidence_level"].lower()
            if "high" in confidence_level:
                genius_factor_score = 85 + (parsed_metrics["role_alignment_score"] - 50) * 0.2
            elif "moderate" in confidence_level:
                genius_factor_score = 70 + (parsed_metrics["role_alignment_score"] - 50) * 0.15  
            elif "low" in confidence_level:
                genius_factor_score = 55 + (parsed_metrics["role_alignment_score"] - 50) * 0.1
            else:  # very low
                genius_factor_score = 35 + (parsed_metrics["role_alignment_score"] - 50) * 0.05
                
            output_dict["genius_factor_score"] = max(25, min(95, int(genius_factor_score)))
            
            risk_analysis = await cls._perform_risk_analysis(output_dict, all_answers)
            
            # Override risk analysis scores with more realistic values based on parsed metrics
            if parsed_metrics["role_alignment_score"] < 40:
                risk_analysis["scores"]["retention_risk_score"] = min(90, int(70 + (40 - parsed_metrics["role_alignment_score"])))
            elif parsed_metrics["role_alignment_score"] < 70:
                risk_analysis["scores"]["retention_risk_score"] = int(40 + (70 - parsed_metrics["role_alignment_score"]) * 0.5)
            else:
                risk_analysis["scores"]["retention_risk_score"] = max(10, int(40 - (parsed_metrics["role_alignment_score"] - 70) * 0.8))
                
            # Now set retention risk level using the calculated retention risk score and proper risk categorization
            retention_risk_score = risk_analysis["scores"]["retention_risk_score"]
            if retention_risk_score <= 30:
                risk_category = "Low"
            elif retention_risk_score <= 60:
                risk_category = "Medium"  
            else:
                risk_category = "High"
                
            if "current_role_alignment_analysis" in output_dict:
                output_dict["current_role_alignment_analysis"]["retention_risk_level"] = f"{risk_category}"
                
            # Adjust genius factor score in risk analysis based on confidence and alignment
            base_genius_score = output_dict["genius_factor_score"]
            risk_analysis["scores"]["genius_factor_score"] = max(25, min(95, base_genius_score))
            
            # Mobility score based on genius factor strength and role alignment
            if parsed_metrics["role_alignment_score"] > 70:
                mobility_score = 40 + (parsed_metrics["role_alignment_score"] - 70) * 0.5  # Lower mobility need when well-aligned
            else:
                mobility_score = 60 + (70 - parsed_metrics["role_alignment_score"]) * 0.4   # Higher mobility opportunity when misaligned
                
            risk_analysis["scores"]["mobility_opportunity_score"] = max(30, min(90, int(mobility_score)))
            
            logger.info(f"Final scores: genius_factor={output_dict['genius_factor_score']}, retention_risk={risk_analysis['scores']['retention_risk_score']}, mobility={risk_analysis['scores']['mobility_opportunity_score']}")
         
            return {
                "status": "success",
                "report": output_dict,
                "risk_analysis": risk_analysis
            }

        except Exception as e:
            logger.error(f"Error in generate_career_recommendation: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    @classmethod
    
        
    async def _perform_risk_analysis(cls, report: Dict[str, Any], all_answers: Any) -> Dict[str, Any]:
            # Define state for LangGraph
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

            # Extract genius factors and company from report
            genius_factors = []
            try:
                gfp = report.get("genius_factor_profile", {})
                if isinstance(gfp, dict):
                    if gfp.get("primary_genius_factor"):
                        genius_factors.append(gfp.get("primary_genius_factor"))
                    if gfp.get("secondary_genius_factor"):
                        genius_factors.append(gfp.get("secondary_genius_factor"))
            except Exception:
                pass
            if not genius_factors:
                if "primary_genius_factor" in report:
                    genius_factors.append(report["primary_genius_factor"])
                if "secondary_genius_factor" in report:
                    genius_factors.append(report["secondary_genius_factor"])
            if not genius_factors:
                genius_factors = ["General Talent"]
                logger.warning("No genius factors in report, using default: General Talent")

            # Get company from report or use default
            company = report.get("company", "Fortune 1000 Company")

            # Build deterministic fingerprint from answers for variability
            try:
                answers_serialized = json.dumps(all_answers, sort_keys=True, ensure_ascii=False)
            except Exception:
                answers_serialized = str(all_answers)
            fingerprint_hex = hashlib.md5(answers_serialized.encode("utf-8")).hexdigest()
            answer_fingerprint = int(fingerprint_hex[:8], 16)

            # Log input report for debugging
            logger.debug(f"Input report: {json.dumps(report, indent=2)}")

            # Node for retention trends search
            async def retention_search_node(state: State) -> State:
                search_results = []
                try:
                    tavily = TavilySearchResults(api_key=settings.TAVILY_API_KEY, max_results=3)
                    queries = [
                        f"employee retention statistics trends {state['company']} 2024",
                        f"employee turnover prevention strategies {state['company']}"
                    ]
                    for query in queries:
                        try:
                            results = tavily.invoke({"query": query})
                            if results and isinstance(results, list):
                                search_results.extend(results)
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            logger.warning(f"Retention query failed for '{query}': {e}")
                            continue
                  
                except Exception as e:
                    logger.error(f"Retention search error: {e}")
                    search_results = [{
                        "title": f"Retention Trends for {state['company']}",
                        "content": f"Generic retention trends for {state['company']} focus on employee engagement and well-being.",
                        "url": "https://example.com/retention-trends"
                    }]
                logger.debug(f"Retention search results: {json.dumps(search_results, indent=2)}")
                return {"search_results": search_results}

            # Node for mobility programs search
            async def mobility_search_node(state: State) -> State:
                search_results = []
                try:
                    tavily = TavilySearchResults(api_key=settings.TAVILY_API_KEY, max_results=3)
                    queries = [
                        f"internal mobility programs best practices {state['company']}",
                        f"talent retention innovative approaches {state['company']}"
                    ]
                    for query in queries:
                        try:
                            results = tavily.invoke({"query": query})
                            if results and isinstance(results, list):
                                search_results.extend(results)
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            logger.warning(f"Mobility query failed for '{query}': {e}")
                            continue
                 
                except Exception as e:
                    logger.error(f"Mobility search error: {e}")
                    search_results = [{
                        "title": f"Mobility Trends for {state['company']}",
                        "content": f"Generic mobility trends for {state['company']} emphasize upskilling and career pathing.",
                        "url": "https://example.com/mobility-trends"
                    }]
                logger.debug(f"Mobility search results: {json.dumps(search_results, indent=2)}")
                return {"search_results": search_results}

            # Node to analyze search results and compute comprehensive analysis
            async def analyze_node(state: State) -> State:
                llm = ChatOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    model="gpt-4o-mini",
                    temperature=0.5,  # Increased for variability
                    max_tokens=3000
                )

                # Enhanced prompt for score variability, now including all_answers
                analysis_prompt = PromptTemplate(
                    template=(
                        "You are an HR risk analyst specializing in employee retention and mobility. "
                        "Analyze the following employee report: {report}\n\n"
                        "And the full employee answers/responses: {all_answers}\n\n"
                        "And the following web search results on retention, mobility, and career trends: {search_results}\n\n"
                        "Provide a comprehensive analysis tailored to the employee's profile, answers, and genius factors ({genius_factors}) "
                        "for {company_name}. Ensure scores vary based on tenure, role, performance, genius factors, and insights from the all_answers (e.g., employee satisfaction, career aspirations, feedback on company culture).\n\n"
                        "1. SCORES (0-100 scale, must reflect report specifics and all_answers):\n"
                        "   - Genius Factor Alignment Score: Higher (80-100) if {genius_factors} align with {company_name}'s needs (e.g., innovation for tech, leadership for management) and answers show strong personal fit. Lower (50-70) for generic or misaligned factors, or if answers indicate dissatisfaction.\n"
                        "   - Retention Risk Score: Higher (50-80) for low tenure (<3 years), poor performance, role misalignment, or negative sentiments in answers (e.g., intent to leave). Lower (10-40) for high tenure (>5 years), strong fit, or positive feedback in answers.\n"
                        "   - Mobility Opportunity Score: Higher (80-100) if {genius_factors} and role/answers suggest internal growth opportunities (e.g., expressed interest in advancement). Lower (50-70) if limited by company structure or answers show stagnation.\n\n"
                        "2. TRENDS ANALYSIS (specific to employee, answers, and {company_name}):\n"
                        "   - Retention trends in the industry/{company_name}\n"
                        "   - Internal mobility patterns and innovations\n"
                        "   - Risk factors specific to this employee's profile and answers\n\n"
                        "3. RECOMMENDATIONS (3-5 actionable, employee-specific items):\n"
                        "   - Retention strategies tailored to employee and answers\n"
                        "   - Mobility enhancement opportunities based on expressed interests\n"
                        "   - Risk mitigation actions\n\n"
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
                    input_variables=["report", "all_answers", "search_results", "genius_factors", "company_name"]
                )

                try:
                    # Format the prompt with actual data
                    formatted_prompt = await analysis_prompt.ainvoke({
                        "report": json.dumps(state["report"], indent=2),
                        "all_answers": json.dumps(state["all_answers"], indent=2),
                        "search_results": json.dumps(state["search_results"], indent=2),
                        "genius_factors": ', '.join(state["genius_factors"]),
                        "company_name": state["company"]
                    })
                    
                    # Debug: Log the exact prompt sent to the LLM
                    logger.debug(f"LLM Prompt: {formatted_prompt.text}")
                    
                    # Get response from LLM with structured output
                    response = await llm.ainvoke(formatted_prompt, response_format={"type": "json_object"})
                    
                    content = response.content.strip()
                    
                    # Debug: Log the raw LLM response
                    logger.debug(f"LLM Response: {content}")
                    
                    # Parse the JSON response
                    analysis_data = json.loads(content)
                    
                    # Use LLM-generated scores directly (no rule-based override, as LLM now considers all_answers fully)
                    scores = analysis_data.get("scores", {
                        "genius_factor_score": 50,
                        "retention_risk_score": 50,
                        "mobility_opportunity_score": 50
                    })
                    
                    # Optional: Light deterministic variability from fingerprint for subtle tuning, but preserve LLM intent
                    fp = state["answer_fingerprint"]
                    base_var = (fp % 11) - 5  # -5 to +5 adjustment
                    for key in scores:
                        scores[key] = int(min(100, max(0, scores[key] + base_var)))
                    
                    trends = analysis_data.get("trends", {
                        "retention_trends": "Unable to analyze trends",
                        "mobility_trends": "Unable to analyze mobility trends",
                        "risk_factors": "Unable to identify specific risk factors"
                    })
                    
                    recommendations = analysis_data.get("recommendations", [
                        f"Implement career development for {', '.join(state['genius_factors'])}",
                        f"Create mobility pathways for {state['company']}",
                        "Enhance mentorship programs"
                    ])
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    logger.error(f"Raw content that failed to parse: {content}")
                    scores = {
                        "genius_factor_score": 50,
                        "retention_risk_score": 50,
                        "mobility_opportunity_score": 50
                    }
                    trends = {
                        "retention_trends": "Error in trend analysis",
                        "mobility_trends": "Error in mobility analysis",
                        "risk_factors": "Analysis error occurred"
                    }
                    recommendations = [f"Conduct manual analysis for {state['company']} due to system error"]
                except Exception as e:
                    logger.error(f"Error in analyze_node: {e}")
                    scores = {
                        "genius_factor_score": 50,
                        "retention_risk_score": 50,
                        "mobility_opportunity_score": 50
                    }
                    trends = {
                        "retention_trends": f"Analysis error: {str(e)}",
                        "mobility_trends": f"Analysis error: {str(e)}",
                        "risk_factors": f"Analysis error: {str(e)}"
                    }
                    recommendations = [f"Review manually for {state['company']} due to analysis error"]

                analysis_summary = "Comprehensive risk analysis completed with trends and recommendations."
                return {
                    "analysis": analysis_summary,
                    "scores": scores,
                    "trends": trends,
                    "recommendations": recommendations
                }

            # Build the LangGraph with parallel workflow
            graph = StateGraph(State)
            graph.add_node("retention_search", retention_search_node)
            graph.add_node("mobility_search", mobility_search_node)
            # graph.add_node("career_search", career_search_node)
            graph.add_node("analyze", analyze_node)
            
            # Parallel execution: all search nodes run concurrently
            graph.add_edge("retention_search", "analyze")
            graph.add_edge("mobility_search", "analyze")
            # graph.add_edge("career_search", "analyze")
            graph.add_edge("analyze", END)
            
            # Set parallel entry points
            graph.set_entry_point("retention_search")
            graph.set_entry_point("mobility_search")
            # graph.set_entry_point("career_search")
            
            app = graph.compile()

            # Invoke the graph
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