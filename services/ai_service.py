import json
import os
import re
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
                logger.info("Loading existing FAISS index...")
                cls._vector_store = FAISS.load_local(
                    faiss_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Vector store loaded from existing index")
                return cls._vector_store
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Rebuilding...")

        logger.info("Creating new FAISS index from PDFs...")
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
                logger.info(f"Processing PDF {i}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                for chunk in chunks:
                    chunk.metadata['source_file'] = os.path.basename(pdf_file)
                all_chunks.extend(chunks)
                logger.info(f"  ✅ Added {len(chunks)} chunks from {os.path.basename(pdf_file)}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
                continue

        if not all_chunks:
            raise ValueError("No documents were loaded from PDF files")

        logger.info(f"Creating embeddings for {len(all_chunks)} total chunks...")
        try:
            cls._vector_store = FAISS.from_documents(all_chunks, embeddings)
            cls._vector_store.save_local(faiss_path)
            current_hash = cls._get_pdf_files_hash(pdf_files)
            with open(f"{faiss_path}.hash", 'w') as f:
                f.write(current_hash)
            logger.info("✅ Vector store created and saved successfully")
            logger.info(f"📊 Total chunks indexed: {len(all_chunks)}")
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
    @classmethod
    async def analyze_majority_answers(cls, majority_results: List[Dict[str, Any]]) -> str:
        vector_store = cls.initialize_vector_store()
        genius_factors = []
        all_responses = ""
        factor_counts_part_iv = {}
        all_response_letters = set()

        # Collect responses and identify genius factors from all parts
        for part in majority_results:
            part_name = part['part']
            responses = part.get('majorityOptions', [])
            all_responses += f"Part: {part_name}\nResponses: {', '.join(responses)}\nMax Count: {part.get('maxCount', 0)}\n\n"
            # Track responses from Part IV specifically
            if part_name == "Part IV: Genius Factor Mapping Assessment":
                for response in responses:
                    factor_counts_part_iv[response] = factor_counts_part_iv.get(response, 0) + part.get('maxCount', 0)
            # Collect all response letters across parts for secondary factor fallback
            all_response_letters.update(responses)

        # Determine primary and secondary genius factors (letters)
        sorted_factors = sorted(factor_counts_part_iv.items(), key=lambda x: x[1], reverse=True)
        if not sorted_factors:
            # If no Part IV responses, use the most frequent letter from other parts
            all_counts = {}
            for part in majority_results:
                for response in part.get('majorityOptions', []):
                    all_counts[response] = all_counts.get(response, 0) + part.get('maxCount', 0)
            sorted_all = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
            primary_letter = sorted_all[0][0] if sorted_all else 'A'  # Default to 'A' if no responses
            secondary_letter = sorted_all[1][0] if len(sorted_all) > 1 else 'A'
        else:
            primary_letter = sorted_factors[0][0]
            # Select secondary letter: prefer Part IV, then other parts, then default
            if len(sorted_factors) > 1:
                secondary_letter = sorted_factors[1][0]
            else:
                # Look for other responses in Part IV or other parts
                other_letters = all_response_letters - {primary_letter}
                secondary_letter = next(iter(other_letters), 'A')  # Default to 'A' if no other letters

        # Map letters to full genius names
        def get_genius_name(letter):
            mapping = {
                'A': "Tech Genius",
                'B': "Social Genius",
                'C': "Visual Genius",
                'D': "Word Genius",
                'E': "Athletic Genius",
                'F': "Number Genius",
                'G': "Eco Genius",
                'H': "Word Genius (Communication Focus)",
                'I': "Spiritual Genius"
            }
            return mapping.get(letter, "Unknown Genius")

        primary_name = get_genius_name(primary_letter)
        secondary_name = get_genius_name(secondary_letter)
        genius_factors = [primary_name, secondary_name]

        # Define queries tailored to each PDF source
        queries = [
            f"Genius factor definitions and characteristics for {primary_name} and {secondary_name}",
            f"Primary and secondary industries for genius factors {', '.join(genius_factors)}",
            f"Content from Genius Factor Framework Analysis of {', '.join(genius_factors)}",
            f"Get all data Employee Mobility and Retention Research Findings for {', '.join(genius_factors)}"
        ]

        # Map queries to source files (basenames)
        source_files = [
            "(68 Questions) Genius Factor Assessment for Fortune 1000 HR Departments.pdf",
            "Genius Factor to Fortune 1000 Industry Mapping.pdf",
            "Genius Factor Framework Analysis.pdf",
            "retention & internal mobility research_findings.pdf"
        ]

        query_source_pairs = list(zip(queries, source_files))

        # Retrieve top 1 document from each source using the corresponding query and metadata filter
        all_docs = []
        for query, source_basename in query_source_pairs:
            logger.info(f"Searching in {source_basename} for: {query}")
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1, "filter": {"source_file": source_basename}}
            )
            docs = await retriever.aget_relevant_documents(query)
            if docs:
                all_docs.append(docs[0])

        # Format retrieved data as plain text
        result_text = f"Genius Factors Identified:\nPrimary: {primary_name} ({primary_letter})\nSecondary: {secondary_name} ({secondary_letter})\n\nResponses:\n{all_responses}\nRetrieved Information:\n"
        for i, doc in enumerate(all_docs, 1):
            content = doc.page_content.replace('\n', ' ').strip()[:1500]
            source = doc.metadata.get('source_file', 'unknown')
            page = doc.metadata.get('page', 0)
            result_text += f"Document {i} - Source: {source} (Page {page})\nContent: {content}\n\n"

        logger.info(f"Retrieved {len(all_docs)} documents (one from each source)")
        return result_text
    @classmethod
    async def generate_career_recommendation(cls, analysis_result: str) -> Dict[str, Any]:
        try:
            # Check if analysis_result is empty or invalid
            if not analysis_result or not isinstance(analysis_result, str):
                return {"status": "error", "message": "Invalid or empty analysis results provided"}

            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o",
                temperature=0.3,
                max_tokens=3000
            )

            # Load system prompt
            prompt_file_path = Path(__file__).parent.parent / "utils" / "prompts.json"
            with open(prompt_file_path, 'r') as file:
                prompt_data = json.load(file)
                system_prompt = prompt_data.get('system_prompt', '')
                if not system_prompt:
                    raise ValueError("System prompt not found in JSON file")

            # Add key naming reinforcement to the system prompt
            key_naming_instruction = """
            
CRITICAL KEY NAMING REQUIREMENTS FOR GRAPHQL COMPATIBILITY:
- In the "internal_career_opportunities" object, the timeline field MUST use the exact key name: "progress_transition_timeline"
- Do NOT use "transition_timeline". Only "progress_transition_timeline" is valid.
- For timeline keys, use ONLY these exact formats: "six_months", "one_year", "two_years" (NOT "6_months", "1_year", "2_years")
- For career pathway keys, use underscores instead of spaces: "Development_Track", "Security_Track", "AI_Track"
- NEVER start any key with a number (e.g., "6_months" is INVALID, use "six_months")
- Replace all spaces in keys with underscores
- Remove all special characters from keys except underscores

VALID EXAMPLE:
"internal_career_opportunities": {
    "progress_transition_timeline": {
        "six_months": "Complete training in advanced UX design principles and tools.",
        "one_year": "Take on lead design projects to build portfolio and experience.",
        "two_years": "Pursue leadership roles in design or product management."
    },
    "career_pathways": {
        "Development_Track": "Engineer → Senior → Lead → CTO",
        "Security_Track": "Analyst → Engineer → Architect → CISO"
    }
}

INVALID EXAMPLE (DO NOT USE):
"internal_career_opportunities": {
    "transition_timeline": {
        "6_months": "Invalid - starts with number",
        "1_year": "Invalid - starts with number"
    }
}
            """

            # Initialize Pydantic output parser
            parser = PydanticOutputParser(pydantic_object=IndividualEmployeeReport)

            # Enhanced prompt template with key naming instructions
            report_prompt = PromptTemplate(
                template=system_prompt + key_naming_instruction + "\n\nAnalysis Data:\n{analysis_data}\n\n{format_instructions}\nGenerate the report:",
                input_variables=["analysis_data"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            # Log the analysis data
            logger.debug(f"Analysis data: {analysis_result}")

            # Generate the report
            chain = report_prompt | llm
            raw_output = await chain.ainvoke({"analysis_data": analysis_result})
            
            # PRE-PROCESSING: Fix the raw text before Pydantic parsing
            raw_text = raw_output.content if hasattr(raw_output, 'content') else str(raw_output)
            
            # Fix the problematic keys in the raw JSON string - COMPREHENSIVE FIXES
            key_fixes = [
                # Fix transition_timeline -> progress_transition_timeline
                (r'"transition_timeline":', '"progress_transition_timeline":'),
                # Fix numeric timeline keys - COMPREHENSIVE
                (r'"6_months":', '"six_months":'),
                (r'"6_month":', '"six_months":'),
                (r'"1_year":', '"one_year":'),
                (r'"1_years":', '"one_year":'),
                (r'"2_years":', '"two_years":'),
                (r'"2_year":', '"two_years":'),
                (r'"3_years":', '"three_years":'),
                (r'"3_year":', '"three_years":'),
                (r'"4_years":', '"four_years":'),
                (r'"4_year":', '"four_years":'),
                (r'"5_years":', '"five_years":'),
                (r'"5_year":', '"five_years":'),
                (r'"7_years":', '"seven_years":'),
                (r'"7_year":', '"seven_years":'),
                (r'"8_years":', '"eight_years":'),
                (r'"8_year":', '"eight_years":'),
                (r'"9_years":', '"nine_years":'),
                (r'"9_year":', '"nine_years":'),
                (r'"10_years":', '"ten_years":'),
                (r'"10_year":', '"ten_years":'),
                # Fix career pathway keys with spaces
                (r'"Development Track":', '"Development_Track":'),
                (r'"Security Track":', '"Security_Track":'),
                (r'"AI Track":', '"AI_Track":'),
                (r'"Data Track":', '"Data_Track":'),
                (r'"Leadership Track":', '"Leadership_Track":'),
                (r'"Tech Track":', '"Tech_Track":'),
                (r'"Product Track":', '"Product_Track":'),
                (r'"Design Track":', '"Design_Track":'),
                (r'"Marketing Track":', '"Marketing_Track":'),
                (r'"Engineering Track":', '"Engineering_Track":'),
                # Fix any remaining problematic characters
                (r'"short_term":', '"short_term":'),
                (r'"long_term":', '"long_term":'),
            ]
            
            # Apply all the fixes
            fixed_text = raw_text
            fixes_applied = []
            for old_pattern, new_pattern in key_fixes:
                if old_pattern in fixed_text:
                    fixed_text = fixed_text.replace(old_pattern, new_pattern)
                    fixes_applied.append(f"{old_pattern} -> {new_pattern}")
            
            if fixes_applied:
                logger.info(f"Pre-processing fixes applied: {fixes_applied}")
            else:
                logger.debug("No pre-processing fixes needed")
            
            # Now parse with Pydantic using the fixed text
            try:
                import json
                # Parse as JSON first to validate
                json_data = json.loads(fixed_text)
                # Then create the Pydantic object from the dict
                output = IndividualEmployeeReport.parse_obj(json_data)
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing failed after pre-processing: {je}")
                # Fallback to original parsing
                chain_with_parser = report_prompt | llm | parser
                output = await chain_with_parser.ainvoke({"analysis_data": analysis_result})
            except Exception as pe:
                logger.error(f"Pydantic parsing failed: {pe}")
                # Fallback to original parsing
                chain_with_parser = report_prompt | llm | parser
                output = await chain_with_parser.ainvoke({"analysis_data": analysis_result})
            
            # POST-PROCESSING: Additional safety checks on the parsed object
            output_dict = output.dict()
            
            # Fix internal_career_opportunities key naming
            internal_career = output_dict.get("internal_career_opportunities", {})
            
            # 1. Remap transition_timeline -> progress_transition_timeline
            if "transition_timeline" in internal_career and "progress_transition_timeline" not in internal_career:
                internal_career["progress_transition_timeline"] = internal_career.pop("transition_timeline")
                logger.warning("Remapped 'transition_timeline' -> 'progress_transition_timeline'")
            
            # 2. Fix numeric timeline keys (6_months -> six_months, etc.) - COMPREHENSIVE
            progress_timeline = internal_career.get("progress_transition_timeline", {})
            if progress_timeline:
                key_mappings = {
                    "6_months": "six_months",
                    "6_month": "six_months",
                    "1_year": "one_year", 
                    "1_years": "one_year",
                    "2_years": "two_years",
                    "2_year": "two_years",
                    "3_years": "three_years",
                    "3_year": "three_years",
                    "4_years": "four_years",
                    "4_year": "four_years",
                    "5_years": "five_years",
                    "5_year": "five_years",
                    "7_years": "seven_years",
                    "7_year": "seven_years",
                    "8_years": "eight_years",
                    "8_year": "eight_years",
                    "9_years": "nine_years",
                    "9_year": "nine_years",
                    "10_years": "ten_years",
                    "10_year": "ten_years",
                }
                
                for old_key, new_key in key_mappings.items():
                    if old_key in progress_timeline:
                        progress_timeline[new_key] = progress_timeline.pop(old_key)
                        logger.warning(f"Remapped timeline key: '{old_key}' -> '{new_key}'")
            
            # 3. Fix career pathway keys (spaces -> underscores)
            career_pathways = internal_career.get("career_pathways", {})
            if career_pathways:
                pathway_mappings = {
                    "Development Track": "Development_Track",
                    "Security Track": "Security_Track", 
                    "AI Track": "AI_Track",
                    "Data Track": "Data_Track",
                    "Leadership Track": "Leadership_Track"
                }
                
                for old_key, new_key in pathway_mappings.items():
                    if old_key in career_pathways:
                        career_pathways[new_key] = career_pathways.pop(old_key)
                        logger.warning(f"Remapped pathway key: '{old_key}' -> '{new_key}'")
            
            # 4. Apply similar fixes to other sections that might have problematic keys
            def fix_dict_keys(data, section_name=""):
                """Recursively fix dictionary keys to be GraphQL/Prisma compatible"""
                if isinstance(data, dict):
                    fixed_data = {}
                    for key, value in data.items():
                        # Replace spaces with underscores
                        fixed_key = str(key).replace(" ", "_")
                        # Replace other problematic characters
                        fixed_key = re.sub(r'[^a-zA-Z0-9_]', '_', fixed_key)
                        # Remove consecutive underscores
                        fixed_key = re.sub(r'_+', '_', fixed_key).strip('_')
                        
                        if fixed_key != key:
                            logger.debug(f"Fixed key in {section_name}: '{key}' -> '{fixed_key}'")
                        
                        fixed_data[fixed_key] = fix_dict_keys(value, section_name)
                    return fixed_data
                elif isinstance(data, list):
                    return [fix_dict_keys(item, section_name) for item in data]
                else:
                    return data
            
            # Apply key fixes to all sections
            output_dict = fix_dict_keys(output_dict, "report")
            
            # Generate risk analysis
            risk_analysis = await cls._perform_risk_analysis(output_dict)

            # Final canonicalization logging for internal_career_opportunities
            try:
                ico = output_dict.get('internal_career_opportunities', {})
                if isinstance(ico, dict):
                    timeline = ico.get('progress_transition_timeline') or ico.get('transition_timeline')
                    if timeline and isinstance(timeline, dict):
                        logger.info(f"Return canonicalization check - timeline keys: {list(timeline.keys())}")
                        if 'transition_timeline' in ico:
                            logger.warning("Return stage: unexpected 'transition_timeline' present")
            except Exception as log_e:
                logger.debug(f"Canonicalization logging skipped: {log_e}")
         
            return {
                "status": "success",
                "report": output_dict,
                "risk_analysis": risk_analysis,
                "metadata": {
                    "processingTimestamp": "06:47 PM PKT, August 29, 2025",
                    "modelUsed": "gpt-4o-mini",
                    "dataSourcesUsed": [
                        "(68 Questions) Genius Factor Assessment for Fortune 1000 HR Departments.pdf",
                        "Genius Factor Framework Analysis.pdf",
                        "Genius Factor to Fortune 1000 Industry Mapping.pdf",
                        "retention & internal mobility research_findings.pdf"
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_career_recommendation: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    @classmethod
    async def _perform_risk_analysis(cls, report: Dict[str, Any]) -> Dict[str, Any]:
        # Define state for LangGraph
        class State(TypedDict):
            report: Dict[str, Any]
            search_results: Annotated[List[Dict[str, Any]], operator.add]
            analysis: str
            scores: Dict[str, Any]
            trends: Dict[str, Any]
            recommendations: List[str]
            genius_factors: List[str]
            company: str

        # Extract genius factors and company from report
        genius_factors = []
        if "primary_genius_factor" in report:
            genius_factors.append(report["primary_genius_factor"])
        if "secondary_genius_factor" in report:
            genius_factors.append(report["secondary_genius_factor"])
        if not genius_factors:
            genius_factors = ["General Talent"]
            logger.warning("No genius factors in report, using default: General Talent")

        # Get company from report or use default
        company = report.get("company", "Fortune 1000 Company")

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
                logger.info(f"Retention search found {len(search_results)} results")
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
                logger.info(f"Mobility search found {len(search_results)} results")
            except Exception as e:
                logger.error(f"Mobility search error: {e}")
                search_results = [{
                    "title": f"Mobility Trends for {state['company']}",
                    "content": f"Generic mobility trends for {state['company']} emphasize upskilling and career pathing.",
                    "url": "https://example.com/mobility-trends"
                }]
            logger.debug(f"Mobility search results: {json.dumps(search_results, indent=2)}")
            return {"search_results": search_results}

        # Node for career development search
      
        # Node to analyze search results and compute comprehensive analysis
        async def analyze_node(state: State) -> State:
            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.5,  # Increased for variability
                max_tokens=3000
            )

            # Enhanced prompt for score variability
            analysis_prompt = PromptTemplate(
                template=(
                    "You are an HR risk analyst specializing in employee retention and mobility. "
                    "Analyze the following employee report: {report}\n\n"
                    "And the following web search results on retention, mobility, and career trends: {search_results}\n\n"
                    "Provide a comprehensive analysis tailored to the employee's profile and genius factors ({genius_factors}) "
                    "for {company_name}. Ensure scores vary based on tenure, role, performance, and genius factors.\n\n"
                    "1. SCORES (0-100 scale, must reflect report specifics):\n"
                    "   - Genius Factor Alignment Score: Higher (80-100) if {genius_factors} align with {company_name}'s needs (e.g., innovation for tech, leadership for management). Lower (50-70) for generic or misaligned factors.\n"
                    "   - Retention Risk Score: Higher (50-80) for low tenure (<3 years), poor performance, or role misalignment. Lower (10-40) for high tenure (>5 years) or strong fit.\n"
                    "   - Mobility Opportunity Score: Higher (80-100) if {genius_factors} and role suggest internal growth opportunities. Lower (50-70) if limited by company structure.\n\n"
                    "2. TRENDS ANALYSIS (specific to employee and {company_name}):\n"
                    "   - Retention trends in the industry/{company_name}\n"
                    "   - Internal mobility patterns and innovations\n"
                    "   - Risk factors specific to this employee's profile\n\n"
                    "3. RECOMMENDATIONS (3-5 actionable, employee-specific items):\n"
                    "   - Retention strategies tailored to employee\n"
                    "   - Mobility enhancement opportunities\n"
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
                input_variables=["report", "search_results", "genius_factors", "company_name"]
            )

            try:
                # Format the prompt with actual data
                formatted_prompt = await analysis_prompt.ainvoke({
                    "report": json.dumps(state["report"], indent=2),
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
                
                # Rule-based score adjustment for variability
                scores = analysis_data.get("scores", {
                    "genius_factor_score": 50,
                    "retention_risk_score": 50,
                    "mobility_opportunity_score": 50
                })
                
                # Adjust scores based on report specifics
                if "tenure" in state["report"]:
                    tenure = state["report"]["tenure"]
                    scores["retention_risk_score"] = min(100, max(0, scores["retention_risk_score"] + (5 - tenure) * 10))
                if "role" in state["report"] and any(factor.lower() in state["report"]["role"].lower() for factor in state["genius_factors"]):
                    scores["genius_factor_score"] = min(100, scores["genius_factor_score"] + 10)
                if len(state["genius_factors"]) > 1 or "career development" in json.dumps(state["search_results"]).lower():
                    scores["mobility_opportunity_score"] = min(100, scores["mobility_opportunity_score"] + 10)
                
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
            "search_results": [],
            "analysis": "",
            "scores": {},
            "trends": {},
            "recommendations": [],
            "genius_factors": genius_factors,
            "company": company
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