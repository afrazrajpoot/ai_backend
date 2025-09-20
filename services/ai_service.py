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

        # Updated queries to retrieve ALL data containing the genius factor names
        queries = [
            f'"{primary_name}" OR "{secondary_name}"',  # Get all documents containing either genius factor name
            f'"{primary_name}" OR "{secondary_name}"',  # Same for industry mapping
            f'"{primary_name}" OR "{secondary_name}"',  # Same for framework analysis
            f'"{primary_name}" OR "{secondary_name}"'   # Same for retention research
        ]

        # Map queries to source files (basenames)
        source_files = [
            "(68 Questions) Genius Factor Assessment for Fortune 1000 HR Departments.pdf",
            "Genius Factor to Fortune 1000 Industry Mapping.pdf",
            "Genius Factor Framework Analysis.pdf",
            "retention & internal mobility research_findings.pdf"
        ]

        query_source_pairs = list(zip(queries, source_files))

        # Retrieve ALL relevant documents from each source using the corresponding query and metadata filter
        all_docs = []
        for query, source_basename in query_source_pairs:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1000, "filter": {"source_file": source_basename}}  # High k to get all possible docs
            )
            docs = await retriever.aget_relevant_documents(query)
            if docs:
                all_docs.extend(docs)
                print(f"Retrieved {len(docs)} documents from {source_basename}")

        # Remove duplicates based on content and source to avoid repetition
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hash(doc.page_content[:500] + doc.metadata.get('source_file', ''))
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        # Format retrieved data as plain text
        result_text = f"Genius Factors Identified:\nPrimary: {primary_name} ({primary_letter})\nSecondary: {secondary_name} ({secondary_letter})\n\n"
        result_text += f"Total Documents Retrieved: {len(unique_docs)}\n\nResponses:\n{all_responses}\n\nRetrieved Information:\n"
        
        for i, doc in enumerate(unique_docs, 1):
            content = doc.page_content.replace('\n', ' ').strip()
            source = doc.metadata.get('source_file', 'unknown')
            page = doc.metadata.get('page', 0)
            result_text += f"=== Document {i} ===\nSource: {source} (Page {page})\nContent: {content}\n\n"

        return result_text
    
    @classmethod
    async def generate_career_recommendation(cls, analysis_result: str, all_answers: Any) -> Dict[str, Any]:
        try:
            # Check if analysis_result is empty or invalid
            if not analysis_result or not isinstance(analysis_result, str):
                return {"status": "error", "message": "Invalid or empty analysis results provided"}

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
                if not system_prompt:
                    raise ValueError("System prompt not found in JSON file")

            # Initialize Pydantic output parser
            parser = PydanticOutputParser(pydantic_object=IndividualEmployeeReport)

            # Prompt template with format instructions
            report_prompt = PromptTemplate(
                template=system_prompt + "\n\nAnalysis Data:\n{analysis_data}\n\n{format_instructions}\nGenerate the report:",
                input_variables=["analysis_data"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            # Log the analysis data
            logger.debug(f"Analysis data: {analysis_result}")

            # Render and log the full prompt
            full_prompt_str = report_prompt.format(analysis_data=analysis_result)
            print(f"Full prompt:\n{full_prompt_str}")
        
            chain = report_prompt | llm | parser
            output = await chain.ainvoke({"analysis_data": analysis_result})
            risk_analysis = await cls._perform_risk_analysis(output.dict(), all_answers)
         
            return {
                "status": "success",
                "report": output.dict(),
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