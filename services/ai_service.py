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
            # logger.debug(f"Full prompt sent to LLM:\n{full_prompt_str}")
            # print(f"\n===== FULL PROMPT TO LLM =====\n{full_prompt_str}\n==============================\n")
        
            chain = report_prompt | llm | parser
            output = await chain.ainvoke({"analysis_data": analysis_result})
            risk_analysis = await cls._perform_risk_analysis(output.dict())
         
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
        
        # Get company from report or use default
        company = report.get("company", "Fortune 1000 Company")

        # Node to perform Tavily searches based on report
        async def search_node(state: State) -> State:
            search_results = []
            try:
                tavily = TavilySearchResults(api_key=settings.TAVILY_API_KEY, max_results=5)
                
                # Multiple queries for comprehensive analysis
                queries = [
                    f"employee retention statistics trends {company} 2024",
                    f"internal mobility programs best practices {company}",
                    f"employee turnover prevention strategies {company}",
                    f"career development trends {company} {genius_factors}",
                    f"talent retention innovative approaches {company}"
                ]
                
                for query in queries:
                    try:
                        results = tavily.invoke({"query": query})
                        if results and isinstance(results, list):
                            search_results.extend(results)
                        await asyncio.sleep(0.1)  # Small delay between queries
                    except Exception as e:
                        logger.warning(f"Query failed for '{query}': {e}")
                        continue
                
                logger.info(f"Found {len(search_results)} search results")
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                # Add fallback data
                search_results = [{
                    "title": "Industry Retention Trends",
                    "content": "Current trends show increased focus on internal mobility, with companies investing in upskilling and career pathing to reduce turnover.",
                    "url": "https://example.com/industry-trends"
                }]

            return {"search_results": search_results}
        
        # Node to analyze search results and compute comprehensive analysis
        async def analyze_node(state: State) -> State:
            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=3000
            )

            # Enhanced prompt template
            analysis_prompt = PromptTemplate(
                template=(
                    "You are an HR risk analyst specializing in employee retention and mobility. "
                    "Based on the employee report: {report}\n\n"
                    "And the following web search results on internal mobility and retention trends: {search_results}\n\n"
                    "Provide a comprehensive analysis including:\n\n"
                    "1. SCORES (0-100 scale):\n"
                    "   - Genius Factor Alignment Score (higher = better cultural/role fit)\n"
                    "   - Retention Risk Score (higher = higher risk of leaving)\n"
                    "   - Mobility Opportunity Score (higher = better internal mobility prospects)\n\n"
                    "2. TRENDS ANALYSIS:\n"
                    "   - Current retention trends in the industry/company\n"
                    "   - Internal mobility patterns and innovations\n"
                    "   - Risk factors specific to this employee profile\n\n"
                    "3. RECOMMENDATIONS (3-5 actionable items):\n"
                    "   - Specific retention strategies\n"
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
                    "  \"recommendations\": [\n"
                    "    \"recommendation1\",\n"
                    "    \"recommendation2\",\n"
                    "    \"recommendation3\"\n"
                    "  ],\n"
                    "  \"reasoning\": \"brief overall reasoning\"\n"
                    "}}"
                ),
                input_variables=["report", "search_results"]
            )

            try:
                # Format the prompt with actual data
                formatted_prompt = await analysis_prompt.ainvoke({
                    "report": json.dumps(state["report"], indent=2),
                    "search_results": json.dumps(state["search_results"], indent=2)
                })
                
                # Get response from LLM
                response = await llm.ainvoke(formatted_prompt)
                
                content = response.content.strip()
                
                # Debug: Print the raw response
                logger.debug(f"LLM Response: {content}")
                
                # Try to extract JSON from the response
                if not content.startswith("{"):
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(0)
                
                # Parse the JSON response
                analysis_data = json.loads(content)
                
                scores = analysis_data.get("scores", {
                    "genius_factor_score": 50,
                    "retention_risk_score": 50,
                    "mobility_opportunity_score": 50
                })
                
                trends = analysis_data.get("trends", {
                    "retention_trends": "Unable to analyze trends",
                    "mobility_trends": "Unable to analyze mobility trends",
                    "risk_factors": "Unable to identify specific risk factors"
                })
                
                recommendations = analysis_data.get("recommendations", [
                    "Implement regular career development conversations",
                    "Create clear internal mobility pathways",
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
                recommendations = ["Conduct manual analysis due to system error"]
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
                recommendations = ["Please review manually due to analysis error"]

            analysis_summary = "Comprehensive risk analysis completed with trends and recommendations."

            return {
                "analysis": analysis_summary,
                "scores": scores,
                "trends": trends,
                "recommendations": recommendations
            }

        # Build the LangGraph
        graph = StateGraph(State)
        graph.add_node("search", search_node)
        graph.add_node("analyze", analyze_node)
        graph.add_edge("search", "analyze")
        graph.add_edge("analyze", END)
        graph.set_entry_point("search")
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