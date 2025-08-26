import json
import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import pickle
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.logger import logger
from config import settings
from utils.models import IndividualEmployeeReport

# Existing AIService Class with Updates
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
    async def analyze_majority_answers(cls, majority_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        vector_store = cls.initialize_vector_store()
        genius_factors = []
        all_responses = {}
        
        for part in majority_results:
            part_name = part['part']
            responses = part['majorityOptions']
            all_responses[part_name] = {'responses': responses, 'maxCount': part.get('maxCount', 0)}
            if "Genius Factor Mapping" in part_name:
                genius_factors.extend(responses)
        
        if not genius_factors:
            return {
                "status": "success",
                "message": "No Genius Factors identified in assessment",
                "allResponses": all_responses
            }
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 20, "lambda_mult": 0.7}
        )
        
        queries = [
            f"Genius Factors career recommendations industry mapping: {', '.join(genius_factors)}",
            "retention internal mobility career development strategies",
            "genius factor assessment scoring interpretation guidelines",
            f"hybrid genius factor combinations involving {', '.join(genius_factors)}",
            f"career pathways and roles for {', '.join(genius_factors)}",
            "cross-industry opportunities for genius factors",
            "employee retention strategies aligned with genius factors",
            f"detailed characteristics and skills for {', '.join(genius_factors)}"
        ]
        
        all_docs = []
        for query in queries:
            logger.info(f"Searching for: {query}")
            docs = await retriever.aget_relevant_documents(query)
            all_docs.extend(docs)
        
        seen_content = set()
        unique_docs = [doc for doc in all_docs if doc.page_content.replace('\n', ' ').strip() not in seen_content and len(doc.page_content.strip()) >= 50]
        
        results_by_source = {}
        career_recommendations = []
        industry_mappings = []
        assessment_guidelines = []
        genius_factor_details = []
        retention_insights = []
        
        for doc in unique_docs:
            content = doc.page_content.replace('\n', ' ').strip()
            source = doc.metadata.get('source_file', os.path.basename(doc.metadata.get('source', 'unknown')))
            page = doc.metadata.get('page', 0)
            
            doc_info = {"content": content[:1500], "source": source, "page": page, "chunk_id": hashlib.md5(content.encode()).hexdigest()[:8]}
            if source not in results_by_source:
                results_by_source[source] = []
            results_by_source[source].append(doc_info)
            
            content_lower = content.lower()
            if any(factor.lower() in content_lower for factor in genius_factors):
                if "career" in content_lower or "role" in content_lower or "job" in content_lower:
                    career_recommendations.append(doc_info)
                elif "fortune" in content_lower or "industry" in content_lower or "mapping" in content_lower:
                    industry_mappings.append(doc_info)
                else:
                    genius_factor_details.append(doc_info)
            if "assessment" in content_lower or "scoring" in content_lower:
                assessment_guidelines.append(doc_info)
            if "retention" in content_lower or "mobility" in content_lower:
                retention_insights.append(doc_info)
        
        total_documents = len(seen_content)
        sources_coverage = list(results_by_source.keys())
        
        logger.info(f"📊 Retrieved {total_documents} unique chunks from {len(sources_coverage)} sources")
        
        return {
            "status": "success",
            "assessmentData": {
                "identifiedFactors": genius_factors,
                "allResponses": all_responses,
                "totalDocumentsRetrieved": total_documents,
                "sourcesCovered": sources_coverage
            },
            "retrievedData": {
                "bySource": results_by_source,
                "careerRecommendations": career_recommendations,
                "industryMappings": industry_mappings,
                "assessmentGuidelines": assessment_guidelines,
                "geniusFactorDetails": genius_factor_details,
                "retentionInsights": retention_insights
            },
            "summary": {
                "primaryFactors": genius_factors,
                "documentsPerSource": {source: len(docs) for source, docs in results_by_source.items()},
                "totalRelevantChunks": total_documents,
                "dataCompleteness": "comprehensive" if total_documents >= 10 else "partial"
            },
            "vectorStoreStats": cls.get_vector_store_stats()
        }

    @classmethod
    async def generate_career_recommendation(cls, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if analysis_result.get('status') != 'success':
                return {"status": "error", "message": "Invalid analysis results provided"}

            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=3000
            )

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

            chain = report_prompt | llm | parser

            data_str = json.dumps(analysis_result, indent=2)
            logger.debug(f"Analysis data: {data_str}")

            # Generate the report
            output = await chain.ainvoke({"analysis_data": data_str})

            return {
                "status": "success",
                "report": output.dict(),
                "metadata": {
                    "processingTimestamp": "07:03 PM PKT, August 25, 2025",
                    "modelUsed": "Azure Chat Model",
                    "dataSourcesUsed": analysis_result.get('assessmentData', {}).get('sourcesCovered', [])
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_career_recommendation: {str(e)}")
            return {"status": "error", "error": str(e)}