# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict, Optional
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_openai import AzureOpenAIEmbeddings
# import os
# from dotenv import load_dotenv

# app = FastAPI()

# # Load environment variables
# load_dotenv()
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
# azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
# azure_api_version = os.getenv("AZURE_API_VERSION")

# if not all([azure_endpoint, azure_api_key, azure_deployment_name, azure_api_version]):
#     raise ValueError("Missing Azure OpenAI environment variables in .env file")

# # Pydantic model for assessment data
# class OptionCounts(BaseModel):
#     A: Optional[int] = 0
#     B: Optional[int] = 0
#     C: Optional[int] = 0
#     D: Optional[int] = 0
#     E: Optional[int] = 0
#     F: Optional[int] = 0
#     G: Optional[int] = 0
#     H: Optional[int] = 0
#     I: Optional[int] = 0

# class AssessmentPart(BaseModel):
#     part: str
#     optionCounts: OptionCounts

# class AssessmentData(BaseModel):
#     data: List[AssessmentPart]

# # Initialize FAISS vector store from PDFs
# def initialize_vector_store():
#     pdf_files = [
#         "./pdfs/document1.pdf",
#         "./pdfs/document2.pdf",
#         "./pdfs/document3.pdf",
#         "./pdfs/document4.pdf",
#         "./pdfs/document5.pdf"
#     ]
    
#     # Load and process PDFs
#     all_chunks = []
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\n\n", "\n", ".", " ", ""]
#     )
    
#     for pdf_file in pdf_files:
#         if not os.path.exists(pdf_file):
#             raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        
#         loader = PyPDFLoader(pdf_file)
#         documents = loader.load()
#         chunks = text_splitter.split_documents(documents)
#         all_chunks.extend(chunks)
    
#     # Create embeddings and FAISS store
#     embeddings = AzureOpenAIEmbeddings(
#         azure_endpoint=azure_endpoint,
#         api_key=azure_api_key,
#         azure_deployment=azure_deployment_name,
#         openai_api_version=azure_api_version
#     )
#     vector_store = FAISS.from_documents(all_chunks, embeddings)
#     vector_store.save_local("faiss_index")
    
#     return vector_store

# # Analyze assessment data
# def analyze_assessment_data(data: List[AssessmentPart]):
#     results = []
#     for part_data in data:
#         option_counts = part_data.optionCounts.dict(exclude_none=True)
#         if not option_counts:
#             result = {"part": part_data.part, "majorityOptions": None, "maxCount": 0}
#             results.append(result)
#             continue
        
#         # Find maximum count and majority options
#         max_count = max(option_counts.values(), default=0)
#         majority_options = [
#             option for option, count in option_counts.items() if count == max_count and count > 0
#         ]
#         majority_options.sort()  # Sort for consistent display
        
#         result = {
#             "part": part_data.part,
#             "majorityOptions": majority_options if majority_options else None,
#             "maxCount": max_count
#         }
#         results.append(result)
    
#     return results

# @app.post("/analyze-assessment")
# async def analyze_assessment(assessment_data: AssessmentData):
#     try:
#         # Initialize FAISS vector store (runs once to process PDFs)
#         vector_store = initialize_vector_store()
        
#         # Analyze the provided assessment data
#         results = analyze_assessment_data(assessment_data.data)
        
#         return {
#             "status": "success",
#             "results": results
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)