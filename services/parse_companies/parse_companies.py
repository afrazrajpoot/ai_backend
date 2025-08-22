# Updated services/parse_companies/parse_companies.py
from typing import Dict
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import asyncio
from fastapi import UploadFile
import json

# For parsing ODT files
from odf.opendocument import load
from odf.text import P, H
from odf import teletype

# For parsing ODS spreadsheets
from odf.opendocument import load as load_ods
from odf.table import Table, TableRow, TableCell


async def parse_file(file: UploadFile) -> str:
    """
    Parse an uploaded file and return its text.
    Supports Excel (.xls, .xlsx), ODT (.odt), and ODS (.ods) files.
    """
    filename = file.filename
    if not filename:
        raise ValueError("Uploaded file has no filename.")

    await file.seek(0)

    # Excel files
    if filename.endswith(('.xls', '.xlsx')):
        content = await file.read()
        # Wrap bytes with BytesIO so pandas can read it
        from io import BytesIO
        df = pd.read_excel(BytesIO(content))
        if df.empty:
            return ""
        text = " ".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))
        return text

    # ODT files
    elif filename.endswith('.odt'):
        await file.seek(0)
        try:
            doc = load(file.file)
            text_elements = []

            paragraphs = doc.getElementsByType(P)
            for p in paragraphs:
                try:
                    text_content = teletype.extractText(p)
                    if text_content.strip():
                        text_elements.append(text_content.strip())
                except:
                    text_content = "".join(
                        node.data for node in p.childNodes if hasattr(node, "data")
                    )
                    if text_content.strip():
                        text_elements.append(text_content.strip())

            headings = doc.getElementsByType(H)
            for h in headings:
                try:
                    text_content = teletype.extractText(h)
                    if text_content.strip():
                        text_elements.append(text_content.strip())
                except:
                    pass

            return "\n".join(text_elements)
        except Exception as e:
            raise ValueError(f"Error parsing ODT file: {str(e)}")

    # ODS (spreadsheet) files
    elif filename.endswith('.ods'):
        await file.seek(0)
        try:
            doc = load_ods(file.file)
            text_elements = []

            tables = doc.getElementsByType(Table)
            for table in tables:
                for row in table.getElementsByType(TableRow):
                    row_data = []
                    for cell in row.getElementsByType(TableCell):
                        cell_text = teletype.extractText(cell)
                        if cell_text:
                            row_data.append(cell_text.strip())
                    if row_data:
                        text_elements.append(" ".join(row_data))

            return "\n".join(text_elements)
        except Exception as e:
            raise ValueError(f"Error parsing ODS file: {str(e)}")

    else:
        raise ValueError(f"Unsupported file type: {filename}")


async def process_text_with_langchain(text: str) -> Dict:
    """
    Convert text into JSON using LangChain + OpenAI.
    """
    prompt = PromptTemplate(
        input_variables=["content"],
        template=(
            "You are an AI that converts tabular or textual data into JSON. "
            "Extract meaningful information from the following text and output ONLY valid JSON format (no markdown, no code blocks):\n\n{content}"
        )
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    result = await asyncio.to_thread(
        lambda: llm.invoke(prompt.format(content=text))
    )
    
    content = result.content.strip()
    
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    
    content = content.strip()
    
    try:
        parsed_json = json.loads(content)
        return parsed_json
    except json.JSONDecodeError as e:
        return {
            "raw_response": content,
            "parse_error": str(e),
            "extracted_text": text[:500]
        }
