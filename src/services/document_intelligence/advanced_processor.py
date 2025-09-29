"""Advanced document processing with OCR, table extraction, and intelligent parsing."""

import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import base64
from datetime import datetime

# Document processing
import fitz  # PyMuPDF for better PDF handling
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pdfplumber import PDF as PDFPlumber
import camelot
from tabulate import tabulate

# Advanced text processing
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    PythonCodeTextSplitter,
    LatexTextSplitter
)
from transformers import pipeline
import spacy
import yake  # Keyword extraction

# Structured data extraction
import re
import dateutil.parser
from email_validator import validate_email
import phonenumbers

from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import cache_manager
from src.services.langchain.gemini_service import gemini_service

logger = LoggerAdapter(__name__)


class DocumentIntelligence:
    """Advanced document processing with AI-powered extraction."""
    
    def __init__(self):
        """Initialize document intelligence services."""
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize keyword extractor
        self.keyword_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # Max n-grams
            dedupLim=0.7,
            top=20
        )
        
        # Initialize specialized splitters
        self.splitters = {
            "markdown": MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            ),
            "html": HTMLHeaderTextSplitter(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                ]
            ),
            "code": PythonCodeTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            "latex": LatexTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            "default": RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=True
            )
        }
    
    async def process_document_advanced(
        self,
        file_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
        perform_ocr: bool = True,
        extract_metadata: bool = True,
        extract_entities: bool = True,
        extract_keywords: bool = True,
        smart_chunking: bool = True,
        chunk_size: int = 1000
    ) -> Dict[str, Any]:
        """Process document with advanced extraction capabilities."""
        try:
            file_type = Path(file_path).suffix.lower()
            result = {
                "file_info": self._get_file_info(file_path),
                "content": "",
                "chunks": [],
                "metadata": {},
                "tables": [],
                "images": [],
                "entities": {},
                "keywords": [],
                "summary": "",
                "structure": {}
            }
            
            # Process based on file type
            if file_type in [".pdf", ".PDF"]:
                result = await self._process_pdf_advanced(
                    file_path, result, extract_images, extract_tables, perform_ocr
                )
            elif file_type in [".docx", ".doc"]:
                result = await self._process_word_advanced(file_path, result)
            elif file_type in [".xlsx", ".xls"]:
                result = await self._process_excel_advanced(file_path, result)
            elif file_type in [".pptx", ".ppt"]:
                result = await self._process_powerpoint_advanced(file_path, result)
            elif file_type in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
                result = await self._process_image_advanced(file_path, result, perform_ocr)
            elif file_type in [".html", ".htm"]:
                result = await self._process_html_advanced(file_path, result)
            elif file_type in [".md", ".markdown"]:
                result = await self._process_markdown_advanced(file_path, result)
            elif file_type in [".py", ".js", ".java", ".cpp", ".cs", ".go", ".rs"]:
                result = await self._process_code_advanced(file_path, result)
            else:
                result = await self._process_text_advanced(file_path, result)
            
            # Extract metadata
            if extract_metadata:
                result["metadata"] = await self._extract_metadata(result["content"])
            
            # Extract entities
            if extract_entities:
                result["entities"] = await self._extract_entities(result["content"])
            
            # Extract keywords
            if extract_keywords:
                result["keywords"] = self._extract_keywords(result["content"])
            
            # Smart chunking
            if smart_chunking:
                result["chunks"] = await self._smart_chunk(
                    result["content"], file_type, chunk_size
                )
            
            # Generate summary
            result["summary"] = await self._generate_summary(result["content"])
            
            # Analyze document structure
            result["structure"] = self._analyze_structure(result["content"])
            
            logger.info(f"Advanced processing complete: {Path(file_path).name}")
            return result
            
        except Exception as e:
            logger.error(f"Advanced processing failed: {e}")
            raise
    
    async def _process_pdf_advanced(
        self,
        file_path: str,
        result: Dict,
        extract_images: bool,
        extract_tables: bool,
        perform_ocr: bool
    ) -> Dict:
        """Advanced PDF processing with OCR and extraction."""
        try:
            # Use PyMuPDF for better PDF handling
            pdf_document = fitz.open(file_path)
            text_content = []
            
            for page_num, page in enumerate(pdf_document, 1):
                # Extract text
                page_text = page.get_text()
                
                # If no text found and OCR is enabled, perform OCR
                if not page_text.strip() and perform_ocr:
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                    img_data = pix.tobytes("png")
                    
                    # Perform OCR
                    img = Image.open(io.BytesIO(img_data))
                    page_text = pytesseract.image_to_string(img, lang='eng')
                
                text_content.append(f"[Page {page_num}]\n{page_text}")
                
                # Extract images if requested
                if extract_images:
                    for img_index, img in enumerate(page.get_images()):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        result["images"].append({
                            "page": page_num,
                            "index": img_index,
                            "format": image_ext,
                            "data": base64.b64encode(image_bytes).decode(),
                            "size": len(image_bytes)
                        })
            
            result["content"] = "\n\n".join(text_content)
            
            # Extract tables using camelot
            if extract_tables:
                try:
                    tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
                    for i, table in enumerate(tables):
                        result["tables"].append({
                            "index": i,
                            "page": table.page,
                            "data": table.df.to_dict('records'),
                            "html": table.df.to_html(),
                            "accuracy": table.accuracy
                        })
                except Exception as e:
                    logger.warning(f"Table extraction with camelot failed: {e}")
                    # Fallback to pdfplumber
                    with PDFPlumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages, 1):
                            tables = page.extract_tables()
                            for table_index, table in enumerate(tables):
                                if table:
                                    result["tables"].append({
                                        "index": table_index,
                                        "page": page_num,
                                        "data": table,
                                        "html": tabulate(table, tablefmt="html")
                                    })
            
            # Extract PDF metadata
            result["metadata"]["pdf_info"] = {
                "pages": len(pdf_document),
                "author": pdf_document.metadata.get("author", ""),
                "title": pdf_document.metadata.get("title", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "keywords": pdf_document.metadata.get("keywords", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "creation_date": str(pdf_document.metadata.get("creationDate", "")),
                "modification_date": str(pdf_document.metadata.get("modDate", ""))
            }
            
            pdf_document.close()
            return result
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    async def _process_image_advanced(
        self,
        file_path: str,
        result: Dict,
        perform_ocr: bool
    ) -> Dict:
        """Process image with OCR and enhancement."""
        try:
            img = Image.open(file_path)
            
            # Store image data
            with open(file_path, "rb") as f:
                img_bytes = f.read()
                result["images"].append({
                    "index": 0,
                    "format": img.format,
                    "data": base64.b64encode(img_bytes).decode(),
                    "size": len(img_bytes),
                    "dimensions": f"{img.width}x{img.height}"
                })
            
            if perform_ocr:
                # Enhance image for better OCR
                img_cv = cv2.imread(file_path)
                
                # Convert to grayscale
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Apply thresholding to preprocess the image
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # Perform OCR with multiple languages support
                try:
                    text = pytesseract.image_to_string(thresh, lang='eng+fra+deu+spa')
                    result["content"] = text
                    
                    # Also get OCR data with confidence scores
                    ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
                    
                    # Filter for high confidence text
                    high_conf_text = []
                    for i, conf in enumerate(ocr_data['conf']):
                        if int(conf) > 60:  # Confidence threshold
                            text = ocr_data['text'][i]
                            if text.strip():
                                high_conf_text.append(text)
                    
                    result["metadata"]["ocr_confidence"] = {
                        "average": np.mean([int(c) for c in ocr_data['conf'] if int(c) > 0]),
                        "high_confidence_text": " ".join(high_conf_text)
                    }
                    
                except Exception as e:
                    logger.error(f"OCR failed: {e}")
                    result["content"] = ""
            
            # Use Gemini Vision for image analysis
            if gemini_service:
                try:
                    analysis = await gemini_service.analyze_image(
                        img_bytes,
                        "Describe this image in detail. What text, objects, or information can you see?"
                    )
                    result["metadata"]["ai_analysis"] = analysis
                except Exception as e:
                    logger.warning(f"Gemini Vision analysis failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
    
    async def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract structured metadata from content."""
        metadata = {
            "dates": [],
            "emails": [],
            "phone_numbers": [],
            "urls": [],
            "currencies": [],
            "percentages": [],
            "addresses": [],
            "social_security": [],
            "credit_cards": [],
            "ips": []
        }
        
        # Date extraction
        date_pattern = r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w+ \d{1,2}, \d{4})\b'
        for match in re.finditer(date_pattern, content):
            try:
                parsed_date = dateutil.parser.parse(match.group(), fuzzy=True)
                metadata["dates"].append({
                    "text": match.group(),
                    "parsed": parsed_date.isoformat(),
                    "position": match.start()
                })
            except:
                pass
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, content):
            try:
                valid = validate_email(match.group())
                metadata["emails"].append({
                    "email": valid.email,
                    "position": match.start()
                })
            except:
                pass
        
        # Phone number extraction
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}'
        for match in re.finditer(phone_pattern, content):
            try:
                parsed = phonenumbers.parse(match.group(), None)
                if phonenumbers.is_valid_number(parsed):
                    metadata["phone_numbers"].append({
                        "text": match.group(),
                        "formatted": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                        "position": match.start()
                    })
            except:
                pass
        
        # URL extraction
        url_pattern = r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/[^\s]*)?'
        metadata["urls"] = [match.group() for match in re.finditer(url_pattern, content)]
        
        # Currency extraction
        currency_pattern = r'[$€£¥₹]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|INR)'
        metadata["currencies"] = [match.group() for match in re.finditer(currency_pattern, content)]
        
        # Percentage extraction
        percentage_pattern = r'\d+(?:\.\d+)?%'
        metadata["percentages"] = [match.group() for match in re.finditer(percentage_pattern, content)]
        
        # IP Address extraction
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        metadata["ips"] = [match.group() for match in re.finditer(ip_pattern, content)]
        
        # SSN detection (masked)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, content):
            metadata["social_security"] = ["[DETECTED - MASKED FOR SECURITY]"]
        
        # Credit card detection (masked)
        cc_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        if re.search(cc_pattern, content):
            metadata["credit_cards"] = ["[DETECTED - MASKED FOR SECURITY]"]
        
        return metadata
    
    async def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities using NLP."""
        doc = self.nlp(content[:1000000])  # Limit to 1M chars for performance
        
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "money": [],
            "products": [],
            "events": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["persons"].append(ent.text)
            elif ent.label_ in ["ORG", "COMPANY"]:
                entities["organizations"].append(ent.text)
            elif ent.label_ in ["LOC", "GPE"]:
                entities["locations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["money"].append(ent.text)
            elif ent.label_ == "PRODUCT":
                entities["products"].append(ent.text)
            elif ent.label_ == "EVENT":
                entities["events"].append(ent.text)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_keywords(self, content: str) -> List[Dict[str, Any]]:
        """Extract keywords and key phrases."""
        keywords = self.keyword_extractor.extract_keywords(content)
        
        return [
            {"keyword": kw[0], "score": kw[1]}
            for kw in keywords[:20]  # Top 20 keywords
        ]
    
    async def _smart_chunk(
        self,
        content: str,
        file_type: str,
        chunk_size: int
    ) -> List[Dict[str, Any]]:
        """Smart content chunking based on document type."""
        chunks = []
        
        # Select appropriate splitter
        if file_type in [".md", ".markdown"]:
            splitter = self.splitters["markdown"]
        elif file_type in [".html", ".htm"]:
            splitter = self.splitters["html"]
        elif file_type in [".py", ".js", ".java", ".cpp", ".cs", ".go", ".rs"]:
            splitter = self.splitters["code"]
        elif file_type in [".tex", ".latex"]:
            splitter = self.splitters["latex"]
        else:
            splitter = self.splitters["default"]
        
        # Split content
        if hasattr(splitter, 'split_text'):
            split_docs = splitter.split_text(content)
        else:
            split_docs = splitter.split_text(content)
        
        # Create chunks with metadata
        for i, chunk in enumerate(split_docs):
            chunk_content = chunk.page_content if hasattr(chunk, 'page_content') else chunk
            chunk_metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            
            # Extract keywords for each chunk
            chunk_keywords = self.keyword_extractor.extract_keywords(chunk_content)
            
            chunks.append({
                "index": i,
                "content": chunk_content,
                "metadata": chunk_metadata,
                "size": len(chunk_content),
                "keywords": [kw[0] for kw in chunk_keywords[:5]],
                "summary": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content
            })
        
        return chunks
    
    async def _generate_summary(self, content: str) -> str:
        """Generate AI-powered summary."""
        try:
            # Limit content for summarization
            content_preview = content[:5000]
            
            prompt = f"""
            Provide a concise summary of the following document content:
            
            {content_preview}
            
            Summary should include:
            1. Main topic/purpose
            2. Key points
            3. Important findings or conclusions
            
            Keep the summary under 200 words.
            """
            
            summary = await gemini_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return summary
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            # Fallback to simple extraction
            sentences = content.split('. ')[:5]
            return '. '.join(sentences)
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure."""
        lines = content.split('\n')
        
        structure = {
            "total_lines": len(lines),
            "total_words": len(content.split()),
            "total_characters": len(content),
            "paragraphs": len([l for l in lines if l.strip()]),
            "empty_lines": len([l for l in lines if not l.strip()]),
            "average_line_length": np.mean([len(l) for l in lines if l.strip()]) if lines else 0,
            "sections": [],
            "has_headers": False,
            "has_lists": False,
            "has_tables": False,
            "has_code": False
        }
        
        # Detect headers (lines that are likely titles)
        for line in lines:
            if line.strip() and len(line.strip()) < 100:
                if line.isupper() or line.strip().endswith(':'):
                    structure["sections"].append(line.strip())
                    structure["has_headers"] = True
        
        # Detect lists
        list_patterns = [r'^\s*[-*•]\s+', r'^\s*\d+\.\s+', r'^\s*[a-z]\)\s+']
        for pattern in list_patterns:
            if re.search(pattern, content, re.MULTILINE):
                structure["has_lists"] = True
                break
        
        # Detect tables (simple heuristic)
        if '|' in content and content.count('|') > 10:
            structure["has_tables"] = True
        
        # Detect code blocks
        code_patterns = [r'```', r'def\s+\w+\(', r'function\s+\w+\(', r'class\s+\w+']
        for pattern in code_patterns:
            if re.search(pattern, content):
                structure["has_code"] = True
                break
        
        return structure
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata."""
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "name": path.name,
            "extension": path.suffix,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "mime_type": mimetypes.guess_type(file_path)[0]
        }


# Global instance
document_intelligence = DocumentIntelligence()