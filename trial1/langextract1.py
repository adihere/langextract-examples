# LangExtract example: extract fund information from PDF files

import os
import json
import sys
import textwrap
import configparser
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import PyPDF2
import langextract as lx
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_pdf(pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> str:
    """
    Read text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        start_page: First page to read (0-based index)
        end_page: Last page to read (0-based index, None for all pages)
    
    Returns:
        Extracted text from the PDF
    """
    try:
        # Print path information for debugging
        logger.info(f"Current Working Directory: {Path.cwd()}")
        logger.info(f"Script Location: {Path(__file__).parent.resolve()}")
        logger.info(f"Resolved PDF path: {Path(pdf_path).resolve()}")
        
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            reader = PyPDF2.PdfReader(file)
            
            # Validate page range
            total_pages = len(reader.pages)
            if end_page is None:
                end_page = total_pages
            else:
                end_page = min(end_page, total_pages)
            
            start_page = max(0, min(start_page, total_pages - 1))
            
            # Extract text from specified pages
            text_content = []
            for page_num in range(start_page, end_page):
                page = reader.pages[page_num]
                text_content.append(page.extract_text())
            
            return "\n".join(text_content)
    
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        sys.exit(1)

def setup_api_key():
    """Setup the API key for LangExtract"""
    api_key = os.getenv("LANGEXTRACT_API_KEY")
    if not api_key:
        # Try loading from .env file if it exists
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith('LANGEXTRACT_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        os.environ["LANGEXTRACT_API_KEY"] = api_key
                        break
    
    # If still no API key, check GEMINI_API_KEY
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            os.environ["LANGEXTRACT_API_KEY"] = api_key
    
    if not api_key:
        print("Error: No API key found. Please set LANGEXTRACT_API_KEY in your environment or .env file")
        print("You can get an API key from: https://cloud.google.com/vertex-ai/docs/generative-ai/access-api")
        sys.exit(1)

# Setup API key before proceeding
setup_api_key()


# Load configuration from JSON file
def load_config(config_path: str = 'config.json') -> Dict:
    """
    Load extraction configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing extraction configuration
    """
    try:
        config_path = Path(__file__).parent / config_path
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['extraction_config']
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        sys.exit(1)

# Load configuration
config = load_config()
prompt = config['prompt']
model_id = config['model_id']
max_chunk_size = config['max_chunk_size']
max_workers = config.get('max_workers', 1)  # Default to 2 workers
extraction_passes = config.get('extraction_passes', 1)  # Default to 1 pass
chunk_overlap = config.get('chunk_overlap', 200)  # Default to 200 characters overlap

# Set environment variables to control LangExtract parallelism
os.environ['LANGEXTRACT_MAX_WORKERS'] = str(max_workers)

# Convert JSON examples to LangExtract ExampleData objects
examples = [
    lx.data.ExampleData(
        text=ex['text'],
        extractions=[
            lx.data.Extraction(
                extraction_class=ext['extraction_class'],
                extraction_text=ext['extraction_text'],
                attributes=ext['attributes']
            )
            for ext in ex['extractions']
        ]
    )
    for ex in config['examples']
]

# 2) Read and process PDF file
def process_pdf(pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> Optional[lx.data.AnnotatedDocument]:
    """
    Process a PDF file and extract entities.
    
    Args:
        pdf_path: Path to the PDF file
        start_page: First page to process (0-based index)
        end_page: Last page to process (0-based index, None for all pages)
    
    Returns:
        Extraction results or None if processing failed
    """
    try:
        # Read PDF content
        print(f"Reading PDF file: {pdf_path}")
        text_content = read_pdf(pdf_path, start_page, end_page)
        
        if not text_content.strip():
            print("Warning: No text content extracted from PDF")
            return None
            
        def create_overlapping_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
            """Create chunks of text with overlap to maintain context."""
            chunks = []
            start = 0
            text_length = len(text)
            
            while start < text_length:
                end = start + chunk_size
                if end > text_length:
                    end = text_length
                
                # Add the chunk
                chunks.append(text[start:end])
                
                # Move start position, accounting for overlap
                start = end - overlap if end < text_length else text_length
                
            return chunks

        # Create overlapping chunks for better context
        text_chunks = create_overlapping_chunks(text_content, max_chunk_size, chunk_overlap)
        
        # Process chunks sequentially to avoid parallel requests
        all_results = []
        seen_extractions = set()  # Track unique extractions
        
        for i, chunk in enumerate(text_chunks, 1):
            logger.info(f"Processing chunk {i} of {len(text_chunks)}...")
            
            # Make specified number of passes over each chunk
            chunk_results = []
            for pass_num in range(extraction_passes):
                try:
                    result = lx.extract(
                        text_or_documents=chunk,
                        prompt_description=prompt,
                        examples=examples,
                        model_id=model_id,
                    )
                    
                    # Filter out duplicate extractions
                    for extraction in result.extractions:
                        # Create a unique key for each extraction
                        extraction_key = (
                            extraction.extraction_class,
                            extraction.extraction_text
                        )
                        
                        if extraction_key not in seen_extractions:
                            chunk_results.append(extraction)
                            seen_extractions.add(extraction_key)
                    
                except Exception as e:
                    logger.error(f"Error in pass {pass_num + 1} for chunk {i}: {str(e)}")
                    continue
            
            all_results.extend(chunk_results)
        
        # Combine results into a single AnnotatedDocument
        if all_results:
            final_result = lx.data.AnnotatedDocument(
                text=text_content,
                extractions=all_results
            )
            return final_result
        return None
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

def load_settings(settings_file: str = 'settings.ini') -> Dict[str, Any]:
    """
    Load settings from the INI file.
    
    Args:
        settings_file: Path to the settings INI file
        
    Returns:
        Dictionary containing all settings
    """
    try:
        config = configparser.ConfigParser()
        settings_path = Path(__file__).parent / settings_file
        if not settings_path.exists():
            logger.error(f"Settings file not found: {settings_path}")
            sys.exit(1)
            
        config.read(settings_path)
        
        # Get settings with defaults and resolve paths
        script_dir = Path(__file__).parent.resolve()
        pdf_path = config.get('DEFAULT', 'pdf_path', fallback='input.pdf')
        
        # Resolve paths relative to script directory if they're not absolute
        if not Path(pdf_path).is_absolute():
            pdf_path = str(script_dir / pdf_path)
            
        settings = {
            'pdf_path': pdf_path,
            'start_page': config.getint('PROCESSING', 'start_page', fallback=0),
            'end_page': config.getint('PROCESSING', 'end_page', fallback=-1),
            'jsonl_output': str(script_dir / config.get('OUTPUT', 'jsonl_output', fallback='fund_extractions.jsonl')),
            'visualization_output': str(script_dir / config.get('OUTPUT', 'visualization_output', fallback='fund_visualization.html'))
        }
        
        # Log path information
        logger.info("Path Information:")
        logger.info(f"Script Directory: {script_dir}")
        logger.info(f"PDF Path: {settings['pdf_path']}")
        logger.info(f"JSONL Output: {settings['jsonl_output']}")
        logger.info(f"Visualization Output: {settings['visualization_output']}")
        
        # Convert -1 end_page to None for processing all pages
        if settings['end_page'] == -1:
            settings['end_page'] = None
            
        return settings
        
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        sys.exit(1)

# Load settings
settings = load_settings()
result = process_pdf(settings['pdf_path'], settings['start_page'], settings['end_page'])

def format_fund_info(result: Optional[lx.data.AnnotatedDocument]) -> Dict[str, Any]:
    """
    Format the extraction results into a structured fund information dictionary.
    
    Args:
        result: The extraction results from LangExtract
        
    Returns:
        Dictionary containing formatted fund information
    """
    if not result or not result.extractions:
        return {}
        
    fund_info = {
        'isin': None,
        'asset_class': None,
        'expense_ratio': None,
        'issuer': None,
        'attributes': {}
    }
    
    for extraction in result.extractions:
        if extraction.extraction_class in fund_info:
            fund_info[extraction.extraction_class] = extraction.extraction_text
            if extraction.attributes:
                fund_info['attributes'][extraction.extraction_class] = extraction.attributes
    
    return fund_info

def print_fund_info(fund_info: Dict[str, Any]) -> None:
    """Print formatted fund information."""
    if not fund_info:
        logger.warning("No fund information to display")
        return
        
    print("\nExtracted Fund Information:")
    print("-" * 50)
    print(f"ISIN: {fund_info.get('isin', 'Not found')}")
    print(f"Asset Class: {fund_info.get('asset_class', 'Not found')}")
    print(f"Total Expense Ratio: {fund_info.get('expense_ratio', 'Not found')}")
    print(f"Issuer: {fund_info.get('issuer', 'Not found')}")
    
    if fund_info.get('attributes'):
        print("\nAdditional Information:")
        for field, attrs in fund_info['attributes'].items():
            for key, value in attrs.items():
                print(f"- {field} {key}: {value}")

# Save results in JSONL format
def save_fund_info(fund_info: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """
    Save fund information to JSONL file.
    
    Args:
        fund_info: The fund information to save
        settings: Application settings containing output paths
    """
    try:
        output_file = settings['jsonl_output']
        with open(output_file, 'a') as f:
            json.dump(fund_info, f)
            f.write('\n')
        logger.info(f"Saved fund information to {output_file}")
        
        # Generate visualization
        html_content = lx.visualize(output_file)
        content = html_content.data if hasattr(html_content, "data") else str(html_content)
        
        # Ensure UTF-8 charset
        if "<meta charset=" not in content:
            if "<head>" in content:
                content = content.replace("<head>", '<meta charset="utf-8">', 1)
            else:
                content = '<meta charset="utf-8">\n' + content
        
        # Save visualization
        viz_path = Path(settings['visualization_output'])
        viz_path.write_text(content, encoding="utf-8")
        logger.info(f"Saved visualization to {viz_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
