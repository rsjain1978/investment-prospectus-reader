import os
from typing import List, Dict, Optional
import json
from pathlib import Path
import asyncio
import fitz  # PyMuPDF
from PIL import Image
import base64
from io import BytesIO
from openai import AsyncOpenAI
import logging
from datetime import datetime
import aiohttp
from dotenv import load_dotenv
from extraction_prompt import EXTRACTION_PROMPT
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prospectus_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class InvestmentProspectusExtractor:
    def __init__(self, api_key: str, max_retries: int = 3, timeout: float = 30.0, dpi: int = 200):
        """Initialize the extractor with OpenAI API key and configuration."""
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout
        )
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_images = 0
        self.max_retries = max_retries
        self.timeout = timeout
        self.dpi = dpi
        self.output_dir = Path('extracted_pages')
        self.data_dir = Path('extracted_data')
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.extraction_prompt = EXTRACTION_PROMPT
        self.batch_size = 5  # Set batch size


    def extract_page_as_image(self, pdf_path: str, page_number: int) -> Optional[Path]:
        """Convert a single PDF page to an image using PyMuPDF."""
        try:
            output_path = self.output_dir / f"page_{page_number:03d}.png"
            if output_path.exists():
                logger.info(f"Page {page_number} already exists. Skipping.")
                return output_path

            # Open PDF and render the specific page
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]  # Page numbers are zero-indexed
            pix = page.get_pixmap(dpi=self.dpi)  # Render the page as a pixmap
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image.save(output_path, "PNG", optimize=True)
            logger.info(f"Page {page_number} saved to {output_path}.")
            return output_path
        except Exception as e:
            logger.error(f"Error extracting page {page_number}: {str(e)}")
            return None

    def image_to_base64(self, image_path: Path) -> str:
        """Convert image at `image_path` to base64 string."""
        try:
            with Image.open(image_path) as image:
                # Resize image if too large (max dimension 2000px)
                max_size = 2000
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)

                # Convert to base64
                buffered = BytesIO()
                image.save(buffered, format="JPEG", optimize=True, quality=85)
                return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise

    async def extract_information(self, image_path: Path, page_num: int) -> Dict:
        """Extract information from a single image using OpenAI's API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                base64_image = self.image_to_base64(image_path)
                self.total_images += 1

                # Log the payload (excluding sensitive data)
                logger.info(f"Page {page_num}: Sending request to OpenAI API (Attempt {attempt + 1}/{self.max_retries})")

                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Correct model name
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a JSON-only response bot. Always respond with valid JSON, never include explanations or additional text."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.extraction_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    timeout=self.timeout
                )

                # Log response metadata for debugging
                logger.info(f"Page {page_num}: Received response from OpenAI API (Attempt {attempt + 1})")

                # Parse response content
                try:
                    content = response.choices[0].message.content.strip()
                    if not content.startswith('{'):
                        content = content[content.find('{'):content.rfind('}') + 1]

                    extracted_data = json.loads(content)
                    return {
                        "page_number": page_num,
                        "status": "success",
                        "data": extracted_data
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Page {page_num}: Failed to parse JSON response (Attempt {attempt + 1}): {str(e)}")
                    logger.error(f"Raw response content: {content}")
                    raise

            except aiohttp.ClientError as e:
                logger.error(f"Page {page_num}: Connection error (Attempt {attempt + 1}): {str(e)}")
                logger.error(traceback.format_exc())
            except asyncio.TimeoutError:
                logger.error(f"Page {page_num}: Request timed out (Attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Page {page_num}: Unexpected error (Attempt {attempt + 1}): {str(e)}")
                logger.error(traceback.format_exc())

            # Log retry information
            if attempt < self.max_retries - 1:
                logger.warning(f"Page {page_num}: Retrying (Attempt {attempt + 2}/{self.max_retries})...")
                await asyncio.sleep(2)  # Small delay before retrying

        # Log final failure after all retries
        logger.error(f"Page {page_num}: Failed after {self.max_retries} attempts.")
        return {
            "page_number": page_num,
            "status": "error",
            "error": "Max retries exceeded"
        }

    def save_page_data(self, page_data: Dict) -> None:
        """Save extracted data for a single page to a JSON file."""
        try:
            page_number = page_data["page_number"]
            output_path = self.data_dir / f"page_{page_number:03d}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved extracted data for page {page_number} to {output_path}")
        except Exception as e:
            logger.error(f"Error saving page data: {str(e)}")
            raise

    async def process_prospectus(self, pdf_path: str, start_page: int, end_page: int) -> Dict:
        """Process pages from a PDF asynchronously, in batches."""
        try:
            start_time = datetime.now()
            tasks = []

            # Collect tasks for each page
            for page_number in range(start_page, end_page + 1):
                image_path = self.extract_page_as_image(pdf_path, page_number)

                # Skip pages that failed to extract
                if image_path:
                    task = self.extract_information(image_path, page_number)
                    tasks.append(task)

            # Process tasks in batches
            results = []
            for i in range(0, len(tasks), self.batch_size):
                batch = tasks[i:i + self.batch_size]
                logger.info(f"Processing batch {i // self.batch_size + 1} with {len(batch)} tasks.")
                batch_results = await asyncio.gather(*batch)
                
                # Save each page's data immediately after processing
                for page_data in batch_results:
                    if page_data["status"] == "success":
                        self.save_page_data(page_data)
                
                results.extend(batch_results)

                # Optional: Add a short delay between batches to avoid throttling
                await asyncio.sleep(1)

            # Calculate summary statistics
            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "error"]
            
            summary = {
                "metadata": {
                    "pdf_path": pdf_path,
                    "start_page": start_page,
                    "end_page": end_page,
                    "total_pages_processed": len(results),
                    "successful_extractions": len(successful),
                    "failed_extractions": len(failed),
                    "processing_time": str(datetime.now() - start_time)
                },
                "results_location": str(self.data_dir)
            }
            
            # Save summary to a separate file
            summary_path = self.data_dir / "extraction_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {len(results)} pages in {len(results) // self.batch_size + 1} batches.")
            logger.info(f"Results saved in {self.data_dir}")
            return summary
            
        except Exception as e:
            logger.error(f"Error processing prospectus: {str(e)}")
            raise

    @staticmethod
    def aggregate_all_pages(data_dir: Path) -> Dict:
        """
        Aggregate data from all page JSONs into a consolidated format.
        Returns a dictionary with each entity type and its values found across all pages.
        """
        try:
            consolidated = {
                "charges_and_fees": {},
                "nav_information": {},
                "general_risks": {},
                "asset_class_specific_risks": {}
            }

            # Get all JSON files except the summary
            json_files = [f for f in data_dir.glob("page_*.json")]
            
            for json_file in sorted(json_files):
                with open(json_file, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    
                    if page_data["status"] != "success":
                        continue
                        
                    page_number = page_data["page_number"]
                    extracted_data = page_data["data"]
                    
                    # Process each main category
                    for category in consolidated.keys():
                        if category in extracted_data:
                            # Process each field in the category
                            for field, content in extracted_data[category].items():
                                if not content["value"]:  # Skip empty values
                                    continue
                                    
                                # Initialize field if not exists
                                if field not in consolidated[category]:
                                    consolidated[category][field] = []
                                    
                                # Add the value with its source page
                                consolidated[category][field].append({
                                    "value": content["value"],
                                    "page": page_number,
                                    "passage": content["passage"]
                                })

            # Save consolidated data
            output_path = data_dir / "consolidated_data.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(consolidated, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Consolidated data saved to {output_path}")
            return consolidated
            
        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            raise


async def main():
    # Replace with your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")

    # Replace with your PDF file path
    pdf_path = "./Fidelity Investment Funds III.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    extractor = InvestmentProspectusExtractor(api_key)

    # Get total pages in PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    # Process all pages
    start_page = 40
    end_page = 50

    logger.info(f"Starting extraction for all pages (1 to {end_page})...")
    summary = await extractor.process_prospectus(pdf_path, start_page, end_page)

    # Print final summary
    print("\nExtraction Summary:")
    print(f"Total pages processed: {summary['metadata']['total_pages_processed']}")
    print(f"Successful extractions: {summary['metadata']['successful_extractions']}")
    print(f"Failed extractions: {summary['metadata']['failed_extractions']}")
    print(f"Processing time: {summary['metadata']['processing_time']}")
    print(f"Results saved in: {summary['results_location']}")

    # Aggregate all extracted data
    print("\nAggregating extracted data...")
    consolidated_data = InvestmentProspectusExtractor.aggregate_all_pages(Path(summary['results_location']))
    print(f"Consolidated data saved to: {summary['results_location']}/consolidated_data.json")


if __name__ == "__main__":
    asyncio.run(main())
