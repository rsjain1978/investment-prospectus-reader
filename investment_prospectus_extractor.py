import os
from typing import List, Dict, Optional
import json
from pathlib import Path
import asyncio
from pdf2image import convert_from_path
import base64
from io import BytesIO
from PIL import Image
from openai import AsyncOpenAI
import logging
from datetime import datetime

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
    def __init__(self, api_key: str):
        """Initialize the extractor with OpenAI API key."""
        self.client = AsyncOpenAI(api_key=api_key)
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_images = 0
        self.extraction_prompt = """
        Please analyze this investment prospectus page and extract the following information in a structured format:
        
        1. Charges and Fees:

        ACDS Preliminary Charges: Fees or costs incurred at the preliminary stage of a financial arrangement.
        Investment Management Charges: Charges for managing the investment or fund on behalf of the investor.
        ACDS Service Charges: Fees associated with services provided under the ACDS framework.
        ACDS Registrar Charges: Costs related to the maintenance of records and administrative tasks by the registrar.
        Minimum Withdrawals: The minimum amount allowed for withdrawal in the financial arrangement.
        
        2. NAV (Net Asset Value) Information:

        Basis of Calculation: The methodology or criteria used to calculate the Net Asset Value.
        Expected Percentage of NAV Per Fund: The anticipated proportion or percentage of the NAV allocated to specific funds.
        
        3. General Risks:

        Risk to Capital Income: The potential loss or reduction of capital or income invested.
        Foreign Currency Risk: The risk arising from fluctuations in foreign currency exchange rates impacting investments.
        
        4. Asset Class-Specific Risks:

        Credit Risk: The risk of default by borrowers or failure to meet financial obligations.
        Sovereign Debt Risk: Risks related to investments in government debt instruments, including the risk of default by a government.
        
        Instructions:
        For each value extracted, specify a detailed passage of text in which it appears so that its easy to refer the source of extracted value. Return the extracted data in the following JSON structure:
        {
            "charges_and_fees": {
                "acds_preliminrary_charges": {"value": "", "passage": ""},
                "investment_management_charges": {"value": "", "passage": ""},
                "acds_service_charges": {"value": "", "passage": ""},
                "acds_registrar_charges": {"value": "", "passage": ""},
                "minimum_withdrawals": {"value": "", "passage": ""}
            },
            "nav_information": {
                "basis_of_calculation": {"value": "", "passage": ""},
                "expected_percentage_of_nav_per_fund": {"value": "", "passage": ""},
            }
            "general_risks": {
                "risk_to_capital_income": {"value": "", "passage": ""},
                "foreign_currency_risk": {"value": "", "passage": ""},
            }
            "asset_class_specific_risks": {
                "credit_risk": {"value": "", "passage": ""},
                "sovereign_debt_risk": {"value": "", "passage": ""},
            }
        }
        """
        
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images with error handling and save to folder."""
        try:
            output_dir = Path('extracted_pages')
            
            # Check if directory exists and contains images
            if output_dir.exists():

                logger.info("Existing images found. Loading images...")

                existing_images = sorted(output_dir.glob('page_*.png'))
                if existing_images:
                    logger.info(f"Found {len(existing_images)} existing images in {output_dir}")
                    # Load existing images
                    images = [Image.open(img_path) for img_path in existing_images]
                    return images
            
            # If no existing images found, create directory and convert PDF
            output_dir.mkdir(exist_ok=True)
            
            # Convert PDF to images
            logger.info("No existing images found. Converting PDF to images...")
            images = convert_from_path(
                pdf_path,
                dpi=300,  # Higher DPI for better quality
                fmt="PNG"
            )
            logger.info(f"Successfully converted PDF to {len(images)} images")
            
            # Save each image
            for i, image in enumerate(images, 1):
                image_path = output_dir / f"page_{i:03d}.png"
                image.save(image_path, "PNG", optimize=True)
                logger.info(f"Saved page {i} to {image_path}")
                
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string with compression."""
        try:
            # Optimize image size before conversion
            buffered = BytesIO()
            image.save(buffered, format="PNG", optimize=True, quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise

    async def extract_information(self, image: Image.Image, page_num: int) -> Dict:
        """Extract information from a single image using OpenAI's API."""
        try:
            base64_image = self.image_to_base64(image)
            self.total_images += 1
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
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
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Track usage directly from response
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                self.total_tokens += total_tokens
                
                logger.info(f"""
                            Page {page_num} Usage:
                            - Prompt tokens: {prompt_tokens}
                            - Completion tokens: {completion_tokens}
                            - Total tokens: {total_tokens}
                            """)
            
            try:
                content = response.choices[0].message.content
                print(f"\nRaw response from OpenAI for page {page_num}:")
               
                # Clean the content string if needed
                content = content.strip()
                if not content.startswith('{'):
                    # Extract JSON if it's wrapped in other text
                    content = content[content.find('{'):content.rfind('}')+1]
                
                extracted_data = json.loads(content)
                
                return {
                    "page_number": page_num,
                    "status": "success",
                    "data": extracted_data,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    } if hasattr(response, 'usage') else None
                }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response for page {page_num}: {str(e)}")
                logger.error(f"Raw content: {content}")
                return {
                    "page_number": page_num,
                    "status": "error",
                    "error": "Invalid JSON response",
                    "raw_content": content,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    } if hasattr(response, 'usage') else None
                }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            return {
                "page_number": page_num,
                "status": "error",
                "error": str(e)
            }

    async def process_prospectus(self, pdf_path: str) -> Dict:
        """Process entire prospectus and return consolidated results."""
        start_time = datetime.now()
        
        try:
            print("\nConverting pages to images:")
            images = self.convert_pdf_to_images(pdf_path)
            tasks = []
            
            print("\nProcessing each page:")
            for i, image in enumerate(images[46:48], start=47):
                task = self.extract_information(image, i)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Consolidate successful extractions and usage statistics
            successful_extractions = [r for r in results if r["status"] == "success"]
            failed_extractions = [r for r in results if r["status"] == "error"]
            
            # Calculate total token usage
            total_usage = {
                "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results if r.get("usage")),
                "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in results if r.get("usage")),
                "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in results if r.get("usage"))
            }
            
            final_output = {
                "metadata": {
                    "document_path": pdf_path,
                    "total_pages": len(images),
                    "successful_extractions": len(successful_extractions),
                    "failed_extractions": len(failed_extractions),
                    "processing_time": str(datetime.now() - start_time),
                    "usage_statistics": {
                        "total_images_processed": self.total_images,
                        "token_usage": total_usage
                    }
                },
                "extracted_data": results
            }
            
            # Print usage summary
            print("\nUsage Statistics:")
            print(f"Total images processed: {self.total_images}")
            print(f"Prompt tokens: {total_usage['prompt_tokens']}")
            print(f"Completion tokens: {total_usage['completion_tokens']}")
            print(f"Total tokens: {total_usage['total_tokens']}")
            
            return final_output
            
        except Exception as e:
            logger.error(f"Error processing prospectus: {str(e)}")
            raise

    def aggregate_data(json_data):
        aggregated = {
            "charges_and_fees": {},
            "nav_information": {},
            "general_risks": {},
            "asset_class_specific_risks": {}
        }

        # Iterate through extracted data
        for item in json_data["extracted_data"]:
            page_number = item["page_number"]
            for key in aggregated.keys():
                if key in item["data"]:
                    for sub_key, sub_value in item["data"][key].items():
                        if sub_value["value"]:  # Check if value is not blank
                            if sub_key not in aggregated[key]:
                                aggregated[key][sub_key] = []
                            aggregated[key][sub_key].append({
                                "page_number": page_number,
                                "value": sub_value["value"],
                                "passage": sub_value["passage"]
                            })

        return aggregated

    def save_results(self, results: Dict, output_path: Optional[str] = None) -> str:
        """Save results to a JSON file with timestamp."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"prospectus_extraction_{timestamp}.json"
            
            aggregated_results = aggregate_data(results)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

async def main():
    # Replace with your OpenAI API key
    api_key = ""
    
    # Replace with your PDF file path
    pdf_path = "../Fidelity Investment Funds III.pdf"
    
    extractor = InvestmentProspectusExtractor(api_key)
    
    try:
        print("\nExtraction started:")
        results = await extractor.process_prospectus(pdf_path)
        output_path = extractor.save_results(results)
        
        # Print summary
        print("\nExtraction Summary:")
        print(f"Total pages processed: {results['metadata']['total_pages']}")
        print(f"Successful extractions: {results['metadata']['successful_extractions']}")
        print(f"Failed extractions: {results['metadata']['failed_extractions']}")
        print(f"Processing time: {results['metadata']['processing_time']}")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 