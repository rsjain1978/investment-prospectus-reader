# Investment Prospectus Reader

An intelligent system that automates the analysis of investment prospectuses and financial documents using advanced AI techniques. This tool helps investors and analysts extract key information, analyze risks, and make informed investment decisions.

## Features

- **Automated Document Processing**: Extract text from PDF prospectuses and financial documents
- **Key Information Extraction**: Automatically identify and extract:
  - Investment objectives and strategies
  - Risk factors and considerations
  - Fee structures and expenses
  - Historical performance data
  - Management team information
  - Portfolio composition
- **Smart Analysis**:
  - Risk assessment and scoring
  - Investment strategy classification
  - Fee comparison with industry standards
  - Performance metrics analysis
- **Report Generation**: Create concise summaries and detailed analysis reports
- **Data Export**: Export findings in various formats (CSV, JSON, PDF)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/investment-prospectus-reader.git
cd investment-prospectus-reader
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your API keys and settings in `.env`:
```
OPENAI_API_KEY=your_api_key_here
OCR_API_KEY=your_ocr_api_key_here
```

## Usage

### Basic Usage

```python
from prospectus_reader import ProspectusAnalyzer

# Initialize the analyzer
analyzer = ProspectusAnalyzer()

# Analyze a prospectus
results = analyzer.analyze_document("path/to/prospectus.pdf")

# Generate summary report
analyzer.generate_report(results, "output_report.pdf")
```

### Advanced Features

```python
# Custom analysis with specific focus areas
results = analyzer.analyze_document(
    "path/to/prospectus.pdf",
    focus_areas=["risks", "fees", "performance"],
    detail_level="detailed"
)

# Comparative analysis
comparison = analyzer.compare_prospectuses([
    "fund1_prospectus.pdf",
    "fund2_prospectus.pdf"
])
```

## Project Structure

```
investment-prospectus-reader/
├── src/
│   ├── analyzer/
│   │   ├── document_processor.py
│   │   ├── information_extractor.py
│   │   └── risk_analyzer.py
│   ├── models/
│   │   └── ml_models.py
│   └── utils/
│       ├── pdf_utils.py
│       └── text_processing.py
├── tests/
├── examples/
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT models
- Tesseract OCR for document processing
- Financial industry experts for domain knowledge

## Contact

For questions and support, please open an issue in the GitHub repository or contact the maintainers directly.
