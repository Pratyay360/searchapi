# DataOrchestra

A comprehensive data processing and extraction toolkit.

## Features

- Web scraping and crawling
- Document processing (PDF, DOCX, etc.)
- Text cleaning and normalization
- Data extraction and transformation

## Installation

```bash
pip install dataorchestra
```

## Usage

### Basic Usage

```python
from dataorchestra import fetch_url, process_pdf, clean_text

# Fetch a web page
result = fetch_url("https://example.com")
print(f"Fetched {result.output_path}")

# Process a PDF document
result = process_pdf("document.pdf")
print(f"Processed PDF: {result.output_path}")

# Clean text
cleaned_text = clean_text("This is some text with URLs: https://example.com and emails: user@example.com")
print(cleaned_text)
```

### Advanced Usage

```python
from dataorchestra import crawl_website, process_directory

# Crawl a website
results = crawl_website("https://example.com")
print(f"Crawled {len(results)} pages")

# Process a directory of documents
process_directory("documents")
```

## Configuration

DataOrchestra can be configured using environment variables:

```bash
export DATAORCHESTRA_LOG_LEVEL=debug
export DATAORCHESTRA_OUTPUT_DIR=./output
export DATAORCHESTRA_WORKERS=10
```

## License

MIT