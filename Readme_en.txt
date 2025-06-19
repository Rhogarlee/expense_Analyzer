# Smart Expense Analyzer

## Project Overview

The Smart Expense Analyzer is an end-to-end solution combining VLM (Visual Language Models) and LLM (Large Language Models) for automated processing, analysis, and monitoring of corporate expenses. The system can extract structured data from receipt images, perform categorization, anomaly detection, and generate easy-to-understand reports and recommendations.

This `README.md` file will briefly explain the usage of several core modules within the system.

## Core Module Usage Instructions

### 1. VLM Module (Visual Language Model Module)

**File Name:** `vlm_receipt_to_markdown.py`

**Functionality:** Responsible for converting receipt images into structured Markdown format text. The VLM model can identify text, numbers, and their spatial relationships and hierarchical structures within the image, transforming this visual information into machine-readable and format-preserving text.

**Usage:**

```bash
python vlm_receipt_to_markdown.py --image /path/to/receipt.jpg --model qwen2-vl-2b --output receipt.md
```

**Parameter Description:**
- `--image`: Path to the receipt image (required)
- `--model`: VLM model to use (optional: `qwen2-vl-2b`, `minicpm`, `gpt-4o-mini`, `claude-3.5`). Defaults to `qwen2-vl-2b`.
- `--api-key`: API key (required when using commercial models)
- `--output`: Path to the output Markdown file (optional). If not specified, output will be printed to standard output.

### 2. LLM Module (Large Language Model Module)

**File Name:** `llm_markdown_parser.py`

**Functionality:** Receives Markdown formatted receipt data output from the VLM module, and leverages its powerful natural language understanding and generation capabilities to extract key information, perform expense categorization, and initial anomaly detection, finally outputting standardized JSON formatted data.

**Usage:**

```bash
python llm_markdown_parser.py /path/to/receipt.md --model llama3 --output analysis.json
```

**Parameter Description:**
- First argument: Path to the Markdown receipt file (required)
- `--model`: LLM model to use (optional: `llama3`, `mistral`, `gpt-4`, `claude-3`). Defaults to `llama3`.
- `--api-key`: API key (required when using commercial models)
- `--output`: Path to the output JSON analysis result file (optional). If not specified, output will be printed to standard output.

### 3. Batch Processor

**File Name:** `batch_processor.py`

**Functionality:** Coordinates the VLM, LLM, and data integration modules to enable batch processing of large quantities of receipt images or Excel data, significantly improving processing efficiency. It supports a parallel processing mechanism, allowing configuration of the number of parallel workers to effectively utilize multi-core CPU resources.

**Usage:**

```bash
# Process all receipt images in a specified directory
python batch_processor.py --input-dir /path/to/receipts --output-dir /path/to/results --vlm-model qwen2-vl-2b --llm-model llama3

# Import data from an Excel file for processing
python batch_processor.py --input-excel /path/to/expenses.xlsx --output-dir /path/to/results --llm-model llama3
```

**Parameter Description:**
- `--input-dir`: Path to the input directory containing receipt images (mutually exclusive with `--input-excel`)
- `--input-excel`: Path to the Excel file containing expense data (mutually exclusive with `--input-dir`)
- `--output-dir`: Path to the output directory for results (JSON files and summary reports) (required)
- `--vlm-model`: VLM model to use for batch processing (optional, only required when processing images)
- `--llm-model`: LLM model to use for batch processing (optional)
- `--api-key`: API key (required when using commercial models)
- `--workers`: Number of parallel workers (optional, defaults to the number of CPU cores)

### 4. Dataspace Connector

**File Name:** `dataspace_connector.py`

**Functionality:** Provides an interface for connecting with the company Dataspace, supporting various connection methods and authentication mechanisms. This allows the system to not only process new receipts but also integrate historical data for comprehensive analysis. This module is primarily called as part of a backend service or data integration pipeline and does not directly provide a command-line interface.

**Usage:**

This module is typically imported and called by other Python scripts (e.g., `batch_processor.py` or other data integration services). Below is a simplified Python code example demonstrating how to use `dataspace_connector.py`:

```python
from dataspace_connector import DataspaceConnector

# Initialize the connector, may require authentication information
connector = DataspaceConnector(api_endpoint="https://your.dataspace.api", auth_token="your_token")

# Example: Read data from Dataspace
data = connector.read_data(query="SELECT * FROM expenses WHERE date > \'2024-01-01\'")
print("Data read from Dataspace:", data)

# Example: Write data to Dataspace
new_expense = {"date": "2025-06-18", "amount": 100.0, "category": "Office Supplies"}
connector.write_data(table_name="expenses", data=new_expense)
print("Data written to Dataspace")
```

**Key Methods:**
- `__init__(api_endpoint, auth_token, ...)`: Initializes the connector, configuring connection parameters and authentication information.
- `read_data(query, ...)`: Reads data from Dataspace, supporting SQL queries or other query languages.
- `write_data(table_name, data, ...)`: Writes data to the specified table in Dataspace.
- `update_data(table_name, data, condition, ...)`: Updates data in Dataspace.
- `delete_data(table_name, condition, ...)`: Deletes data from Dataspace.

**Notes:**
- Specific `api_endpoint` and `auth_token` need to be configured according to your company Dataspace.
- Ensure your environment has all necessary dependencies installed, such as libraries for database connections or API calls.

## End-to-End Usage (Simplified)

You can chain the VLM and LLM modules to achieve end-to-end receipt processing:

1. **Image to Markdown:**
   ```bash
   python vlm_receipt_to_markdown.py --image your_receipt.jpg --output temp_receipt.md
   ```

2. **Markdown Parsing and Analysis:**
   ```bash
   python llm_markdown_parser.py temp_receipt.md --output final_analysis.json
   ```

3. **View Results:**
   ```bash
   cat final_analysis.json
   ```

Alternatively, you can use `batch_processor.py` to automate this process for multiple receipts or Excel files.

## Installation and Dependencies

Please ensure your Python environment has all necessary dependencies installed. Typically, you can use `pip` to install them:

```bash
pip install -r requirements.txt
```

(If `requirements.txt` does not exist, please manually install required libraries based on each module's `import` statements, such as `Pillow`, `requests`, `transformers`, etc.)

## Contributing

Contributions to this project are welcome! If you have any suggestions or find bugs, feel free to submit an Issue or Pull Request.

## License

[Please fill in your project license information here, e.g., MIT License]


