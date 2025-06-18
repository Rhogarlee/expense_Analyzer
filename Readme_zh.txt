# Smart Expense Analyzer

## 專案概述

智能費用分析系統是一個結合 VLM（視覺語言模型）和 LLM（大型語言模型）的端到端解決方案，用於自動處理、分析和監控企業費用。系統能夠從收據圖片中提取結構化數據，進行分類、異常檢測，並生成易於理解的報告和建議。

本 `README.md` 文件將簡要說明系統中幾個核心模組的用法。

## 核心模組用法說明

### 1. VLM 模組 (Visual Language Model Module)

**檔案名稱:** `vlm_receipt_to_markdown.py`

**功能:** 負責將收據圖片轉換為結構化的 Markdown 格式文本。VLM 模型能夠識別圖片中的文字、數字以及它們之間的空間關係和層次結構，將這些視覺信息轉化為機器可讀且保留格式的文本。

**使用方法:**

```bash
python vlm_receipt_to_markdown.py --image /path/to/receipt.jpg --model qwen2-vl-2b --output receipt.md
```

**參數說明:**
- `--image`：收據圖片路徑 (必填)
- `--model`：使用的 VLM 模型 (可選：`qwen2-vl-2b`, `minicpm`, `gpt-4o-mini`, `claude-3.5`)。預設為 `qwen2-vl-2b`。
- `--api-key`：API 金鑰 (使用商業模型時需要)
- `--output`：輸出 Markdown 檔案路徑 (可選)。如果未指定，則輸出到標準輸出。

### 2. LLM 模組 (Large Language Model Module)

**檔案名稱:** `llm_markdown_parser.py`

**功能:** 接收 VLM 模組輸出的 Markdown 格式收據數據，並利用其強大的自然語言理解和生成能力，從中提取關鍵信息，進行費用分類和初步的異常檢測，最終輸出標準化的 JSON 格式數據。

**使用方法:**

```bash
python llm_markdown_parser.py /path/to/receipt.md --model llama3 --output analysis.json
```

**參數說明:**
- 第一個參數：Markdown 收據文件路徑 (必填)
- `--model`：使用的 LLM 模型 (可選：`llama3`, `mistral`, `gpt-4`, `claude-3`)。預設為 `llama3`。
- `--api-key`：API 金鑰 (使用商業模型時需要)
- `--output`：輸出 JSON 分析結果的文件路徑 (可選)。如果未指定，則輸出到標準輸出。

### 3. Batch Processor (批量處理協調器)

**檔案名稱:** `batch_processor.py`

**功能:** 統籌 VLM、LLM 及數據整合模組，實現對大量收據圖片或 Excel 數據的批量處理，大幅提升處理效率。它支援並行處理機制，可配置並行工作者數量，有效利用多核 CPU 資源。

**使用方法:**

```bash
# 處理指定目錄下的所有收據圖片
python batch_processor.py --input-dir /path/to/receipts --output-dir /path/to/results --vlm-model qwen2-vl-2b --llm-model llama3

# 從 Excel 文件導入數據進行處理
python batch_processor.py --input-excel /path/to/expenses.xlsx --output-dir /path/to/results --llm-model llama3
```

**參數說明:**
- `--input-dir`：包含收據圖片的輸入目錄路徑 (與 `--input-excel` 二選一)
- `--input-excel`：包含費用數據的 Excel 文件路徑 (與 `--input-dir` 二選一)
- `--output-dir`：輸出結果（JSON 文件和摘要報告）的目錄路徑 (必填)
- `--vlm-model`：批量處理時使用的 VLM 模型 (可選，僅當處理圖片時需要)
- `--llm-model`：批量處理時使用的 LLM 模型 (可選)
- `--api-key`：API 金鑰 (使用商業模型時需要)
- `--workers`：並行處理的工作者數量 (可選，預設為 CPU 核心數)

### 4. Dataspace Connector (數據空間連接器)

**檔案名稱:** `dataspace_connector.py`

**功能:** 提供與公司 Dataspace 的連接介面，支援多種連接方式和認證機制。這使得系統不僅能處理新收據，還能整合歷史數據進行全面分析。此模組主要作為後端服務或數據整合流程的一部分被調用，不直接提供命令行介面。

**使用方法:**

此模組通常會被其他 Python 腳本（例如 `batch_processor.py` 或其他數據整合服務）導入並調用。以下是一個簡化的 Python 程式碼示例，展示如何使用 `dataspace_connector.py`：

```python
from dataspace_connector import DataspaceConnector

# 初始化連接器，可能需要提供認證資訊
connector = DataspaceConnector(api_endpoint="https://your.dataspace.api", auth_token="your_token")

# 範例：從 Dataspace 讀取數據
data = connector.read_data(query="SELECT * FROM expenses WHERE date > '2024-01-01'")
print("從 Dataspace 讀取的數據:", data)

# 範例：寫入數據到 Dataspace
new_expense = {"date": "2025-06-18", "amount": 100.0, "category": "Office Supplies"}
connector.write_data(table_name="expenses", data=new_expense)
print("數據已寫入 Dataspace")
```

**主要方法:**
- `__init__(api_endpoint, auth_token, ...)`: 初始化連接器，配置連接參數和認證資訊。
- `read_data(query, ...)`: 從 Dataspace 讀取數據，支援 SQL 查詢或其他查詢語言。
- `write_data(table_name, data, ...)`: 將數據寫入 Dataspace 的指定表格。
- `update_data(table_name, data, condition, ...)`: 更新 Dataspace 中的數據。
- `delete_data(table_name, condition, ...)`: 從 Dataspace 中刪除數據。

**注意事項:**
- 具體的 `api_endpoint` 和 `auth_token` 需要根據您的公司 Dataspace 配置。
- 確保您的環境已安裝所有必要的依賴庫，例如用於數據庫連接或 API 調用的庫。

## 端到端使用流程 (簡化)

您可以將 VLM 和 LLM 模組串聯起來，實現端到端的收據處理：

1. **圖片轉 Markdown:**
   ```bash
   python vlm_receipt_to_markdown.py --image your_receipt.jpg --output temp_receipt.md
   ```

2. **Markdown 解析與分析:**
   ```bash
   python llm_markdown_parser.py temp_receipt.md --output final_analysis.json
   ```

3. **查看結果:**
   ```bash
   cat final_analysis.json
   ```

或者，您可以使用 `batch_processor.py` 來自動化這個流程，處理多個收據或 Excel 文件。

## 安裝與依賴

請確保您的 Python 環境已安裝所有必要的依賴庫。通常，您可以使用 `pip` 來安裝：

```bash
pip install -r requirements.txt
```

(如果 `requirements.txt` 不存在，請根據各模組的 `import` 語句手動安裝所需庫，例如 `Pillow`, `requests`, `transformers` 等。)

## 貢獻

歡迎對本專案提出貢獻！如果您有任何建議或發現錯誤，請隨時提交 Issue 或 Pull Request。

## 授權

[請在此處填寫您的專案授權資訊，例如 MIT License]


