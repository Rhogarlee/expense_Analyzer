"""
批量處理協調器

這個模組提供批量處理多個收據的功能，支援並行處理以提高效率。
可以處理目錄中的多張收據圖片，或從Excel等資料來源導入資料。
"""

import os
import json
import time
import logging
import argparse
import concurrent.futures
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# 導入自定義模組
from vlm_receipt_to_markdown import ReceiptToMarkdownConverter
from llm_markdown_parser import MarkdownExpenseAnalyzer

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """批量處理收據的協調器類"""
    
    def __init__(self, 
                 vlm_model: str = "qwen2-vl-2b", 
                 llm_model: str = "llama3",
                 vlm_api_key: Optional[str] = None,
                 llm_api_key: Optional[str] = None,
                 vlm_api_endpoint: Optional[str] = None,
                 llm_api_endpoint: Optional[str] = None,
                 parallel_workers: int = 4):
        """
        初始化批量處理協調器
        
        Args:
            vlm_model: 使用的VLM模型名稱
            llm_model: 使用的LLM模型名稱
            vlm_api_key: VLM API金鑰(如果使用商業模型)
            llm_api_key: LLM API金鑰(如果使用商業模型)
            vlm_api_endpoint: VLM API端點(如果使用自託管模型)
            llm_api_endpoint: LLM API端點(如果使用自託管模型)
            parallel_workers: 並行工作者數量
        """
        self.vlm_model = vlm_model
        self.llm_model = llm_model
        self.vlm_api_key = vlm_api_key
        self.llm_api_key = llm_api_key
        self.vlm_api_endpoint = vlm_api_endpoint
        self.llm_api_endpoint = llm_api_endpoint
        self.parallel_workers = parallel_workers
        
        self.results = []
        self.errors = []
        self.warnings = []
        
        logger.info(f"初始化批量處理協調器，VLM模型: {vlm_model}, LLM模型: {llm_model}, 並行工作者: {parallel_workers}")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        處理單張收據圖片
        
        Args:
            image_path: 收據圖片路徑
            
        Returns:
            處理結果字典
        """
        start_time = time.time()
        result = {
            "status": "success",
            "processing_time": 0,
            "image_path": image_path,
            "structured_data": {},
            "category": "",
            "anomalies": []
        }
        
        try:
            # 1. 初始化VLM轉換器
            vlm_converter = ReceiptToMarkdownConverter(
                model_name=self.vlm_model,
                api_key=self.vlm_api_key,
                api_endpoint=self.vlm_api_endpoint
            )
            
            # 2. 將收據圖片轉換為Markdown
            markdown_text = vlm_converter.convert(image_path)
            
            # 3. 初始化LLM分析器
            llm_analyzer = MarkdownExpenseAnalyzer(
                model_name=self.llm_model,
                api_key=self.llm_api_key,
                api_endpoint=self.llm_api_endpoint
            )
            
            # 4. 解析Markdown並進行分析
            analysis_result = llm_analyzer.analyze(markdown_text)
            
            # 5. 提取結果
            result["structured_data"] = analysis_result["structured_data"]
            result["category"] = analysis_result["analysis"]["category"]
            result["anomalies"] = analysis_result["analysis"]["anomalies_detected"]
            
        except Exception as e:
            logger.error(f"處理圖片 {image_path} 時發生錯誤: {str(e)}")
            result["status"] = "error"
            result["error_message"] = str(e)
            self.errors.append({"image_path": image_path, "error": str(e)})
        
        # 計算處理時間
        result["processing_time"] = time.time() - start_time
        
        return result
    
    def process_directory(self, directory_path: str, recursive: bool = False, file_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        處理目錄中的所有收據圖片
        
        Args:
            directory_path: 收據圖片目錄路徑
            recursive: 是否遞迴處理子目錄
            file_extensions: 要處理的檔案副檔名列表，默認為 ['.jpg', '.jpeg', '.png']
            
        Returns:
            處理結果列表
        """
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png']
            
        # 收集所有圖片檔案路徑
        image_paths = []
        if recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in file_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    image_paths.append(os.path.join(directory_path, file))
        
        logger.info(f"在目錄 {directory_path} 中找到 {len(image_paths)} 個圖片檔案")
        
        # 使用線程池並行處理圖片
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # 提交所有任務
            future_to_path = {executor.submit(self.process_image, path): path for path in image_paths}
            
            # 收集結果
            success_count = 0
            fail_count = 0
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    if result["status"] == "success":
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"處理圖片 {path} 時發生未捕獲的異常: {str(e)}")
                    self.results.append({
                        "status": "error",
                        "processing_time": 0,
                        "image_path": path,
                        "error_message": str(e)
                    })
                    self.errors.append({"image_path": path, "error": str(e)})
                    fail_count += 1
        
        logger.info(f"完成處理 {len(image_paths)} 個圖片檔案，成功: {success_count}, 失敗: {fail_count}")
        
        return self.results
    
    def process_excel_data(self, excel_path: str, sheet_name: Optional[str] = None, mapping_config: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        處理Excel檔案中的收據資料
        
        Args:
            excel_path: Excel檔案路徑
            sheet_name: 工作表名稱，默認為第一個工作表
            mapping_config: 欄位映射配置，用於將Excel欄位映射到標準格式
            
        Returns:
            處理結果列表
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("處理Excel需要pandas庫，請安裝: pip install pandas openpyxl")
            raise ImportError("處理Excel需要pandas庫，請安裝: pip install pandas openpyxl")
        
        start_time = time.time()
        
        try:
            # 讀取Excel檔案
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.info(f"從Excel檔案 {excel_path} 讀取了 {len(df)} 筆資料")
            
            # 應用欄位映射
            if mapping_config:
                # 將Excel欄位映射到標準格式
                mapped_data = []
                for _, row in df.iterrows():
                    item = {}
                    for std_field, excel_field in mapping_config.items():
                        if excel_field in row:
                            item[std_field] = row[excel_field]
                    mapped_data.append(item)
            else:
                # 直接使用Excel資料
                mapped_data = df.to_dict('records')
            
            # 初始化LLM分析器
            llm_analyzer = MarkdownExpenseAnalyzer(
                model_name=self.llm_model,
                api_key=self.llm_api_key,
                api_endpoint=self.llm_api_endpoint
            )
            
            # 處理每筆資料
            excel_results = []
            for idx, data in enumerate(mapped_data):
                item_start_time = time.time()
                result = {
                    "status": "success",
                    "processing_time": 0,
                    "excel_row": idx + 2,  # Excel行號從2開始(標題為1)
                    "structured_data": data,
                    "category": "",
                    "anomalies": []
                }
                
                try:
                    # 使用LLM進行分類和異常檢測
                    category = llm_analyzer.categorize_expense(data)
                    anomalies = llm_analyzer.detect_anomalies(data, category)
                    
                    result["category"] = category
                    result["anomalies"] = anomalies
                    
                except Exception as e:
                    logger.error(f"處理Excel行 {idx + 2} 時發生錯誤: {str(e)}")
                    result["status"] = "error"
                    result["error_message"] = str(e)
                    self.errors.append({"excel_row": idx + 2, "error": str(e)})
                
                # 計算處理時間
                result["processing_time"] = time.time() - item_start_time
                excel_results.append(result)
            
            # 添加到總結果
            self.results.extend(excel_results)
            
            # 計算成功和失敗數量
            success_count = sum(1 for r in excel_results if r["status"] == "success")
            fail_count = len(excel_results) - success_count
            
            logger.info(f"完成處理 {len(excel_results)} 筆Excel資料，成功: {success_count}, 失敗: {fail_count}")
            
            return excel_results
            
        except Exception as e:
            logger.error(f"處理Excel檔案 {excel_path} 時發生錯誤: {str(e)}")
            self.errors.append({"excel_path": excel_path, "error": str(e)})
            return []
    
    def process_dataspace(self, config_path: str) -> List[Dict[str, Any]]:
        """
        從公司Dataspace獲取並處理資料
        
        Args:
            config_path: Dataspace連接配置檔案路徑
            
        Returns:
            處理結果列表
        """
        try:
            # 讀取配置檔案
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 根據配置連接Dataspace
            # 這裡需要根據實際的Dataspace API進行實現
            # 以下為示例代碼框架
            
            connection_type = config.get("connection_type", "")
            
            if connection_type == "rest_api":
                # 連接REST API
                api_url = config.get("api_url", "")
                api_key = config.get("api_key", "")
                
                # 實現REST API連接邏輯
                logger.info(f"連接REST API: {api_url}")
                
            elif connection_type == "database":
                # 連接資料庫
                db_type = config.get("db_type", "")
                db_host = config.get("db_host", "")
                db_name = config.get("db_name", "")
                db_user = config.get("db_user", "")
                db_password = config.get("db_password", "")
                
                # 實現資料庫連接邏輯
                logger.info(f"連接資料庫: {db_type} - {db_host}/{db_name}")
                
            elif connection_type == "file_system":
                # 連接檔案系統
                file_path = config.get("file_path", "")
                file_type = config.get("file_type", "")
                
                # 實現檔案系統連接邏輯
                logger.info(f"連接檔案系統: {file_path} ({file_type})")
                
            else:
                logger.error(f"不支援的連接類型: {connection_type}")
                self.errors.append({"config_path": config_path, "error": f"不支援的連接類型: {connection_type}"})
                return []
            
            # 模擬從Dataspace獲取資料
            # 實際實現應根據具體的Dataspace API
            dataspace_results = []
            
            logger.info(f"從Dataspace獲取了 {len(dataspace_results)} 筆資料")
            
            # 添加到總結果
            self.results.extend(dataspace_results)
            
            return dataspace_results
            
        except Exception as e:
            logger.error(f"處理Dataspace配置 {config_path} 時發生錯誤: {str(e)}")
            self.errors.append({"config_path": config_path, "error": str(e)})
            return []
    
    def save_results(self, output_path: str, format_type: str = "json") -> None:
        """
        保存處理結果
        
        Args:
            output_path: 輸出檔案路徑
            format_type: 輸出格式，支援 'json', 'csv', 'excel'
        """
        if not self.results:
            logger.warning("沒有結果可保存")
            return
        
        try:
            if format_type.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, ensure_ascii=False, indent=2)
                    
            elif format_type.lower() == "csv":
                try:
                    import pandas as pd
                except ImportError:
                    logger.error("保存為CSV需要pandas庫，請安裝: pip install pandas")
                    raise ImportError("保存為CSV需要pandas庫，請安裝: pip install pandas")
                
                # 將結果轉換為扁平結構
                flat_results = []
                for result in self.results:
                    flat_result = {
                        "status": result.get("status", ""),
                        "processing_time": result.get("processing_time", 0),
                        "image_path": result.get("image_path", ""),
                        "excel_row": result.get("excel_row", ""),
                        "category": result.get("category", ""),
                        "anomalies": ", ".join(result.get("anomalies", [])),
                    }
                    
                    # 添加結構化資料的主要欄位
                    structured_data = result.get("structured_data", {})
                    if structured_data:
                        merchant_info = structured_data.get("merchant_info", {})
                        flat_result["merchant_name"] = merchant_info.get("name", "")
                        flat_result["merchant_address"] = merchant_info.get("address", "")
                        
                        transaction_info = structured_data.get("transaction_info", {})
                        flat_result["transaction_date"] = transaction_info.get("date", "")
                        flat_result["transaction_time"] = transaction_info.get("time", "")
                        flat_result["receipt_number"] = transaction_info.get("receipt_number", "")
                        
                        price_summary = structured_data.get("price_summary", {})
                        flat_result["subtotal"] = price_summary.get("subtotal", "")
                        flat_result["tax"] = price_summary.get("tax", "")
                        flat_result["total"] = price_summary.get("total", "")
                        
                        payment_info = structured_data.get("payment_info", {})
                        flat_result["payment_method"] = payment_info.get("method", "")
                    
                    flat_results.append(flat_result)
                
                # 保存為CSV
                df = pd.DataFrame(flat_results)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
            elif format_type.lower() == "excel":
                try:
                    import pandas as pd
                except ImportError:
                    logger.error("保存為Excel需要pandas庫，請安裝: pip install pandas openpyxl")
                    raise ImportError("保存為Excel需要pandas庫，請安裝: pip install pandas openpyxl")
                
                # 將結果轉換為扁平結構 (與CSV相同)
                flat_results = []
                for result in self.results:
                    flat_result = {
                        "status": result.get("status", ""),
                        "processing_time": result.get("processing_time", 0),
                        "image_path": result.get("image_path", ""),
                        "excel_row": result.get("excel_row", ""),
                        "category": result.get("category", ""),
                        "anomalies": ", ".join(result.get("anomalies", [])),
                    }
                    
                    # 添加結構化資料的主要欄位
                    structured_data = result.get("structured_data", {})
                    if structured_data:
                        merchant_info = structured_data.get("merchant_info", {})
                        flat_result["merchant_name"] = merchant_info.get("name", "")
                        flat_result["merchant_address"] = merchant_info.get("address", "")
                        
                        transaction_info = structured_data.get("transaction_info", {})
                        flat_result["transaction_date"] = transaction_info.get("date", "")
                        flat_result["transaction_time"] = transaction_info.get("time", "")
                        flat_result["receipt_number"] = transaction_info.get("receipt_number", "")
                        
                        price_summary = structured_data.get("price_summary", {})
                        flat_result["subtotal"] = price_summary.get("subtotal", "")
                        flat_result["tax"] = price_summary.get("tax", "")
                        flat_result["total"] = price_summary.get("total", "")
                        
                        payment_info = structured_data.get("payment_info", {})
                        flat_result["payment_method"] = payment_info.get("method", "")
                    
                    flat_results.append(flat_result)
                
                # 保存為Excel
                df = pd.DataFrame(flat_results)
                df.to_excel(output_path, index=False)
                
            else:
                logger.error(f"不支援的輸出格式: {format_type}")
                raise ValueError(f"不支援的輸出格式: {format_type}")
                
            logger.info(f"處理結果已保存至: {output_path}")
            
        except Exception as e:
            logger.error(f"保存結果時發生錯誤: {str(e)}")
            raise
    
    def generate_summary_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成處理結果摘要報告
        
        Args:
            output_path: 摘要報告輸出檔案路徑，如果為None則不保存
            
        Returns:
            摘要報告字典
        """
        if not self.results:
            logger.warning("沒有結果可生成摘要")
            return {}
        
        # 計算基本統計資料
        total_receipts = len(self.results)
        success_count = sum(1 for r in self.results if r.get("status") == "success")
        failed_count = total_receipts - success_count
        success_rate = success_count / total_receipts if total_receipts > 0 else 0
        
        # 計算平均處理時間
        processing_times = [r.get("processing_time", 0) for r in self.results if r.get("status") == "success"]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # 統計分類分佈
        category_distribution = {}
        for result in self.results:
            if result.get("status") == "success":
                category = result.get("category", "未分類")
                category_distribution[category] = category_distribution.get(category, 0) + 1
        
        # 統計異常檢測
        anomalies_detected = 0
        anomalies_list = []
        for result in self.results:
            if result.get("status") == "success":
                anomalies = result.get("anomalies", [])
                if anomalies:
                    anomalies_detected += 1
                    anomalies_list.append({
                        "source": result.get("image_path", result.get("excel_row", "未知來源")),
                        "anomalies": anomalies
                    })
        
        # 組合摘要報告
        summary = {
            "total_receipts": total_receipts,
            "success_count": success_count,
            "failed_count": failed_count,
            "success_rate": success_rate,
            "avg_processing_time": avg_processing_time,
            "category_distribution": category_distribution,
            "anomalies_detected": anomalies_detected,
            "anomalies_list": anomalies_list,
            "warnings": self.warnings,
            "errors": self.errors
        }
        
        # 保存摘要報告
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"摘要報告已保存至: {output_path}")
            except Exception as e:
                logger.error(f"保存摘要報告時發生錯誤: {str(e)}")
        
        return summary


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='批量處理收據圖片')
    
    # 輸入來源參數
    input_group = parser.add_argument_group('輸入來源')
    input_group.add_argument('--batch-dir', type=str, help='收據圖片目錄路徑')
    input_group.add_argument('--recursive', action='store_true', help='是否遞迴處理子目錄')
    input_group.add_argument('--excel-input', type=str, help='Excel檔案路徑')
    input_group.add_argument('--sheet-name', type=str, help='Excel工作表名稱')
    input_group.add_argument('--mapping', type=str, help='Excel欄位映射配置檔案路徑')
    input_group.add_argument('--dataspace-connect', type=str, help='Dataspace連接配置檔案路徑')
    
    # 模型參數
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--vlm-model', type=str, default='qwen2-vl-2b', 
                        choices=['qwen2-vl-2b', 'minicpm', 'gpt-4o-mini', 'claude-3.5'],
                        help='使用的VLM模型')
    model_group.add_argument('--llm-model', type=str, default='llama3', 
                        choices=['llama3', 'mistral', 'gpt-4', 'claude-3'],
                        help='使用的LLM模型')
    model_group.add_argument('--vlm-api-key', type=str, help='VLM API金鑰')
    model_group.add_argument('--llm-api-key', type=str, help='LLM API金鑰')
    model_group.add_argument('--vlm-api-endpoint', type=str, help='VLM API端點')
    model_group.add_argument('--llm-api-endpoint', type=str, help='LLM API端點')
    
    # 處理參數
    process_group = parser.add_argument_group('處理配置')
    process_group.add_argument('--parallel', type=int, default=4, help='並行工作者數量')
    
    # 輸出參數
    output_group = parser.add_argument_group('輸出配置')
    output_group.add_argument('--output', type=str, required=True, help='輸出檔案路徑')
    output_group.add_argument('--format', type=str, default='json', choices=['json', 'csv', 'excel'],
                         help='輸出格式')
    output_group.add_argument('--summary', type=str, help='摘要報告輸出檔案路徑')
    
    args = parser.parse_args()
    
    # 檢查是否至少提供一個輸入來源
    if not args.batch_dir and not args.excel_input and not args.dataspace_connect:
        parser.error("至少需要提供一個輸入來源: --batch-dir, --excel-input 或 --dataspace-connect")
    
    try:
        # 初始化批量處理協調器
        processor = BatchProcessor(
            vlm_model=args.vlm_model,
            llm_model=args.llm_model,
            vlm_api_key=args.vlm_api_key,
            llm_api_key=args.llm_api_key,
            vlm_api_endpoint=args.vlm_api_endpoint,
            llm_api_endpoint=args.llm_api_endpoint,
            parallel_workers=args.parallel
        )
        
        # 處理收據圖片目錄
        if args.batch_dir:
            processor.process_directory(args.batch_dir, recursive=args.recursive)
        
        # 處理Excel檔案
        if args.excel_input:
            mapping_config = None
            if args.mapping:
                with open(args.mapping, 'r', encoding='utf-8') as f:
                    mapping_config = json.load(f)
            
            processor.process_excel_data(args.excel_input, sheet_name=args.sheet_name, mapping_config=mapping_config)
        
        # 處理Dataspace
        if args.dataspace_connect:
            processor.process_dataspace(args.dataspace_connect)
        
        # 保存處理結果
        processor.save_results(args.output, format_type=args.format)
        
        # 生成摘要報告
        if args.summary:
            processor.generate_summary_report(args.summary)
        
    except Exception as e:
        logger.error(f"處理過程中發生錯誤: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
