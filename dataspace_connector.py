"""
Dataspace 連接器

這個模組提供與公司 Dataspace 的連接功能，支援多種連接方式和認證機制。
可以獨立使用或與批量處理協調器整合。
"""

import os
import json
import yaml
import logging
import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataspaceConnector:
    """Dataspace 連接器，用於連接公司 Dataspace 並獲取資料"""
    
    def __init__(self, connection_config: Union[str, Dict[str, Any]], auth_config: Optional[Union[str, Dict[str, Any]]] = None):
        """
        初始化 Dataspace 連接器
        
        Args:
            connection_config: 連接配置 (字典或配置檔案路徑)
            auth_config: 認證配置 (字典或配置檔案路徑)
        """
        # 載入連接配置
        if isinstance(connection_config, str):
            self.connection_config = self._load_config_from_file(connection_config)
        else:
            self.connection_config = connection_config
            
        # 載入認證配置
        if auth_config:
            if isinstance(auth_config, str):
                self.auth_config = self._load_config_from_file(auth_config)
            else:
                self.auth_config = auth_config
        else:
            self.auth_config = {}
            
        # 驗證配置
        self._validate_config()
        
        # 初始化連接
        self.connection_type = self.connection_config.get("type", "rest_api").lower()
        self.connection = None
        
        logger.info(f"初始化 DataspaceConnector，連接類型: {self.connection_type}")
    
    def _load_config_from_file(self, config_file: str) -> Dict[str, Any]:
        """
        從檔案載入配置
        
        Args:
            config_file: 配置檔案路徑 (JSON 或 YAML 格式)
            
        Returns:
            配置字典
        """
        logger.info(f"從檔案載入配置: {config_file}")
        
        try:
            if not os.path.exists(config_file):
                logger.error(f"配置檔案不存在: {config_file}")
                return {}
                
            file_ext = os.path.splitext(config_file)[1].lower()
            
            if file_ext == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                logger.error(f"不支援的配置檔案格式: {file_ext}")
                return {}
                
            return config
            
        except Exception as e:
            logger.error(f"載入配置檔案失敗: {str(e)}")
            return {}
    
    def _validate_config(self) -> None:
        """
        驗證連接配置和認證配置
        
        Raises:
            ValueError: 如果配置無效
        """
        # 檢查連接配置
        if not self.connection_config:
            raise ValueError("連接配置為空")
            
        connection_type = self.connection_config.get("type", "").lower()
        if not connection_type:
            raise ValueError("連接類型未指定")
            
        if connection_type not in ["rest_api", "odbc", "jdbc", "database", "file_system"]:
            raise ValueError(f"不支援的連接類型: {connection_type}")
            
        # 根據連接類型檢查必要參數
        if connection_type == "rest_api":
            if "url" not in self.connection_config:
                raise ValueError("REST API 連接需要指定 URL")
        elif connection_type in ["odbc", "jdbc", "database"]:
            if "connection_string" not in self.connection_config:
                raise ValueError(f"{connection_type.upper()} 連接需要指定連接字串")
        elif connection_type == "file_system":
            if "path" not in self.connection_config:
                raise ValueError("檔案系統連接需要指定路徑")
                
        # 檢查認證配置
        auth_method = self.connection_config.get("auth_method", "").lower()
        if auth_method:
            if auth_method == "api_key":
                if not self.auth_config.get("api_key"):
                    raise ValueError("API Key 認證需要提供 API Key")
            elif auth_method == "oauth2":
                if not self.auth_config.get("client_id") or not self.auth_config.get("client_secret"):
                    raise ValueError("OAuth 2.0 認證需要提供 client_id 和 client_secret")
            elif auth_method == "basic":
                if not self.auth_config.get("username") or not self.auth_config.get("password"):
                    raise ValueError("基本認證需要提供用戶名和密碼")
            elif auth_method == "certificate":
                if not self.auth_config.get("cert_file"):
                    raise ValueError("憑證認證需要提供憑證檔案路徑")
    
    def connect(self) -> bool:
        """
        建立與 Dataspace 的連接
        
        Returns:
            連接是否成功
        """
        logger.info(f"建立 {self.connection_type} 連接")
        
        try:
            if self.connection_type == "rest_api":
                # REST API 連接不需要持久連接，每次請求時建立
                return True
                
            elif self.connection_type in ["odbc", "jdbc"]:
                # 這裡是示範代碼，實際使用時需要根據具體環境進行開發
                logger.warning(f"{self.connection_type.upper()} 連接尚未完全實現")
                
                # 模擬連接成功
                self.connection = {"status": "connected", "type": self.connection_type}
                return True
                
            elif self.connection_type == "database":
                # 這裡是示範代碼，實際使用時需要根據具體資料庫進行開發
                logger.warning("資料庫連接尚未完全實現")
                
                # 模擬連接成功
                self.connection = {"status": "connected", "type": "database"}
                return True
                
            elif self.connection_type == "file_system":
                path = self.connection_config.get("path", "")
                if not os.path.exists(path):
                    logger.error(f"檔案系統路徑不存在: {path}")
                    return False
                    
                self.connection = {"status": "connected", "type": "file_system", "path": path}
                return True
                
            else:
                logger.error(f"不支援的連接類型: {self.connection_type}")
                return False
                
        except Exception as e:
            logger.error(f"建立連接失敗: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """
        關閉與 Dataspace 的連接
        """
        if self.connection:
            logger.info(f"關閉 {self.connection_type} 連接")
            
            try:
                # 根據連接類型關閉連接
                if self.connection_type in ["odbc", "jdbc", "database"]:
                    # 這裡是示範代碼，實際使用時需要根據具體環境進行開發
                    pass
                    
                self.connection = None
                
            except Exception as e:
                logger.error(f"關閉連接失敗: {str(e)}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        獲取認證標頭
        
        Returns:
            認證標頭字典
        """
        auth_method = self.connection_config.get("auth_method", "").lower()
        headers = {}
        
        if auth_method == "api_key":
            api_key = self.auth_config.get("api_key", "")
            header_name = self.auth_config.get("header_name", "X-API-Key")
            headers[header_name] = api_key
            
        elif auth_method == "oauth2":
            # 這裡是示範代碼，實際使用時需要實現 OAuth 2.0 流程
            access_token = self.auth_config.get("access_token", "")
            if access_token:
                headers["Authorization"] = f"Bearer {access_token}"
            else:
                logger.warning("OAuth 2.0 認證缺少 access_token")
                
        elif auth_method == "basic":
            import base64
            username = self.auth_config.get("username", "")
            password = self.auth_config.get("password", "")
            auth_string = f"{username}:{password}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_auth}"
            
        return headers
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        執行查詢並返回結果
        
        Args:
            query: 查詢字串或查詢名稱
            params: 查詢參數
            
        Returns:
            查詢結果列表
        """
        logger.info(f"執行查詢: {query}")
        
        # 檢查是否已連接
        if self.connection_type not in ["rest_api"] and not self.connection:
            if not self.connect():
                logger.error("未建立連接，無法執行查詢")
                return []
        
        try:
            # 根據連接類型執行查詢
            if self.connection_type == "rest_api":
                return self._execute_rest_query(query, params)
                
            elif self.connection_type in ["odbc", "jdbc", "database"]:
                return self._execute_db_query(query, params)
                
            elif self.connection_type == "file_system":
                return self._execute_file_query(query, params)
                
            else:
                logger.error(f"不支援的連接類型: {self.connection_type}")
                return []
                
        except Exception as e:
            logger.error(f"執行查詢失敗: {str(e)}")
            return []
    
    def _execute_rest_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        執行 REST API 查詢
        
        Args:
            query: 查詢字串或查詢名稱
            params: 查詢參數
            
        Returns:
            查詢結果列表
        """
        # 檢查是否是預定義查詢
        predefined_queries = self.connection_config.get("queries", {})
        if query in predefined_queries:
            endpoint = predefined_queries[query].get("endpoint", "")
            method = predefined_queries[query].get("method", "GET").upper()
            query_params = predefined_queries[query].get("params", {})
            
            # 合併參數
            if params:
                query_params.update(params)
        else:
            # 假設 query 是端點
            endpoint = query
            method = "GET"
            query_params = params or {}
        
        # 構建 URL
        base_url = self.connection_config.get("url", "")
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # 獲取認證標頭
        headers = self._get_auth_headers()
        headers["Content-Type"] = "application/json"
        
        # 執行請求
        logger.info(f"發送 {method} 請求到 {url}")
        
        try:
            # 這裡是示範代碼，實際使用時需要處理各種 HTTP 方法和錯誤情況
            if method == "GET":
                response = requests.get(url, params=query_params, headers=headers)
            elif method == "POST":
                response = requests.post(url, json=query_params, headers=headers)
            else:
                logger.error(f"不支援的 HTTP 方法: {method}")
                return []
                
            # 檢查回應
            if response.status_code == 200:
                data = response.json()
                
                # 處理不同的回應格式
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # 嘗試找出結果列表
                    for key in ["data", "results", "items", "records"]:
                        if key in data and isinstance(data[key], list):
                            return data[key]
                    
                    # 如果找不到列表，返回字典作為單一項目的列表
                    return [data]
                else:
                    logger.error(f"無法解析回應: {data}")
                    return []
            else:
                logger.error(f"請求失敗: {response.status_code} {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"執行 REST 查詢失敗: {str(e)}")
            return []
    
    def _execute_db_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        執行資料庫查詢
        
        Args:
            query: SQL 查詢字串
            params: 查詢參數
            
        Returns:
            查詢結果列表
        """
        logger.warning("資料庫查詢尚未完全實現，返回模擬資料")
        
        # 這裡是示範代碼，實際使用時需要根據具體資料庫進行開發
        # 模擬查詢結果
        mock_data = [
            {
                "merchant_info": {
                    "name": "模擬商家 DB-1",
                    "address": "模擬地址 1"
                },
                "transaction_info": {
                    "date": "2023-01-01",
                    "receipt_number": "DB-001"
                },
                "price_summary": {
                    "total": 100.0,
                    "tax": 5.0
                }
            },
            {
                "merchant_info": {
                    "name": "模擬商家 DB-2",
                    "address": "模擬地址 2"
                },
                "transaction_info": {
                    "date": "2023-01-02",
                    "receipt_number": "DB-002"
                },
                "price_summary": {
                    "total": 200.0,
                    "tax": 10.0
                }
            }
        ]
        
        return mock_data
    
    def _execute_file_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        執行檔案系統查詢
        
        Args:
            query: 檔案路徑或查詢字串
            params: 查詢參數
            
        Returns:
            查詢結果列表
        """
        if not self.connection:
            logger.error("未建立檔案系統連接")
            return []
            
        base_path = self.connection.get("path", "")
        file_path = os.path.join(base_path, query)
        
        if not os.path.exists(file_path):
            logger.error(f"檔案不存在: {file_path}")
            return []
            
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        return data
                    else:
                        return [data]
                        
            elif file_ext in ['.csv']:
                df = pd.read_csv(file_path)
                return df.to_dict('records')
                
            elif file_ext in ['.xlsx', '.xls']:
                sheet_name = params.get("sheet_name") if params else None
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                return df.to_dict('records')
                
            else:
                logger.error(f"不支援的檔案格式: {file_ext}")
                return []
                
        except Exception as e:
            logger.error(f"讀取檔案失敗: {str(e)}")
            return []
    
    def get_expense_data(self, query: str = "expenses", params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        獲取費用資料並轉換為標準格式
        
        Args:
            query: 查詢字串或查詢名稱
            params: 查詢參數
            
        Returns:
            標準格式的收據資料列表
        """
        logger.info(f"獲取費用資料: {query}")
        
        # 執行查詢
        raw_data = self.execute_query(query, params)
        
        if not raw_data:
            logger.warning("未獲取到費用資料")
            return []
            
        # 轉換為標準格式
        receipt_data_list = []
        
        for item in raw_data:
            # 檢查是否已經是標準格式
            if all(key in item for key in ["merchant_info", "transaction_info", "price_summary"]):
                receipt_data_list.append(item)
                continue
                
            # 轉換為標準格式
            receipt_data = {}
            
            # 商家資訊
            merchant_info = {}
            for key in ["merchant_name", "merchant_address", "merchant_phone", "merchant_tax_id"]:
                if key in item and item[key]:
                    field_name = key.replace("merchant_", "")
                    merchant_info[field_name] = item[key]
                    
            if "merchant_info" in item and isinstance(item["merchant_info"], dict):
                merchant_info.update(item["merchant_info"])
                
            if merchant_info:
                receipt_data["merchant_info"] = merchant_info
            
            # 交易資訊
            transaction_info = {}
            for key in ["transaction_date", "transaction_time", "transaction_receipt_number"]:
                if key in item and item[key]:
                    field_name = key.replace("transaction_", "")
                    transaction_info[field_name] = item[key]
                    
            # 處理常見欄位名稱
            if "date" in item and item["date"]:
                transaction_info["date"] = item["date"]
            if "time" in item and item["time"]:
                transaction_info["time"] = item["time"]
            if "receipt_number" in item and item["receipt_number"]:
                transaction_info["receipt_number"] = item["receipt_number"]
                
            if "transaction_info" in item and isinstance(item["transaction_info"], dict):
                transaction_info.update(item["transaction_info"])
                
            if transaction_info:
                receipt_data["transaction_info"] = transaction_info
            
            # 價格資訊
            price_summary = {}
            for key in ["price_total", "price_subtotal", "price_tax", "price_discounts"]:
                if key in item and item[key] is not None:
                    field_name = key.replace("price_", "")
                    price_summary[field_name] = float(item[key])
                    
            # 處理常見欄位名稱
            if "amount" in item and item["amount"] is not None:
                price_summary["total"] = float(item["amount"])
            if "total" in item and item["total"] is not None:
                price_summary["total"] = float(item["total"])
            if "subtotal" in item and item["subtotal"] is not None:
                price_summary["subtotal"] = float(item["subtotal"])
            if "tax" in item and item["tax"] is not None:
                price_summary["tax"] = float(item["tax"])
                
            if "price_summary" in item and isinstance(item["price_summary"], dict):
                price_summary.update(item["price_summary"])
                
            if price_summary:
                receipt_data["price_summary"] = price_summary
            
            # 付款資訊
            payment_info = {}
            for key in ["payment_method", "payment_card_last_four", "payment_authorization_code"]:
                if key in item and item[key]:
                    field_name = key.replace("payment_", "")
                    payment_info[field_name] = item[key]
                    
            if "payment_info" in item and isinstance(item["payment_info"], dict):
                payment_info.update(item["payment_info"])
                
            if payment_info:
                receipt_data["payment_info"] = payment_info
            
            # 其他資訊
            other_info = {}
            if "category" in item and item["category"]:
                other_info["category"] = item["category"]
            if "description" in item and item["description"]:
                other_info["description"] = item["description"]
            if "notes" in item and item["notes"]:
                other_info["notes"] = item["notes"]
                
            if "other_info" in item and isinstance(item["other_info"], dict):
                other_info.update(item["other_info"])
                
            if other_info:
                receipt_data["other_info"] = other_info
            
            # 收據圖片路徑 (如果有)
            if "receipt_image" in item and item["receipt_image"]:
                receipt_data["receipt_image"] = item["receipt_image"]
            
            receipt_data_list.append(receipt_data)
        
        logger.info(f"轉換了 {len(receipt_data_list)} 筆費用資料")
        return receipt_data_list


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='連接 Dataspace 並獲取費用資料')
    parser.add_argument('--connection', type=str, required=True, help='連接配置檔案路徑')
    parser.add_argument('--auth', type=str, help='認證配置檔案路徑')
    parser.add_argument('--query', type=str, default="expenses", help='查詢字串或查詢名稱')
    parser.add_argument('--output', type=str, help='輸出 JSON 檔案路徑')
    
    args = parser.parse_args()
    
    try:
        # 初始化 Dataspace 連接器
        connector = DataspaceConnector(connection_config=args.connection, auth_config=args.auth)
        
        # 連接 Dataspace
        if not connector.connect():
            print("連接 Dataspace 失敗")
            return 1
            
        # 獲取費用資料
        receipt_data_list = connector.get_expense_data(query=args.query)
        
        # 輸出結果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(receipt_data_list, f, ensure_ascii=False, indent=2)
            print(f"資料已保存至: {args.output}")
        else:
            print(json.dumps(receipt_data_list, ensure_ascii=False, indent=2))
            
        # 關閉連接
        connector.disconnect()
            
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
