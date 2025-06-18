"""
VLM Receipt to Markdown Converter

This module uses Visual Language Models (VLM) to convert receipt images into structured Markdown format.
This preserves the hierarchical structure of receipts for subsequent LLM processing and analysis.

Supported VLM models:
- Qwen2-VL-2B (open source)
- MiniCPM (open source)
- GPT-4o Mini (commercial)
- Claude 3.5 (commercial)
"""

import os
import base64
import requests
import argparse
from PIL import Image
from io import BytesIO
import json
from typing import Dict, Any, Optional, Union, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReceiptToMarkdownConverter:
    """Main class for converting receipt images to Markdown"""
    
    def __init__(self, model_name: str = "qwen2-vl-2b", api_key: Optional[str] = None):
        """
        Initialize the converter
        
        Args:
            model_name: Name of the VLM model to use
            api_key: API key (if using commercial models)
        """
        self.model_name = model_name.lower()
        self.api_key = api_key
        self.supported_models = ["qwen2-vl-2b", "minicpm", "gpt-4o-mini", "claude-3.5"]
        
        if self.model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {', '.join(self.supported_models)}")
        
        if self.model_name in ["gpt-4o-mini", "claude-3.5"] and not api_key:
            raise ValueError(f"Using {model_name} requires an API key")
            
        logger.info(f"Initializing ReceiptToMarkdownConverter with model: {model_name}")
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Preprocess receipt image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processed PIL Image object
        """
        logger.info(f"Preprocessing image: {image_path}")
        
        try:
            # Read image
            img = Image.open(image_path)
            
            # Basic preprocessing (can be extended as needed)
            # 1. Convert to RGB mode (handle RGBA or other modes)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 2. Resize if too large
            max_size = 1600  # Maximum dimension
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                
            # 3. Additional preprocessing steps could be added, such as:
            # - Contrast enhancement
            # - Noise reduction
            # - Rotation correction
            # - Cropping
                
            return img
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def encode_image(self, image: Image.Image) -> str:
        """
        Encode PIL Image as base64 string
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded image string
        """
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def generate_prompt(self) -> str:
        """
        Generate prompt for VLM
        
        Returns:
            Prompt suitable for the current model
        """
        # Base prompt template
        base_prompt = """
Please analyze this receipt image and convert its content into structured Markdown format.
Preserve the hierarchical structure of the receipt, including:

1. Merchant information (name, address, phone, etc.)
2. Transaction information (date, time, receipt number, etc.)
3. Item list (using Markdown table format)
4. Price information (subtotal, tax, total, etc.)
5. Payment information (payment method, last four digits of card, etc.)
6. Other information (return policy, promotional messages, etc.)

Use appropriate Markdown syntax to represent different levels of information:
- Use # for merchant name
- Use ## for main section titles
- Use **bold** to mark important information
- Use tables for item lists
- Maintain original formatting and spatial relationships

Please ensure the output is clean and preserves all information as much as possible.
"""
        
        # Adjust prompt for different models
        if self.model_name == "qwen2-vl-2b":
            return base_prompt + "\nPlease output directly in Markdown format without additional explanations."
        elif self.model_name == "minicpm":
            return base_prompt + "\nPlease output directly in Markdown format without additional explanations or preamble."
        elif self.model_name == "gpt-4o-mini":
            return base_prompt + "\nPlease output only in Markdown format without any other responses."
        elif self.model_name == "claude-3.5":
            return base_prompt + "\nPlease provide only the Markdown format output without any preamble or subsequent explanations."
        
        return base_prompt
    
    def call_open_source_model(self, image: Image.Image, prompt: str) -> str:
        """
        Call open source VLM model
        
        Args:
            image: Preprocessed image
            prompt: Prompt text
            
        Returns:
            Markdown text generated by the model
        """
        logger.info(f"Calling open source model: {self.model_name}")
        
        try:
            # This is demonstration code; actual implementation depends on specific model API or local deployment
            if self.model_name == "qwen2-vl-2b":
                # Example code for using Qwen2-VL-2B model
                # In actual use, this would be replaced with API calls or local model loading
                
                # Simulate API call
                encoded_image = self.encode_image(image)
                
                # This is simulation code; replace with actual API call or model loading in production
                markdown_result = self._simulate_qwen_response(prompt)
                return markdown_result
                
            elif self.model_name == "minicpm":
                # Example code for using MiniCPM model
                encoded_image = self.encode_image(image)
                
                # This is simulation code; replace with actual API call or model loading in production
                markdown_result = self._simulate_minicpm_response(prompt)
                return markdown_result
                
            else:
                raise ValueError(f"Unimplemented open source model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Open source model call failed: {str(e)}")
            raise
    
    def call_commercial_model(self, image: Image.Image, prompt: str) -> str:
        """
        Call commercial VLM model API
        
        Args:
            image: Preprocessed image
            prompt: Prompt text
            
        Returns:
            Markdown text generated by the model
        """
        logger.info(f"Calling commercial model API: {self.model_name}")
        
        try:
            encoded_image = self.encode_image(image)
            
            if self.model_name == "gpt-4o-mini":
                # OpenAI GPT-4o Mini API call
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 4000
                }
                
                # This is simulation code; replace with actual API call in production
                # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                # response_data = response.json()
                # return response_data["choices"][0]["message"]["content"]
                
                # Simulated response
                return self._simulate_gpt4o_response(prompt)
                
            elif self.model_name == "claude-3.5":
                # Anthropic Claude 3.5 API call
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key
                }
                
                payload = {
                    "model": "claude-3-5-sonnet-20240620",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded_image}}
                            ]
                        }
                    ],
                    "max_tokens": 4000
                }
                
                # This is simulation code; replace with actual API call in production
                # response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                # response_data = response.json()
                # return response_data["content"][0]["text"]
                
                # Simulated response
                return self._simulate_claude_response(prompt)
                
            else:
                raise ValueError(f"Unimplemented commercial model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Commercial model API call failed: {str(e)}")
            raise
    
    def convert(self, image_path: str) -> str:
        """
        Convert receipt image to Markdown format
        
        Args:
            image_path: Path to receipt image
            
        Returns:
            Receipt content in Markdown format
        """
        logger.info(f"Starting receipt conversion: {image_path}")
        
        # 1. Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # 2. Generate prompt
        prompt = self.generate_prompt()
        
        # 3. Call appropriate processing function based on model type
        if self.model_name in ["qwen2-vl-2b", "minicpm"]:
            markdown_result = self.call_open_source_model(processed_image, prompt)
        else:  # Commercial models
            markdown_result = self.call_commercial_model(processed_image, prompt)
        
        # 4. Postprocess Markdown (optional)
        markdown_result = self.postprocess_markdown(markdown_result)
        
        logger.info("Receipt conversion completed")
        return markdown_result
    
    def postprocess_markdown(self, markdown: str) -> str:
        """
        Postprocess generated Markdown to ensure format consistency
        
        Args:
            markdown: Original generated Markdown text
            
        Returns:
            Processed Markdown text
        """
        # Add postprocessing logic here, such as:
        # - Fix table formatting
        # - Standardize heading levels
        # - Remove excessive blank lines
        # - Ensure consistency of key information formatting
        
        # Simple postprocessing example
        lines = markdown.split('\n')
        processed_lines = []
        
        for line in lines:
            # Remove excessive blank lines (keep only one consecutive blank line)
            if not line.strip() and processed_lines and not processed_lines[-1].strip():
                continue
                
            # Ensure table formatting is correct
            if '|' in line and '-|-' not in line and not any(line.startswith(p) for p in ['#', '>', '```']):
                # Might be a table row with incorrect formatting, try to fix
                cells = [cell.strip() for cell in line.split('|')]
                processed_lines.append('| ' + ' | '.join(cells) + ' |')
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    # The following are simulation methods for demonstration purposes only
    # In actual use, these should be replaced with real API calls or model loading
    
    def _simulate_qwen_response(self, prompt: str) -> str:
        """Simulate Qwen2-VL-2B response"""
        return """# Starbucks Coffee

## Merchant Information
**Address**: 17 Songzhi Road, Xinyi District, Taipei
**Phone**: (02) 2723-5857

## Transaction Information
**Date**: 2023-05-15
**Time**: 14:32:45
**Receipt Number**: #ST-7845-9261

## Items
| Item | Quantity | Unit Price | Subtotal |
|------|----------|------------|----------|
| Iced Americano | 1 | $120 | $120 |
| Cinnamon Roll | 2 | $85 | $170 |
| Caramel Macchiato | 1 | $145 | $145 |

## Price Summary
**Subtotal**: $435
**Tax (5%)**: $21.75
**Total**: $456.75

## Payment Information
**Payment Method**: VISA
**Card Last Four**: 5678
**Authorization Code**: 123456

## Additional Information
Thank you for your visit!
Join Starbucks Rewards through LINE to earn stars
"""
    
    def _simulate_minicpm_response(self, prompt: str) -> str:
        """Simulate MiniCPM response"""
        return """# FamilyMart

## Merchant Information
**Store Name**: FamilyMart (Fuxing Branch)
**Address**: No. 122, Sec. 1, Fuxing S. Rd., Da'an Dist., Taipei City
**Tax ID**: 28977386

## Transaction Information
**Date**: 2023-06-22
**Time**: 12:15:36
**Terminal/Transaction No.**: 01-56789

## Items
| Item | Quantity | Unit Price | Subtotal |
|------|----------|------------|----------|
| Signature Sandwich | 1 | $45 | $45 |
| Mineral Water | 2 | $20 | $40 |
| Rice Ball - Salmon | 1 | $35 | $35 |

## Price Summary
**Subtotal**: $120
**Tax (5%)**: $6
**Total**: $126

## Payment Information
**Payment Method**: Cash
**Received**: $200
**Change**: $74

## Additional Information
**Bonus Points**: 12 points earned this time
**Accumulated Points**: 156 points
"""
    
    def _simulate_gpt4o_response(self, prompt: str) -> str:
        """Simulate GPT-4o Mini response"""
        return """# Carrefour Supermarket

## Merchant Information
**Store Name**: Carrefour (Neihu Store)
**Address**: No. 128, Sec. 1, Jiuzong Rd., Neihu Dist., Taipei City
**Phone**: (02) 2627-1899
**Tax ID**: 16048502

## Transaction Information
**Date**: 2023-07-10
**Time**: 18:45:22
**Receipt Number**: CR-2023071056782

## Items
| Item | Quantity | Unit Price | Subtotal |
|------|----------|------------|----------|
| Organic Vegetable Mix | 1 | $199 | $199 |
| Chicken Breast (500g) | 2 | $150 | $300 |
| Chocolate Cookies | 3 | $45 | $135 |
| Shampoo | 1 | $180 | $180 |
| Toilet Paper (12 rolls) | 1 | $220 | $220 |

## Price Summary
**Subtotal**: $1,034
**Member Discount**: -$50
**Tax (5%)**: $49.2
**Total**: $1,033.2

## Payment Information
**Payment Method**: Carrefour Co-branded Card
**Card Last Four**: 1234
**Bonus Points**: 103 points earned

## Additional Information
**Member ID**: 76543210
**Return Policy**: Please return within 7 days with receipt
**Next Promotion**: $100 off on $1,500 purchase, valid until 2023/7/31
"""
    
    def _simulate_claude_response(self, prompt: str) -> str:
        """Simulate Claude 3.5 response"""
        return """# McDonald's

## Merchant Information
**Store Name**: McDonald's (Taipei Main Station)
**Address**: 1F, No. 47, Sec. 1, Zhongxiao W. Rd., Zhongzheng Dist., Taipei City
**Phone**: (02) 2311-1234
**Tax ID**: 22520085

## Transaction Information
**Date**: 2023-08-05
**Time**: 19:23:45
**Order Number**: #1087-2546

## Items
| Item | Quantity | Unit Price | Subtotal |
|------|----------|------------|----------|
| Big Mac Meal | 1 | $159 | $159 |
| McNuggets (6 pcs) | 1 | $99 | $99 |
| Coca-Cola (Large) | 1 | $45 | $45 |
| Fries Upsized | 1 | $30 | $30 |

## Price Summary
**Subtotal**: $333
**Tax (5%)**: $16.65
**Total**: $349.65

## Payment Information
**Payment Method**: LINE Pay
**Transaction ID**: LP78901234
**Discount**: McDonald's APP coupon used -$30

## Additional Information
**Dine-in/Takeout**: Dine-in
**Table Number**: 15
**McDonald's Points**: 33 points earned
**Coupon**: Free medium fries with next Big Mac purchase (Valid until: 2023/8/20)
"""


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Convert receipt images to Markdown format')
    parser.add_argument('--image', type=str, required=True, help='Path to receipt image')
    parser.add_argument('--model', type=str, default='qwen2-vl-2b', 
                        choices=['qwen2-vl-2b', 'minicpm', 'gpt-4o-mini', 'claude-3.5'],
                        help='VLM model to use')
    parser.add_argument('--api-key', type=str, help='API key (required for commercial models)')
    parser.add_argument('--output', type=str, help='Path to output Markdown file')
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = ReceiptToMarkdownConverter(model_name=args.model, api_key=args.api_key)
        
        # Convert receipt
        markdown_result = converter.convert(args.image)
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(markdown_result)
            print(f"Markdown saved to: {args.output}")
        else:
            print(markdown_result)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
