# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/share/ml/data/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/home/ml/github/openllm-ocr-annotator/data/outputs/foreign_trade_20250519/dataset"

    dataset = datasets.load_from_disk(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
    )
    # Model specific templates - will override default if specified

    system = """
      You are an expert in foreign trade document analysis. Your task is to extract key information
      from Chinese foreign trade documents with high precision. Pay special attention to:
      1. Document identifiers and numbers
      2. Dates in standard formats
      3. Company names and addresses
      4. Transaction amounts and currencies
      5. Geographic information
    """
    user = """
        Analyze this foreign trade document and extract the following specific fields:"

        Required Fields:
        1. Document Number
            - Format: Any alphanumeric identifier EXACTLY as shown in the document
            - Example: "CONTRACT-2024-001" or "FT20240101"
            - Note: Do not modify or normalize the number
            - Use "N/A" if no clear document number is found or if it's partially illegible

        2. Contract Date
            - Format: yyyy-mm-dd
            - Example: "2024-01-01"

        3. Buyer's Name
            - Format: Full company/entity name
            - Example: "ABC Trading Company Ltd."
            - Note: As much as possible keep the original name.

        4. Buyer's Country/Region
            - Format: Three-letter country code (ISO 3166-1 alpha-3)
            - Example: "USA" for United States, "GBR" for United Kingdom
            - Note: Infer from address or contact details if not explicitly stated

        5. Transaction Amount
            - Format: Number with 2 decimal places + three-letter currency code
            - Example: "123,44.50 USD" or "987,446.54 EUR"

        Important Rules for Missing or Unclear Information:
        1. Missing Values:
            - ALWAYS use exactly "N/A" (not "NA", "n/a", "null", or empty string)
            - Do not try to guess or infer values if you are not highly confident
            - Example: If no document number is found, use "N/A"

        2. Partial Information:
            - If only part of the information is available, use "N/A" rather than incomplete data
            - Example: If date is "2024-xx-xx", use "N/A" instead of partial date

        3. Unclear or Ambiguous Values:
            - When multiple possible values exist, use "N/A" rather than guessing
            - When text is illegible or unclear, use "N/A"

        4. Confidence Scores:
            - Use 0.99 for clear, directly stated values
            - Use 0.85-0.95 for inferred values (e.g., country codes from addresses)
            - Use 0.50 when returning "N/A" due to missing/unclear information

        5. Additional Guidelines:
            - For country codes: only use ISO 3166-1 alpha-3, otherwise use "N/A"
            - For currency codes: only use ISO 4217, otherwise use "N/A"
            - For dates: only use yyyy-mm-dd format, otherwise use "N/A"

        Return the results in this exact JSON format:
        {
            "fields": [
            {
                "field_name": "document_number",
                "value": "CONTRACT-2024-001",
                "confidence": 0.99
            },
            {
                "field_name": "contract_date",
                "value": "N/A",
                "confidence": 0.50
            },
            {
                "field_name": "buyer_name",
                "value": "ABC Trading Company Ltd.",
                "confidence": 0.99
            },
            {
                "field_name": "buyer_country",
                "value": "USA",
                "confidence": 0.85
            },
            {
                "field_name": "transaction_amount",
                "value": "123,456.00 USD",
                "confidence": 0.99
            }
            ],
            "metadata": {
            "document_type": "foreign_trade",
            "language": "English"
            }
        }
        """

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        import io  # 导入 io 模块用于处理字节流

        from PIL import Image  # 导入 Pillow 库

        # 修正 maximum_size 和 max_pixels 的定义，确保它们是整数而不是元组
        maximum_size: int = 20 * 1024 * 1024  # 20MB
        max_pixels: int = 178956970

        def process_single_image(img: Image.Image, image_idx: int) -> Image.Image:
            # 1. 检查并处理最大像素限制
            current_pixels = img.width * img.height
            if current_pixels > max_pixels:
                # 计算缩放比例以适应最大像素，同时保持宽高比
                ratio = (max_pixels / current_pixels) ** 0.5
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)  # LANCZOS 是一种高质量的缩放滤镜
                print(f"Image {image_idx}: Resized due to max_pixels. New dimensions: {img.width}x{img.height}")

            # 2. 检查并处理最大文件大小限制
            # 将图片转换为 RGB 模式，因为 JPEG 不支持 RGBA (alpha 通道)
            if img.mode not in ("RGB", "L"):  # 'L' 是灰度模式，JPEG 也支持
                img = img.convert("RGB")

            img_byte_arr = io.BytesIO()
            quality = 90  # 初始 JPEG 压缩质量

            # 尝试保存并检查大小，如果过大则降低质量或进一步缩放
            while True:
                img_byte_arr.seek(0)  # 重置缓冲区位置
                img_byte_arr.truncate(0)  # 清空缓冲区
                try:
                    img.save(img_byte_arr, format="JPEG", quality=quality, optimize=True)
                except Exception as e:
                    print(f"Warning: Image {image_idx} failed to save with quality {quality}: {e}")
                    # 如果保存失败，可能需要跳过或进一步处理
                    return
                try:
                    current_size = img_byte_arr.tell()
                except Exception as e:
                    print(f"Warning: Image {image_idx} failed to get size: {e}")
                    return None

                if current_size <= maximum_size:
                    break  # 符合大小要求，退出循环

                if quality > 10:  # 避免质量过低
                    quality -= 5  # 降低质量
                    # print(f"Image {image_idx}: Reducing JPEG quality to {quality}. Current size: {current_size / (1024*1024):.2f}MB")
                else:
                    # 质量已降无可降，尝试进一步缩放
                    # 计算新的缩放因子，使其大小接近 maximum_size
                    resize_factor = (maximum_size / current_size) ** 0.5 * 0.9  # 乘以0.9留一点余量
                    new_width = int(img.width * resize_factor)
                    new_height = int(img.height * resize_factor)
                    if new_width < 1 or new_height < 1:  # 避免尺寸过小
                        print(f"Warning: Image {image_idx} became too small during resizing. Skipping further size reduction.")
                        break
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    quality = 75  # 缩放后重置一个中等质量
                    # print(f"Image {image_idx}: Further resized. New dimensions: {img.width}x{img.height}")

            if current_size > maximum_size:
                print(f"Warning: Image {image_idx} still exceeds maximum_size ({maximum_size / (1024 * 1024):.2f}MB) after processing. Final size: {current_size / (1024 * 1024):.2f}MB")

            return img

        def process_fn(example, idx):
            prompt = user
            answer = example.pop("fields")
            raw_image = example.pop("image")  # 获取原始图片

            # 对图片进行处理
            processed_image = process_single_image(raw_image, idx)
            images = [processed_image]  # 保持原始结构，将处理后的图片放入列表中

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,  # 使用处理后的图片列表
                "ability": "document_analysis",  # 根据任务类型，将 "math" 修改为更合适的 "document_analysis"
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
