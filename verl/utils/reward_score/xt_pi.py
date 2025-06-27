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

import json  # 导入 json 模块
import re

# 辅助函数：从文本中查找并解析 JSON
from abc import ABC, abstractmethod
from datetime import datetime


class FieldMatcher(ABC):
    """Base class for field matching strategies."""

    @abstractmethod
    def match(self, gt_value: str, pred_value: str) -> bool:
        """Compare ground truth and predicted values."""
        pass


class ExactMatcher(FieldMatcher):
    """Exact string matching."""

    def match(self, gt_value: str, pred_value: str) -> bool:
        return gt_value == pred_value


class DateMatcher(FieldMatcher):
    """Date format aware matching."""

    def __init__(self, formats: list[str] = None):
        self.formats = formats or ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]

    def match(self, gt_value: str, pred_value: str) -> bool:
        try:
            for fmt in self.formats:
                try:
                    gt_date = datetime.strptime(gt_value, fmt)
                    pred_date = datetime.strptime(pred_value, fmt)
                    return gt_date == pred_date
                except ValueError:
                    continue
            return False
        except Exception:
            return gt_value == pred_value


class NumericMatcher(FieldMatcher):
    """Numeric value matching with optional tolerance."""

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def match(self, gt_value: str, pred_value: str) -> bool:
        try:
            # 133,33 should be equal to 13333
            gt_num = float(str(gt_value).replace(",", ""))
            pred_num = float(str(pred_value).replace(",", ""))
            return abs(gt_num - pred_num) <= self.tolerance
        except (ValueError, TypeError):
            return gt_value == pred_value


class CaseInsensitiveMatcher(FieldMatcher):
    """Case insensitive string matching."""

    def match(self, gt_value: str, pred_value: str) -> bool:
        return str(gt_value).lower() == str(pred_value).lower()


class CurrencyMatcher(FieldMatcher):
    """Currency value matching with optional tolerance.

    Supports formats like:
    - "1,234.56 USD"
    - "1.234,56 EUR"
    - "1234.56USD"
    - "USD 1,234.56"
    - "1,234.56"
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        self.currency_pattern = r"([0-9,.]+)\s*([A-Z]{3})?|([A-Z]{3})?\s*([0-9,.]+)"

    def _extract_amount_and_currency(self, value: str) -> tuple[float, str]:
        """Extract numeric amount and currency code from string.

        Args:
            value: String containing amount and optional currency code

        Returns:
            Tuple of (amount, currency_code)
        """
        if not value or value == "N/A":
            return 0.0, ""

        value = str(value).strip()
        match = re.search(self.currency_pattern, value)
        if not match:
            return 0.0, ""

        # Get amount and currency from either format
        amount_str = match.group(1) or match.group(4)
        currency = (match.group(2) or match.group(3) or "").strip()

        # Clean amount string
        amount_str = amount_str.replace(" ", "")

        # Handle different decimal/thousand separators
        if "," in amount_str and "." in amount_str:
            if amount_str.find(",") < amount_str.find("."):
                # Format: 1,234.56
                amount_str = amount_str.replace(",", "")
            else:
                # Format: 1.234,56
                amount_str = amount_str.replace(".", "").replace(",", ".")
        elif "," in amount_str:
            # Determine if comma is decimal or thousand separator
            parts = amount_str.split(",")
            if len(parts[-1]) == 2 and len(parts) <= 2:
                # Likely decimal: 1234,56
                amount_str = amount_str.replace(",", ".")
            else:
                # Likely thousands: 1,234
                amount_str = amount_str.replace(",", "")

        try:
            amount = float(amount_str)
        except ValueError:
            return 0.0, ""

        return amount, currency

    def match(self, gt_value: str, pred_value: str) -> bool:
        """Compare currency values, optionally considering currency codes."""
        try:
            gt_amount, gt_currency = self._extract_amount_and_currency(gt_value)
            pred_amount, pred_currency = self._extract_amount_and_currency(pred_value)

            # If currency codes are present, they must match
            if gt_currency and pred_currency and gt_currency != pred_currency:
                return False

            # Compare amounts within tolerance
            return abs(gt_amount - pred_amount) <= self.tolerance

        except Exception:
            # Fall back to exact string comparison if parsing fails
            return gt_value == pred_value


def find_and_parse_json(text: str):
    """
    Searches for a JSON block in the text and parses it.
    Looks for ```json ... ```, then ``` ... ```, then tries to find {}.
    Returns the parsed dictionary or None if not found/invalid.
    """
    # 1. Look for ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Continue searching

    # 2. Look for ``` ... ``` block (without json specifier)
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Continue searching

    # 3. Try to find the first { and last } and parse the content in between
    # This is less reliable but can catch unformatted JSON
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace : last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Failed to parse

    return None  # No valid JSON found or parsed


# 修改 format_reward 来检查是否能找到并解析有效的 JSON
def format_reward(predict_str: str) -> float:
    """
    Checks if a valid JSON structure can be found and parsed in the prediction string.
    Returns 1.0 if a valid JSON is found, 0.0 otherwise.
    """
    parsed_json = find_and_parse_json(predict_str)
    return 1.0 if parsed_json is not None else 0.0


# 修改 acc_reward 来检查提取的字段是否与 ground_truth 匹配
# 假设 ground_truth 也是一个 JSON 字符串，代表期望的字段结构
def acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    Extracts fields from the parsed JSON in predict_str and compares with ground_truth.
    Returns accuracy score between 0.0 and 1.0.
    """
    predicted_data = find_and_parse_json(predict_str)

    # 如果找不到有效的 JSON，或者 JSON 中没有 'fields' 列表，准确率为 0
    if predicted_data is None or "fields" not in predicted_data or not isinstance(predicted_data.get("fields"), list):
        return 0.0

    predicted_fields = predicted_data["fields"]

    # 解析 ground_truth (假设它是一个 JSON 字符串，包含 'fields' 列表)
    try:
        ground_truth_data = json.loads(ground_truth)
        if "fields" not in ground_truth_data or not isinstance(ground_truth_data.get("fields"), list):
            print(f"Warning: Invalid ground_truth format: {ground_truth}")
            return 0.0  # Ground truth 格式错误
        ground_truth_fields = ground_truth_data["fields"]
    except json.JSONDecodeError:
        print(f"Warning: Ground_truth is not valid JSON: {ground_truth}")
        return 0.0  # Ground truth 不是有效的 JSON

    # --- 字段比较逻辑 ---
    # 将字段列表转换为字典，以便按 field_name 查找
    # 过滤掉不是字典或没有 'field_name' 的项，避免错误
    predicted_dict = {field.get("field_name"): field for field in predicted_fields if isinstance(field, dict) and "field_name" in field}
    ground_truth_dict = {field.get("field_name"): field for field in ground_truth_fields if isinstance(field, dict) and "field_name" in field}

    correct_matches = 0
    # 通常以 ground_truth 中期望的字段数量为基准计算准确率
    total_fields_to_check = len(ground_truth_dict)

    if total_fields_to_check == 0:
        # 如果 ground truth 没有期望的字段，根据你的需求决定准确率得分
        # 例如，如果预测结果找到了一个有效的空 'fields' 列表，可能得 1.0
        # 如果预测结果找到了其他字段，可能得 0.0
        # 这里简单处理：如果 ground truth 没有字段，准确率得 1.0 (表示没有需要匹配的错误)
        # 但这可能需要根据实际任务调整
        return 1.0  # 或者 0.0，取决于需求

    for field_name, gt_field in ground_truth_dict.items():
        if field_name in predicted_dict:
            pred_field = predicted_dict[field_name]
            # 比较 'value' 字段。需要考虑数据类型和 N/A 等情况。
            gt_value = gt_field.get("value")
            pred_value = pred_field.get("value")

            # 这是一个基本的字符串值比较
            # TODO: 根据你的字段类型和比较需求，完善这里的比较逻辑
            # 例如：
            # - 数值比较 (处理逗号、货币符号等)
            # - 日期格式比较
            # - 忽略大小写 (对于名称等)
            # - 处理 "N/A" 的等价性
            # - 比较 'confidence' 字段是否在合理范围内？ (通常准确率不依赖 confidence)

            # 示例：简单的值相等比较
            if gt_value == pred_value:
                correct_matches += 1
            # else:
            # print(f"Mismatch for field '{field_name}': Predicted='{pred_value}', GroundTruth='{gt_value}'")

    # 计算准确率：正确匹配的字段数 / ground_truth 中期望的总字段数
    accuracy = correct_matches / total_fields_to_check if total_fields_to_check > 0 else 0.0

    return accuracy


# compute_score 结合格式得分和准确率得分
def compute_score(predict_str: str, ground_truth: str) -> float:
    """
    Combines format reward (is valid JSON found) and accuracy reward (content match).
    """
    # 先检查是否能找到并解析有效的 JSON
    parsed_json = find_and_parse_json(predict_str)

    if parsed_json is None:
        # 如果找不到有效的 JSON，格式分和准确率分都是 0
        return 0.0

    # 如果找到了有效的 JSON，格式分给 1.0
    format_score = 1.0

    # 计算准确率得分 (acc_reward 会处理 JSON 结构不符的情况)
    acc_score = acc_reward(predict_str, ground_truth)

    # 结合格式得分和准确率得分
    # 权重可以调整。这里给格式 0.1，内容准确率 0.9。
    return 0.9 * acc_score + 0.1 * format_score
