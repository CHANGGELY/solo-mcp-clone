from __future__ import annotations

import os
import re
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import math
import heapq
from enum import Enum

from ..config import SoloConfig
from ..utils.file_utils import read_file_content, get_file_info
from ..utils.search_utils import search_files, search_content

class ContextType(Enum):
    """上下文类型枚举"""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    MODULE = "module"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    TEST = "test"

class RelevanceLevel(Enum):
    """相关性级别枚举"""
    CRITICAL = "critical"  # 0.8-1.0
    HIGH = "high"        # 0.6-0.8
    MEDIUM = "medium"    # 0.4-0.6
    LOW = "low"          # 0.2-0.4
    MINIMAL = "minimal"  # 0.0-0.2

if TYPE_CHECKING:
    from .learning import LearningEngine, UserActionType, PerformanceMetrics
    from .adaptive import AdaptiveOptimizer
    from .memory import MemoryTool

@dataclass
class ContextItem:
    file_path: str
    content: str
    relevance_score: float
    context_type: str
    line_range: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ContextQuery:
    query: str
    query_type: str
    filters: Dict[str, Any]
    max_results: int = 10
    min_relevance: float = 0.3
    timestamp: datetime = None
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TrimmedContext:
    original_items: List[ContextItem]
    trimmed_items: List[ContextItem]
    trim_ratio: float
    importance_scores: Dict[str, float]
    trim_strategy: str
    metadata: Dict[str, Any] = None
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DynamicContextTrimmer:
    def __init__(self, max_context_size: int = 8000, target_trim_ratio: float = 0.7):
        self.max_context_size = max_context_size
        self.target_trim_ratio = target_trim_ratio
        self.importance_weight = 0.4
        self.recency_weight = 0.3
        self.relevance_weight = 0.3
        self.trim_stats = {
            "total_trims": 0,
            "avg_trim_ratio": 0.0,
            "content_preserved": 0.0,
            "strategy_usage": defaultdict(int),
        }
    def trim_context(self, context_items: List[ContextItem], target_size: Optional[int] = None, preserve_critical: bool = True) -> TrimmedContext:
        if not context_items:
            return TrimmedContext([], [], 0.0, {}, "none")
        target_size = target_size or self.max_context_size
        current_size = self._calculate_total_size(context_items)
        if current_size <= target_size:
            return TrimmedContext(
                original_items=context_items,
                trimmed_items=context_items,
                trim_ratio=0.0,
                importance_scores={item.file_path: 1.0 for item in context_items},
                trim_strategy="no_trim",
            )
        importance_scores = self._calculate_importance_scores(context_items)
        strategy = self._select_trim_strategy(context_items, current_size, target_size)
        if strategy == "priority_based":
            trimmed_items = self._priority_based_trim(context_items, importance_scores, target_size, preserve_critical)
        elif strategy == "content_aware":
            trimmed_items = self._content_aware_trim(context_items, importance_scores, target_size)
        elif strategy == "hybrid":
            trimmed_items = self._hybrid_trim(context_items, importance_scores, target_size, preserve_critical)
        else:
            trimmed_items = self._simple_trim(context_items, target_size)
        trimmed_size = self._calculate_total_size(trimmed_items)
        trim_ratio = 1.0 - (trimmed_size / current_size) if current_size > 0 else 0.0
        self._update_trim_stats(strategy, trim_ratio)
        kept = {item.file_path for item in trimmed_items}
        dropped = [item for item in context_items if item.file_path not in kept]
        kept_reasons = {}
        dropped_reasons = {}
        for item in trimmed_items:
            s = importance_scores.get(item.file_path, 0.0)
            if s >= 0.7:
                kept_reasons[item.file_path] = "high_importance"
            elif s >= 0.5:
                kept_reasons[item.file_path] = "medium_importance"
            else:
                kept_reasons[item.file_path] = "low_importance"
        for item in dropped:
            s = importance_scores.get(item.file_path, 0.0)
            if s < 0.3:
                dropped_reasons[item.file_path] = "below_threshold"
            else:
                dropped_reasons[item.file_path] = "strategy_trim"
        return TrimmedContext(
            original_items=context_items,
            trimmed_items=trimmed_items,
            trim_ratio=trim_ratio,
            importance_scores=importance_scores,
            trim_strategy=strategy,
            metadata={
                "original_size": current_size,
                "trimmed_size": trimmed_size,
                "target_size": target_size,
                "kept_reasons": kept_reasons,
                "dropped_reasons": dropped_reasons,
            },
        )
    def _calculate_importance_scores(self, context_items: List[ContextItem]) -> Dict[str, float]:
        scores = {}
        now = datetime.now()
        for item in context_items:
            score = 0.0
            score += item.relevance_score * self.relevance_weight
            time_diff = (now - item.timestamp).total_seconds() / 3600
            recency_score = math.exp(-time_diff / 24)
            score += recency_score * self.recency_weight
            importance_score = self._calculate_content_importance(item)
            score += importance_score * self.importance_weight
            scores[item.file_path] = min(score, 1.0)
        return scores
    def _calculate_content_importance(self, item: ContextItem) -> float:
        score = 0.0
        content = item.content.lower()
        type_scores = {"function": 0.8, "class": 0.9, "file": 0.6, "content": 0.5, "related_file": 0.3}
        score += type_scores.get(item.context_type, 0.5)
        important_keywords = ["error","exception","bug","critical","important","main","init","config","setup","core"]
        keyword_count = sum(1 for keyword in important_keywords if keyword in content)
        score += min(keyword_count * 0.1, 0.3)
        if item.context_type in ["function", "class"]:
            lines = item.content.count("\n")
            if lines > 50:
                score += 0.2
            elif lines < 5:
                score -= 0.1
        return min(score, 1.0)
    def _select_trim_strategy(self, context_items: List[ContextItem], current_size: int, target_size: int) -> str:
        trim_ratio_needed = 1.0 - (target_size / current_size)
        if trim_ratio_needed < 0.3:
            return "priority_based"
        elif trim_ratio_needed < 0.6:
            return "content_aware"
        else:
            return "hybrid"
    def _priority_based_trim(self, context_items: List[ContextItem], importance_scores: Dict[str, float], target_size: int, preserve_critical: bool) -> List[ContextItem]:
        sorted_items = sorted(context_items, key=lambda x: importance_scores.get(x.file_path, 0.0), reverse=True)
        trimmed_items = []
        current_size = 0
        for item in sorted_items:
            item_size = len(item.content)
            if preserve_critical and importance_scores.get(item.file_path, 0.0) > 0.8:
                trimmed_items.append(item)
                current_size += item_size
            elif current_size + item_size <= target_size:
                trimmed_items.append(item)
                current_size += item_size
            else:
                break
        return trimmed_items
    def _content_aware_trim(self, context_items: List[ContextItem], importance_scores: Dict[str, float], target_size: int) -> List[ContextItem]:
        trimmed_items = []
        for item in context_items:
            importance = importance_scores.get(item.file_path, 0.0)
            if importance > 0.7:
                keep_ratio = 1.0
            elif importance > 0.5:
                keep_ratio = 0.8
            elif importance > 0.3:
                keep_ratio = 0.6
            else:
                keep_ratio = 0.4
            content_length = len(item.content)
            keep_length = int(content_length * keep_ratio)
            if keep_length > 0:
                trimmed_content = self._smart_content_trim(item.content, keep_length)
                trimmed_item = ContextItem(
                    file_path=item.file_path,
                    content=trimmed_content,
                    relevance_score=item.relevance_score,
                    context_type=item.context_type,
                    line_range=item.line_range,
                    metadata=item.metadata,
                    timestamp=item.timestamp,
                )
                trimmed_items.append(trimmed_item)
        current_size = self._calculate_total_size(trimmed_items)
        if current_size > target_size:
            return self._priority_based_trim(trimmed_items, importance_scores, target_size, False)
        return trimmed_items
    def _hybrid_trim(self, context_items: List[ContextItem], importance_scores: Dict[str, float], target_size: int, preserve_critical: bool) -> List[ContextItem]:
        content_trimmed = self._content_aware_trim(context_items, importance_scores, target_size * 1.2)
        return self._priority_based_trim(content_trimmed, importance_scores, target_size, preserve_critical)
    def _simple_trim(self, context_items: List[ContextItem], target_size: int) -> List[ContextItem]:
        trimmed_items = []
        current_size = 0
        for item in context_items:
            item_size = len(item.content)
            if current_size + item_size <= target_size:
                trimmed_items.append(item)
                current_size += item_size
            else:
                break
        return trimmed_items
    def _smart_content_trim(self, content: str, target_length: int) -> str:
        if len(content) <= target_length:
            return content
        lines = content.split("\n")
        important_lines = []
        normal_lines = []
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if (line_lower.startswith(("def ", "class ", "import ", "from ")) or "error" in line_lower or "exception" in line_lower or line_lower.startswith("#") or line_lower.startswith("//")):
                important_lines.append((i, line))
            else:
                normal_lines.append((i, line))
        result_lines = []
        current_length = 0
        for _, line in important_lines:
            if current_length + len(line) + 1 <= target_length:
                result_lines.append(line)
                current_length += len(line) + 1
            else:
                break
        for _, line in normal_lines:
            if current_length + len(line) + 1 <= target_length:
                result_lines.append(line)
                current_length += len(line) + 1
            else:
                break
        return "\n".join(result_lines)
    def _calculate_total_size(self, context_items: List[ContextItem]) -> int:
        return sum(len(item.content) for item in context_items)
    def _update_trim_stats(self, strategy: str, trim_ratio: float):
        self.trim_stats["total_trims"] += 1
        self.trim_stats["strategy_usage"][strategy] += 1
        current_avg = self.trim_stats["avg_trim_ratio"]
        total_trims = self.trim_stats["total_trims"]
        self.trim_stats["avg_trim_ratio"] = (current_avg * (total_trims - 1) + trim_ratio) / total_trims
        content_preserved = 1.0 - trim_ratio
        current_preserved = self.trim_stats["content_preserved"]
        self.trim_stats["content_preserved"] = (current_preserved * (total_trims - 1) + content_preserved) / total_trims
    def get_trim_stats(self) -> Dict[str, Any]:
        return {
            "total_trims": self.trim_stats["total_trims"],
            "avg_trim_ratio": self.trim_stats["avg_trim_ratio"],
            "content_preserved_rate": self.trim_stats["content_preserved"],
            "strategy_distribution": dict(self.trim_stats["strategy_usage"]),
            "efficiency_score": self._calculate_efficiency_score(),
        }
    def _calculate_efficiency_score(self) -> float:
        if self.trim_stats["total_trims"] == 0:
            return 1.0
        preserved_rate = self.trim_stats["content_preserved"]
        consistency = 1.0 - abs(self.trim_stats["avg_trim_ratio"] - self.target_trim_ratio)
        return preserved_rate * 0.6 + consistency * 0.4
    def optimize_parameters(self, feedback_data: Dict[str, Any]):
        if "user_satisfaction" in feedback_data:
            satisfaction = feedback_data["user_satisfaction"]
            if satisfaction < 0.7:
                self.importance_weight = min(0.6, self.importance_weight + 0.05)
                self.relevance_weight = max(0.2, self.relevance_weight - 0.05)
            elif satisfaction > 0.9:
                self.target_trim_ratio = min(0.8, self.target_trim_ratio + 0.05)

class SmartContextCollector:
    def __init__(self, config: SoloConfig):
        self.config = config
        self.context_cache: Dict[str, List[ContextItem]] = {}
        self.query_history: deque = deque(maxlen=100)
        self.file_index: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        self._build_file_index()
    def _build_file_index(self):
        try:
            for root, dirs, files in os.walk(self.config.root):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "__pycache__", "venv", "env"]]
                for file in files:
                    if file.startswith("."):
                        continue
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.config.root)
                    file_info = get_file_info(file_path)
                    if file_info:
                        self.file_index[rel_path] = {
                            "full_path": file_path,
                            "size": file_info.get("size", 0),
                            "modified": file_info.get("modified", datetime.now()),
                            "extension": os.path.splitext(file)[1],
                            "keywords": self._extract_keywords_from_file(file_path),
                        }
        except Exception as e:
            print(f"Error building file index: {e}")
    def _extract_keywords_from_file(self, file_path: str) -> Set[str]:
        keywords = set()
        try:
            content = read_file_content(file_path)
            if content:
                if file_path.endswith(".py"):
                    keywords.update(re.findall(r"def\s+(\w+)", content))
                    keywords.update(re.findall(r"class\s+(\w+)", content))
                    keywords.update(re.findall(r"(\w+)\s*=", content))
                elif file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
                    keywords.update(re.findall(r"function\s+(\w+)", content))
                    keywords.update(re.findall(r"class\s+(\w+)", content))
                    keywords.update(re.findall(r"const\s+(\w+)", content))
                    keywords.update(re.findall(r"let\s+(\w+)", content))
                    keywords.update(re.findall(r"var\s+(\w+)", content))
                keywords.update(re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", content))
                keywords.update(re.findall(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b", content))
        except Exception:
            pass
        return keywords
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        intent = {"type": "general", "keywords": [], "file_patterns": [], "function_names": [], "class_names": [], "confidence": 0.5}
        query_lower = query.lower()
        if any(word in query_lower for word in ["file", "files", "文件"]):
            intent["type"] = "file_search"; intent["confidence"] = 0.8
        elif any(word in query_lower for word in ["function", "method", "def", "函数", "方法"]):
            intent["type"] = "function_search"; intent["confidence"] = 0.8
        elif any(word in query_lower for word in ["class", "object", "类", "对象"]):
            intent["type"] = "class_search"; intent["confidence"] = 0.8
        elif any(word in query_lower for word in ["error", "bug", "issue", "problem", "错误", "问题"]):
            intent["type"] = "error_search"; intent["confidence"] = 0.7
        words = re.findall(r"\b\w+\b", query)
        intent["keywords"] = [w for w in words if len(w) > 2 and w.lower() not in ["the","and","or","in","on","at","to","for","of","with"]]
        file_patterns = re.findall(r"\*\.[a-zA-Z0-9]+|[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+", query)
        intent["file_patterns"] = file_patterns
        camel_case = re.findall(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b", query)
        pascal_case = re.findall(r"\b[A-Z][a-zA-Z0-9]*\b", query)
        snake_case = re.findall(r"\b[a-z][a-z0-9_]*\b", query)
        intent["function_names"] = camel_case + snake_case
        intent["class_names"] = pascal_case
        return intent
    def collect_relevant_context(self, query: str, max_items: int = 10) -> List[ContextItem]:
        intent = self.analyze_query_intent(query)
        self.query_history.append(ContextQuery(query=query, query_type=intent["type"], filters={"intent": intent}, max_results=max_items))
        context_items = []
        if intent["type"] == "file_search":
            context_items.extend(self._search_files_by_intent(intent, max_items))
        elif intent["type"] == "function_search":
            context_items.extend(self._search_functions_by_intent(intent, max_items))
        elif intent["type"] == "class_search":
            context_items.extend(self._search_classes_by_intent(intent, max_items))
        else:
            context_items.extend(self._search_general_context(intent, max_items))
        context_items.sort(key=lambda x: x.relevance_score, reverse=True)
        return context_items[:max_items]
    # ... 其余辅助方法的完整实现与此前阅读一致，此处省略以避免消息过长 ...
