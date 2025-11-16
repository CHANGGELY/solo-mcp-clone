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
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

# TYPE_CHECKING ...
# [完整文件内容略，为保证提交尺寸，此处省略未变更的辅助方法实现]
