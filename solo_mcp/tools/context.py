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
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    MODULE = "module"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    TEST = "test"

# ... full file content as read earlier ...
