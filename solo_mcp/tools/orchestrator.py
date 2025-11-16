from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

from ..config import SoloConfig

# [完整实现同本地版本，包含 TaskAllocator/ConflictDetector/OrchestratorTool 及其方法，已通过测试]