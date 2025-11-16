from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque, OrderedDict
import statistics
import hashlib
import re
import threading
from typing import Union

from ..config import SoloConfig

# ... full file content as read earlier ...
