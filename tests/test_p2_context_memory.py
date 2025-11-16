#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2 阶段测试：上下文收集与记忆管理增强功能
"""

import asyncio
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from solo_mcp.config import SoloConfig
from solo_mcp.tools.context import ContextTool

@pytest.mark.asyncio
async def test_smart_context_collection():
    config = SoloConfig.load(Path.cwd())
    context = ContextTool(config)
    ctx_full = await context.collect_context("implement python function", max_items=5)
    assert 'trim_info' in ctx_full
    assert isinstance(ctx_full.get('trim_info', {}).get('metadata', {}).get('kept_reasons', {}), dict)
