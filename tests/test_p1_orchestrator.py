#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 P1 阶段：多角色任务分配与冲突检测功能
"""

import asyncio
import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solo_mcp.config import SoloConfig
from solo_mcp.tools.roles import RolesTool
from solo_mcp.tools.memory import MemoryTool
from solo_mcp.tools.orchestrator import OrchestratorTool

@pytest.mark.asyncio
async def test_orchestrator_p1():
    print("=== 测试 P1 阶段：多角色任务分配与冲突检测 ===")
    config = SoloConfig.load(Path.cwd())
    roles_tool = RolesTool(config)
    memory_tool = MemoryTool(config)
    orchestrator = OrchestratorTool(config)
    state1 = {
        "goal": "Build a modern web frontend application with React",
        "stack": ["javascript", "react"],
        "history": []
    }
    result1 = await orchestrator.run_round("collab", state1)
    assert 'actions' in result1
    assert isinstance(result1.get('actions', []), list)
    assert 'execution_plan' in result1
    assert isinstance(result1.get('execution_plan', {}), dict)
