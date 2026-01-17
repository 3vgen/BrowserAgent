"""
Agents module - AI агенты для автоматизации браузера
"""

from .simple_agent import SimpleAgent
from .vision_agent import VisionAgent, PageAnalysis
from .action_agent import ActionAgent, Action
from .orchestrator import Orchestrator

__all__ = [
    "SimpleAgent",
    "VisionAgent",
    "PageAnalysis",
    "ActionAgent",
    "Action",
    "Orchestrator"
]