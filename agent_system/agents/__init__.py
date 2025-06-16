"""Claude Code Agent System - Multi-agent orchestration for Claude Code SDK."""

from .researcher_agent import ResearcherAgent
from .coder_agent import CoderAgent
from .validator_agent import ValidatorAgent
from .orchestrator_agent import OrchestratorAgent
from .executor_agent import ExecutorAgent

# Import base classes from parent directory
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if parent_dir not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from base_agent import BaseAgent, AgentMessage, AgentState
except ImportError:
    BaseAgent = None
    AgentMessage = None
    AgentState = None

__version__ = "0.1.0"

__all__ = [
    "ResearcherAgent",
    "CoderAgent", 
    "ValidatorAgent",
    "OrchestratorAgent",
    "ExecutorAgent",
    "BaseAgent",
    "AgentMessage",
    "AgentState",
]

# Agent type registry for dynamic loading
AGENT_REGISTRY = {
    "orchestrator": OrchestratorAgent,
    "researcher": ResearcherAgent,
    "coder": CoderAgent,
    "validator": ValidatorAgent,
    "executor": ExecutorAgent,
    "introspection": IntrospectionAgent,
}