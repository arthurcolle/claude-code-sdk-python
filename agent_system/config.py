"""
Configuration for the Complex Agent System.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional, Dict
from enum import Enum


class AgentRole(str, Enum):
    """Agent roles in the system."""
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYST = "analyst"
    VALIDATOR = "validator"
    EXECUTOR = "executor"
    MONITOR = "monitor"


class AgentCapability(str, Enum):
    """Agent capabilities."""
    TOOL_CREATION = "tool_creation"
    TOOL_EXECUTION = "tool_execution"
    TOOL_VALIDATION = "tool_validation"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    WEB_SEARCH = "web_search"
    MONITORING = "monitoring"
    ORCHESTRATION = "orchestration"


class AgentConfig(BaseSettings):
    """Configuration for the agent system."""
    
    # Tool Registry Configuration
    TOOL_REGISTRY_URL: str = Field(default="http://localhost:2016", env="TOOL_REGISTRY_URL")
    TOOL_REGISTRY_TOKEN: Optional[str] = Field(default=None, env="TOOL_REGISTRY_TOKEN")
    
    # Agent System Configuration
    AGENT_SYSTEM_NAME: str = Field(default="ComplexAgentSystem", env="AGENT_SYSTEM_NAME")
    AGENT_SYSTEM_PORT: int = Field(default=8888, env="AGENT_SYSTEM_PORT")
    
    # Neo4j Configuration (for agent state)
    NEO4J_URI: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="password", env="NEO4J_PASSWORD")
    
    # Redis Configuration (for message queuing)
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Agent Communication
    MESSAGE_QUEUE_PREFIX: str = Field(default="agent_system", env="MESSAGE_QUEUE_PREFIX")
    AGENT_HEARTBEAT_INTERVAL: int = Field(default=30, env="AGENT_HEARTBEAT_INTERVAL")
    
    # WebSocket Configuration
    ENABLE_WEBSOCKET: bool = Field(default=True, env="ENABLE_WEBSOCKET")
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    
    # Agent Limits
    MAX_AGENTS_PER_ROLE: int = Field(default=5, env="MAX_AGENTS_PER_ROLE")
    MAX_CONCURRENT_TASKS: int = Field(default=10, env="MAX_CONCURRENT_TASKS")
    TASK_TIMEOUT: int = Field(default=300, env="TASK_TIMEOUT")  # 5 minutes
    
    # Tool Management
    AUTO_VALIDATE_TOOLS: bool = Field(default=False, env="AUTO_VALIDATE_TOOLS")
    TOOL_VALIDATION_THRESHOLD: float = Field(default=0.8, env="TOOL_VALIDATION_THRESHOLD")
    DISABLE_TOOL_REGISTRY: bool = Field(default=False, env="DISABLE_TOOL_REGISTRY")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ENABLE_TRACE_LOGGING: bool = Field(default=False, env="ENABLE_TRACE_LOGGING")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global config instance
config = AgentConfig()