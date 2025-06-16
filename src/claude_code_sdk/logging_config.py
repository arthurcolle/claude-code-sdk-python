"""
Configuration system for Claude Max advanced logging.

Provides flexible configuration management for game logging with:
- Environment-based configs
- Hot-reloading
- Remote configuration support
- A/B testing capabilities
"""

import os
import json
import yaml
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class LogLevel(Enum):
    """Log levels for the system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConfigSource(Enum):
    """Configuration sources."""
    FILE = "file"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    DEFAULT = "default"


@dataclass
class LoggingProfile:
    """Defines a logging profile for different scenarios."""
    name: str
    level: LogLevel
    enable_console: bool = True
    enable_file: bool = True
    enable_distributed: bool = False
    enable_metrics: bool = True
    enable_events: bool = True
    enable_replay: bool = False
    performance_tracking: bool = True
    anomaly_detection: bool = True
    compression: bool = True
    encryption: bool = False
    retention_days: int = 30
    sampling_rate: float = 1.0  # 1.0 = log everything, 0.1 = log 10%
    
    def should_log(self) -> bool:
        """Determine if we should log based on sampling rate."""
        import random
        return random.random() < self.sampling_rate


# Predefined profiles
PROFILES = {
    "development": LoggingProfile(
        name="development",
        level=LogLevel.DEBUG,
        enable_replay=True,
        compression=False,
        encryption=False,
        retention_days=7,
        sampling_rate=1.0
    ),
    "production": LoggingProfile(
        name="production",
        level=LogLevel.INFO,
        enable_distributed=True,
        compression=True,
        encryption=True,
        retention_days=90,
        sampling_rate=0.1  # Sample 10% in production
    ),
    "performance": LoggingProfile(
        name="performance",
        level=LogLevel.WARNING,
        enable_console=False,
        performance_tracking=True,
        anomaly_detection=True,
        enable_events=False,  # Disable event logging for performance
        sampling_rate=0.5
    ),
    "debug": LoggingProfile(
        name="debug",
        level=LogLevel.DEBUG,
        enable_replay=True,
        compression=False,
        retention_days=1,
        sampling_rate=1.0
    ),
    "minimal": LoggingProfile(
        name="minimal",
        level=LogLevel.ERROR,
        enable_file=False,
        enable_metrics=False,
        enable_events=False,
        performance_tracking=False,
        anomaly_detection=False,
        sampling_rate=0.01  # Only 1% of logs
    )
}


@dataclass
class MetricConfig:
    """Configuration for metric tracking."""
    enabled: bool = True
    flush_interval: float = 5.0  # seconds
    aggregation_window: int = 60  # seconds
    percentiles: List[float] = None
    thresholds: Dict[str, float] = None
    alerts: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = [0.5, 0.95, 0.99]
        if self.thresholds is None:
            self.thresholds = {
                "fps": 30.0,
                "latency": 100.0,
                "cpu_usage": 80.0,
                "memory_usage": 90.0
            }
        if self.alerts is None:
            self.alerts = {}


@dataclass
class EventConfig:
    """Configuration for event processing."""
    enabled: bool = True
    batch_size: int = 100
    flush_interval: float = 1.0
    max_queue_size: int = 10000
    processors: List[str] = None
    filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.processors is None:
            self.processors = ["analytics", "performance", "security"]
        if self.filters is None:
            self.filters = {}


class ConfigManager:
    """
    Manages logging configuration with multiple sources and hot-reloading.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("./config")
        self.configs: Dict[str, Any] = {}
        self.profile: Optional[LoggingProfile] = None
        self.callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._observer: Optional[Observer] = None
        
        # Load initial configuration
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from all sources."""
        with self._lock:
            # 1. Load defaults
            self.configs = self._get_default_config()
            
            # 2. Load from file
            file_config = self._load_file_config()
            if file_config:
                self._merge_config(self.configs, file_config)
            
            # 3. Load from environment
            env_config = self._load_env_config()
            if env_config:
                self._merge_config(self.configs, env_config)
            
            # 4. Load from remote (if configured)
            if self.configs.get("remote_config_enabled"):
                remote_config = self._load_remote_config()
                if remote_config:
                    self._merge_config(self.configs, remote_config)
            
            # 5. Set active profile
            profile_name = self.configs.get("profile", "development")
            self.profile = PROFILES.get(profile_name, PROFILES["development"])
            
            # 6. Apply A/B testing if enabled
            if self.configs.get("ab_testing_enabled"):
                self._apply_ab_testing()
            
            # Notify callbacks
            self._notify_callbacks()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "profile": "development",
            "game_id": "claude_max_game",
            "version": "1.0.0",
            "remote_config_enabled": False,
            "remote_config_url": None,
            "ab_testing_enabled": False,
            "metric_config": asdict(MetricConfig()),
            "event_config": asdict(EventConfig()),
            "storage": {
                "type": "local",
                "path": "./logs",
                "max_size_mb": 1000,
                "compression": True
            },
            "security": {
                "encrypt_logs": False,
                "encryption_key": None,
                "sanitize_pii": True,
                "allowed_ips": []
            },
            "performance": {
                "async_logging": True,
                "buffer_size": 1000,
                "worker_threads": 4
            }
        }
    
    def _load_file_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        config_files = [
            self.base_path / "logging.json",
            self.base_path / "logging.yaml",
            self.base_path / "logging.yml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix == '.json':
                            return json.load(f)
                        else:
                            return yaml.safe_load(f)
                except Exception as e:
                    print(f"Error loading config from {config_file}: {e}")
        
        return None
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            "CLAUDE_MAX_LOG_PROFILE": "profile",
            "CLAUDE_MAX_LOG_LEVEL": "log_level",
            "CLAUDE_MAX_GAME_ID": "game_id",
            "CLAUDE_MAX_REMOTE_CONFIG": "remote_config_enabled",
            "CLAUDE_MAX_REMOTE_CONFIG_URL": "remote_config_url",
            "CLAUDE_MAX_AB_TESTING": "ab_testing_enabled"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert string to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                
                env_config[config_key] = value
        
        return env_config
    
    def _load_remote_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from remote source."""
        url = self.configs.get("remote_config_url")
        if not url:
            return None
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error loading remote config: {e}")
            return None
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configurations."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _apply_ab_testing(self) -> None:
        """Apply A/B testing configuration."""
        # Simple A/B testing based on player ID hash
        player_id = self.configs.get("player_id", "default")
        test_group = int(hashlib.md5(player_id.encode()).hexdigest(), 16) % 100
        
        # Example: 20% of users get enhanced logging
        if test_group < 20:
            self.configs["ab_test_group"] = "enhanced"
            self.profile.enable_replay = True
            self.profile.sampling_rate = 1.0
        else:
            self.configs["ab_test_group"] = "standard"
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of config change."""
        for callback in self.callbacks:
            try:
                callback(self.configs, self.profile)
            except Exception as e:
                print(f"Error in config callback: {e}")
    
    def start_watching(self) -> None:
        """Start watching configuration files for changes."""
        if not self.base_path.exists():
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, config_manager):
                self.config_manager = config_manager
            
            def on_modified(self, event):
                if event.is_directory:
                    return
                
                if any(event.src_path.endswith(ext) for ext in ['.json', '.yaml', '.yml']):
                    print(f"Config file changed: {event.src_path}")
                    self.config_manager.load_config()
        
        self._observer = Observer()
        self._observer.schedule(ConfigFileHandler(self), str(self.base_path), recursive=False)
        self._observer.start()
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
    
    def register_callback(self, callback: Callable) -> None:
        """Register a callback for configuration changes."""
        self.callbacks.append(callback)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key.split('.')
        value = self.configs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        with self._lock:
            keys = key.split('.')
            config = self.configs
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            self._notify_callbacks()
    
    def get_profile(self) -> LoggingProfile:
        """Get the current logging profile."""
        return self.profile or PROFILES["development"]
    
    def set_profile(self, profile_name: str) -> None:
        """Set the logging profile."""
        if profile_name in PROFILES:
            with self._lock:
                self.profile = PROFILES[profile_name]
                self.configs["profile"] = profile_name
                self._notify_callbacks()
        else:
            raise ValueError(f"Unknown profile: {profile_name}")
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            "configs": self.configs.copy(),
            "profile": asdict(self.profile) if self.profile else None,
            "source": ConfigSource.FILE.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return any issues."""
        issues = []
        
        # Check required fields
        required_fields = ["game_id", "profile"]
        for field in required_fields:
            if field not in self.configs:
                issues.append(f"Missing required field: {field}")
        
        # Validate profile
        if self.configs.get("profile") not in PROFILES:
            issues.append(f"Invalid profile: {self.configs.get('profile')}")
        
        # Validate numeric ranges
        if self.profile:
            if not 0 <= self.profile.sampling_rate <= 1:
                issues.append("Sampling rate must be between 0 and 1")
            
            if self.profile.retention_days < 0:
                issues.append("Retention days must be non-negative")
        
        # Validate storage
        storage = self.configs.get("storage", {})
        if storage.get("max_size_mb", 0) < 10:
            issues.append("Storage max_size_mb should be at least 10 MB")
        
        return issues


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def configure_logging(config_path: Optional[Path] = None, **kwargs) -> ConfigManager:
    """
    Configure logging with the given parameters.
    
    Args:
        config_path: Path to configuration directory
        **kwargs: Additional configuration overrides
    
    Returns:
        ConfigManager instance
    """
    manager = get_config_manager()
    
    if config_path:
        manager.base_path = config_path
        manager.load_config()
    
    # Apply any overrides
    for key, value in kwargs.items():
        manager.set(key, value)
    
    return manager