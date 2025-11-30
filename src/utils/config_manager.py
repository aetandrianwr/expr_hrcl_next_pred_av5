"""
Configuration management system for production-level experiments.
Supports YAML-based configuration with override capabilities.
"""

import os
import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json


class ConfigManager:
    """
    Production-level configuration manager.
    Loads YAML configs and provides easy access to parameters.
    """
    
    def __init__(self, config_path: str, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            overrides: Optional dictionary to override config values
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Apply overrides
        if overrides:
            self._apply_overrides(overrides)
        
        # Set device
        self._setup_device()
        
        # Create run directory
        self.run_dir = self._create_run_directory()
        
        # Save config to run directory
        self._save_config()
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply override values to config."""
        for key, value in overrides.items():
            keys = key.split('.')
            d = self.config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    
    def _setup_device(self):
        """Setup compute device."""
        device_config = self.config['system']['device']
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        self.config['system']['device'] = str(self.device)
    
    def _create_run_directory(self) -> Path:
        """Create unique directory for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config['experiment']['name']
        run_name = f"{exp_name}_{timestamp}"
        
        run_dir = Path(self.config['paths']['runs_dir']) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (run_dir / 'checkpoints').mkdir(exist_ok=True)
        (run_dir / 'logs').mkdir(exist_ok=True)
        (run_dir / 'predictions').mkdir(exist_ok=True)
        
        return run_dir
    
    def _save_config(self):
        """Save configuration to run directory."""
        config_save_path = self.run_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def get(self, key: str, default=None):
        """
        Get config value using dot notation.
        
        Args:
            key: Key in dot notation (e.g., 'model.d_model')
            default: Default value if key not found
        
        Returns:
            Config value
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getattr__(self, name):
        """Allow attribute-style access to config sections."""
        if name in ['config', 'config_path', 'device', 'run_dir']:
            return object.__getattribute__(self, name)
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def display(self):
        """Display full configuration."""
        print("=" * 80)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 80)
        print(f"\nExperiment: {self.config['experiment']['name']}")
        print(f"Description: {self.config['experiment']['description']}")
        print(f"Dataset: {self.config['experiment']['dataset']}")
        print(f"Run Directory: {self.run_dir}")
        print(f"\n{'-' * 80}")
        print("CONFIGURATION DETAILS")
        print('-' * 80)
        
        def print_section(section_name, section_data, indent=0):
            """Recursively print configuration sections."""
            prefix = "  " * indent
            if isinstance(section_data, dict):
                print(f"{prefix}{section_name}:")
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        print_section(key, value, indent + 1)
                    else:
                        print(f"{prefix}  {key}: {value}")
            else:
                print(f"{prefix}{section_name}: {section_data}")
        
        for section_name, section_data in self.config.items():
            if section_name != 'experiment':
                print_section(section_name, section_data)
                print()
        
        print("=" * 80)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def save_json(self, path: Optional[Path] = None):
        """Save configuration as JSON."""
        if path is None:
            path = self.run_dir / 'config.json'
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory for this run."""
        return self.run_dir / 'checkpoints'
    
    @property
    def log_dir(self) -> Path:
        """Get log directory for this run."""
        return self.run_dir / 'logs'
    
    @property
    def prediction_dir(self) -> Path:
        """Get prediction directory for this run."""
        return self.run_dir / 'predictions'


def load_config(config_path: str, **overrides) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to YAML config file
        **overrides: Key-value pairs to override config
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path, overrides)
