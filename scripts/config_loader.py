"""
Configuration loader for LLM fine-tuning
Loads and validates config.yaml with environment variable substitution
"""
import os
import yaml
import re
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Load and manage configuration from YAML file"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._resolve_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _resolve_paths(self):
        """Resolve path variables and create directories"""
        storage = self.config['storage']
        shared_fs = storage['shared_fs']
        
        # Resolve relative paths
        storage['data_dir'] = os.path.join(shared_fs, storage['data_dir'])
        storage['checkpoint_dir'] = os.path.join(shared_fs, storage['checkpoint_dir'])
        storage['log_dir'] = os.path.join(shared_fs, storage['log_dir'])
        storage['cache_dir'] = os.path.join(shared_fs, storage['cache_dir'])
        
        # Resolve model cache dir with variable substitution
        cache_dir = self.config['model']['cache_dir']
        cache_dir = cache_dir.replace('${storage.network_disk}', storage['network_disk'])
        self.config['model']['cache_dir'] = cache_dir
        
        # Create directories if they don't exist
        for key in ['data_dir', 'checkpoint_dir', 'log_dir', 'cache_dir']:
            path = storage[key]
            os.makedirs(path, exist_ok=True)
        
        # Create model cache directory
        os.makedirs(self.config['model']['cache_dir'], exist_ok=True)
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        Example: config.get('training.learning_rate')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_cluster_config(self) -> Dict[str, Any]:
        """Get cluster configuration"""
        return self.config['cluster']
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return self.config['storage']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return self.config['dataset']
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration"""
        return self.config['lora']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training']
    
    def get_distributed_config(self) -> Dict[str, Any]:
        """Get distributed training configuration"""
        return self.config['distributed']
    
    def get_checkpointing_config(self) -> Dict[str, Any]:
        """Get checkpointing configuration"""
        return self.config['checkpointing']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config['logging']
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration"""
        return self.config['validation']
    
    def get_benchmarking_config(self) -> Dict[str, Any]:
        """Get benchmarking configuration"""
        return self.config['benchmarking']
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        return self.config['inference']
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size"""
        training = self.get_training_config()
        cluster = self.get_cluster_config()
        
        per_device = training['per_device_train_batch_size']
        grad_accum = training['gradient_accumulation_steps']
        num_gpus = cluster['nodes'] * cluster['gpus_per_node']
        
        return per_device * grad_accum * num_gpus
    
    def print_summary(self):
        """Print configuration summary"""
        cluster = self.get_cluster_config()
        training = self.get_training_config()
        dataset = self.get_dataset_config()
        
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Cluster: {cluster['nodes']} nodes × {cluster['gpus_per_node']} GPUs = {cluster['nodes'] * cluster['gpus_per_node']} GPUs")
        print(f"Model: {self.get('model.model_id')}")
        print(f"Dataset: {dataset['name']} ({dataset.get('max_samples', 'all')} samples)")
        print(f"Epochs: {training['num_train_epochs']}")
        print(f"Batch Size: {training['per_device_train_batch_size']} per device")
        print(f"Gradient Accumulation: {training['gradient_accumulation_steps']} steps")
        print(f"Effective Batch Size: {self.get_effective_batch_size()}")
        print(f"Learning Rate: {training['learning_rate']}")
        print(f"LoRA Rank: {self.get('lora.r')}")
        print(f"Distributed Strategy: {self.get('distributed.strategy').upper()}")
        print(f"Checkpoints: {self.get('storage.checkpoint_dir')}")
        print("=" * 70)


# Convenience function
def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """Load configuration from YAML file"""
    return ConfigLoader(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    config.print_summary()
