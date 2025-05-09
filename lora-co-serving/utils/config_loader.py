import yaml
from dataclasses import dataclass, field
import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2-0.5B-Instruct"

@dataclass
class ControllerConfig:
    inference_batch_size: int = 4
    training_batch_size: int = 1

@dataclass
class PrioritizationConfig:
    strategy: str = "RoundRobin"

@dataclass
class ManagersConfig:
    adapter_cache_size: int = 3

@dataclass
class QueueConfig:
    max_inference_queue_size: int = 100
    max_training_queue_size: int = 10

@dataclass
class EngineConfig:
    inference_batch_timeout_ms: int = 200

@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])

@dataclass
class TrainingConfig:
    lora_config: LoraConfig = field(default_factory=LoraConfig)

@dataclass
class SystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    prioritization: PrioritizationConfig = field(default_factory=PrioritizationConfig)
    managers: ManagersConfig = field(default_factory=ManagersConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(config_path: str = "configs/config.yaml") -> Optional[SystemConfig]:
    """Loads system configuration from a YAML file into a SystemConfig object.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Optional[SystemConfig]: The loaded configuration object, or None on error.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if not config_dict:
             logger.warning(f"Configuration file is empty: {config_path}")
             return SystemConfig()

        def update_dataclass(dc_instance, data_dict):
            for key, value in data_dict.items():
                if hasattr(dc_instance, key):
                    field_instance = getattr(dc_instance, key)
                    if isinstance(field_instance, object) and hasattr(field_instance, '__dataclass_fields__') and isinstance(value, dict):
                        update_dataclass(field_instance, value)
                    else:
                        setattr(dc_instance, key, value)
                else:
                    logger.warning(f"Ignoring unknown config key '{key}' in section {dc_instance.__class__.__name__}")
            return dc_instance

        config = SystemConfig()
        config = update_dataclass(config, config_dict)

        logger.info(f"Configuration loaded successfully from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}", exc_info=True)
        return None
