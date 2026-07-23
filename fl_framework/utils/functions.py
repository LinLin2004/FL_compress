import random
import torch
import numpy as np
import os
import logging
from typing import Any, Dict
from importlib import import_module


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_level: str = "INFO", file_path: str = None):
    """Configures logging to both console (bash) and file."""
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]
    if file_path:
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

def create_component_from_config(config: Dict[str, Any], **extra_params: Any) -> Any:
    """
    Dynamically creates an instance of a component from its configuration.

    The configuration dictionary must contain 'class_path' and an optional 'params' dict.

    Args:
        config (Dict[str, Any]): The configuration dictionary for the component.
        **extra_params: Additional keyword arguments to pass to the constructor.

    Returns:
        Any: An instance of the specified class.
    """
    if not config:
        return None
    
    module_path, class_name = config['class_path'].rsplit('.', 1)
    ComponentClass = getattr(import_module(module_path), class_name)
    
    # Combine params from config file and any extra params passed in
    params = config.get('params', {}) or {}
    params.update(extra_params)
    
    return ComponentClass(**params)