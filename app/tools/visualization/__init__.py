"""
This module initializes the visualization package and registers all renderers.
"""
import logging
import os

from .renderers import *  # This will import and register all renderers
from .factory import ChartFactory
from .renderer_registry import RendererRegistry

__all__ = ['ChartFactory', 'RendererRegistry']

logging.info("[PID: %s] visualization/__init__.py completed.", os.getpid())

# At the end of app/tools/visualization/renderers/__init__.py
