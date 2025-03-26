"""
This module initializes the visualization package and registers all renderers.
"""
from .renderers import *  # This will import and register all renderers
from .factory import ChartFactory
from .renderer_registry import RendererRegistry

__all__ = ['ChartFactory', 'RendererRegistry']
