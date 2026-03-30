"""Data processors with factory pattern.

Processor classes are registered at definition time via decorators and
can be created through :class:`ProcessorFactory`.
"""
from pipeline.processors.base import BaseProcessor
from pipeline.processors.factory import ProcessorFactory
from pipeline.processors.pretrain import PreTrainProcessor
from pipeline.processors.sft import SFTProcessor
from pipeline.processors.dpo import DPOProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorFactory",
    "PreTrainProcessor",
    "SFTProcessor",
    "DPOProcessor",
]
