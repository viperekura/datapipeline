"""Strategy pattern for prompt/response format abstraction."""
from pipeline.strategies.base import PromptStrategy
from pipeline.strategies.factory import StrategyFactory

# Import strategy implementations to trigger decorator registration
from pipeline.strategies.chatml import ChatMLStrategy  # noqa: F401
from pipeline.strategies.alpaca import AlpacaStrategy  # noqa: F401

__all__ = [
    "PromptStrategy",
    "StrategyFactory",
    "ChatMLStrategy",
    "AlpacaStrategy",
]
