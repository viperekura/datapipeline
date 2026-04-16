"""Factory for creating and registering prompt strategies."""

from typing import Dict, List, Type

from pipeline.tokenize import AutoTokenizer
from pipeline.strategies.base import PromptStrategy


class StrategyFactory:
    """Registry and factory for PromptStrategy implementations.

    Supports decorator-based registration for extensible strategy types.

    Example usage::

        @StrategyFactory.register("custom")
        class CustomStrategy(PromptStrategy):
            ...

        strategy = StrategyFactory.create("custom", tokenizer, **kwargs)
    """

    STRATEGY_MAP: Dict[str, Type[PromptStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new strategy class.

        Args:
            name: Registration name for the strategy.

        Returns:
            Decorator function that registers the strategy class.
        """

        def decorator(strategy_cls: Type[PromptStrategy]) -> Type[PromptStrategy]:
            if not issubclass(strategy_cls, PromptStrategy):
                raise TypeError(
                    f"{strategy_cls.__name__} must inherit from PromptStrategy"
                )
            cls.STRATEGY_MAP[name] = strategy_cls
            return strategy_cls

        return decorator

    @classmethod
    def create(cls, name: str, tokenizer: AutoTokenizer, **kwargs) -> PromptStrategy:
        """Create a strategy by name.

        Args:
            name: Registered strategy name (e.g. ``"chatml"``, ``"alpaca"``).
            tokenizer: Tokenizer instance (required by all strategies).
            **kwargs: Forwarded to the strategy constructor.

        Returns:
            Strategy instance.

        Raises:
            ValueError: If name is not registered.
        """
        if name not in cls.STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy: '{name}'. "
                f"Supported types: {sorted(cls.STRATEGY_MAP.keys())}"
            )
        return cls.STRATEGY_MAP[name](tokenizer=tokenizer, **kwargs)

    @classmethod
    def available_types(cls) -> List[str]:
        """Return list of registered strategy type names."""
        return list(cls.STRATEGY_MAP.keys())
