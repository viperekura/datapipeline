"""Factory for creating and registering processors."""

from typing import Dict, List, Any, Optional, Type

from pipeline.processors.base import BaseProcessor
from pipeline.tokenize import AutoTokenizer
from pipeline.strategies import PromptStrategy, StrategyFactory


class ProcessorFactory:
    """Registry and factory for BaseProcessor implementations.

    Supports decorator-based registration for extensible processor types.

    Example usage::

        @ProcessorFactory.register("custom")
        class CustomProcessor(BaseProcessor):
            ...

        processor = ProcessorFactory.create(optimizer, "custom", **kwargs)
    """

    PROCESSOR_MAP: Dict[str, Type[BaseProcessor]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new processor class.

        Args:
            name: Registration name for the processor.

        Returns:
            Decorator function that registers the processor class.
        """

        def decorator(processor_cls: Type[BaseProcessor]) -> Type[BaseProcessor]:
            if not issubclass(processor_cls, BaseProcessor):
                raise TypeError(
                    f"{processor_cls.__name__} must inherit from BaseProcessor"
                )
            cls.PROCESSOR_MAP[name] = processor_cls
            return processor_cls

        return decorator

    @classmethod
    def create(cls, processor_type: str, tokenizer: AutoTokenizer) -> BaseProcessor:
        """Create a processor by type name (uses default ChatMLStrategy for SFT/DPO).

        Args:
            processor_type: Registered processor name (e.g. ``"pt"``, ``"sft"``, ``"dpo"``).
            tokenizer: Tokenizer instance.

        Returns:
            Processor instance.

        Raises:
            ValueError: If processor_type is not registered.
        """
        if processor_type not in cls.PROCESSOR_MAP:
            raise ValueError(
                f"Unknown processor type: '{processor_type}'. "
                f"Supported types: {sorted(cls.PROCESSOR_MAP.keys())}"
            )
        return cls.PROCESSOR_MAP[processor_type](tokenizer)

    @classmethod
    def create_with_strategy(
        cls,
        processor_type: str,
        tokenizer: AutoTokenizer,
        strategy: PromptStrategy,
    ) -> BaseProcessor:
        """Create a processor with a custom strategy.

        Only SFT and DPO processors accept a strategy; PreTrain ignores it.

        Args:
            processor_type: Registered processor name.
            tokenizer: Tokenizer instance.
            strategy: Prompt strategy instance.

        Returns:
            Processor instance configured with strategy.
        """
        if processor_type not in cls.PROCESSOR_MAP:
            raise ValueError(
                f"Unknown processor type: '{processor_type}'. "
                f"Supported types: {sorted(cls.PROCESSOR_MAP.keys())}"
            )

        processor_cls = cls.PROCESSOR_MAP[processor_type]
        if processor_type == "pt":
            return processor_cls(tokenizer)
        return processor_cls(tokenizer, strategy=strategy)

    @classmethod
    def create_with_strategy_name(
        cls,
        processor_type: str,
        tokenizer: AutoTokenizer,
        strategy_name: str,
        **strategy_kwargs,
    ) -> BaseProcessor:
        """Create a processor with a strategy selected by name.

        Args:
            processor_type: Registered processor name.
            tokenizer: Tokenizer instance.
            strategy_name: Registered strategy name (``"chatml"``, ``"alpaca"``, etc.).
            **strategy_kwargs: Forwarded to the strategy constructor.

        Returns:
            Processor instance.
        """
        strategy = StrategyFactory.create(strategy_name, tokenizer, **strategy_kwargs)
        return cls.create_with_strategy(processor_type, tokenizer, strategy)

    @classmethod
    def available_types(cls) -> List[str]:
        """Return list of registered processor type names."""
        return list(cls.PROCESSOR_MAP.keys())
