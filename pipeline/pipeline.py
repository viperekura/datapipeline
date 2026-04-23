"""Pipeline abstraction for composable data processing stages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, TypeVar, Generic
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class Stage(ABC, Generic[T, R]):
    """Abstract base class for pipeline stages.

    A Stage represents a single processing step that transforms input data
    and can be composed with other stages to form a pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stage name for logging/debugging."""
        pass

    @abstractmethod
    def process(self, input_data: T) -> R:
        """Process input data and return transformed output."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes:
        name: Pipeline name for identification.
        enable_logging: Enable per-stage logging.
        continue_on_error: Continue processing if a stage fails.
        error_threshold: Maximum errors before stopping (-1 for unlimited).
    """

    name: str = "pipeline"
    enable_logging: bool = True
    continue_on_error: bool = True
    error_threshold: int = -1


class Pipeline(Generic[T]):
    """Composable pipeline for sequential data processing.

    Example::

        pipeline = Pipeline(config=PipelineConfig(name="data-prep"))
        pipeline.add_stage(TextNormalizationStage(normalizer))
        pipeline.add_stage(TokenizationStage(tokenizer))
        pipeline.add_stage(PackingStage(packer))

        results = pipeline.run(input_data)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._stages: List[Stage] = []
        self._error_count: int = 0

    def add_stage(self, stage: Stage) -> "Pipeline":
        """Add a stage to the pipeline (fluent interface)."""
        self._stages.append(stage)
        return self

    def add_stages(self, *stages: Stage) -> "Pipeline":
        """Add multiple stages at once."""
        self._stages.extend(stages)
        return self

    def run(self, input_data: List[T]) -> List[Any]:
        """Run all stages sequentially on input data.

        Args:
            input_data: List of input items to process.

        Returns:
            List of processed results.
        """
        if self.config.enable_logging:
            logger.info(f"Starting pipeline '{self.config.name}' with {len(self._stages)} stages")

        results = input_data
        for stage in self._stages:
            if self.config.enable_logging:
                logger.info(f"Running stage: {stage.name}")

            new_results = []
            for item in results:
                if self._should_stop():
                    break

                try:
                    result = stage.process(item)
                    if result is not None:
                        new_results.append(result)
                except Exception as e:
                    self._handle_error(stage, item, e)

            results = new_results

        if self.config.enable_logging:
            logger.info(f"Pipeline '{self.config.name}' completed: {len(results)} items")

        return results

    def run_stream(self, input_data: List[T]) -> Iterator[Any]:
        """Run pipeline as a generator for memory-efficient processing.

        Args:
            input_data: List of input items to process.

        Yields:
            Processed results one at a time.
        """
        for stage in self._stages:
            if self.config.enable_logging:
                logger.info(f"Running stage: {stage.name}")

            for item in input_data:
                if self._should_stop():
                    return

                try:
                    result = stage.process(item)
                    if result is not None:
                        yield result
                except Exception as e:
                    self._handle_error(stage, item, e)

    def _should_stop(self) -> bool:
        """Check if pipeline should stop processing."""
        if self.config.error_threshold < 0:
            return False
        return self._error_count >= self.config.error_threshold

    def _handle_error(self, stage: Stage, item: Any, error: Exception) -> None:
        """Handle processing error."""
        self._error_count += 1
        error_msg = f"Error in stage '{stage.name}': {error}"
        if self.config.continue_on_error:
            logger.warning(error_msg)
        else:
            raise RuntimeError(error_msg) from error

    def __repr__(self) -> str:
        stage_names = [s.name for s in self._stages]
        return f"Pipeline(name='{self.config.name}', stages={stage_names})"


# ── Common Stage Implementations ──────────────────────────────────────────────

@dataclass
class TransformStage(Stage):
    """Stage that applies a transformation function.

    Attributes:
        transform: Callable that transforms input to output.
        name: Stage name.
    """

    transform: callable
    name: str
    _name: str = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        self._name = self.name

    def process(self, input_data: T) -> R:
        return self.transform(input_data)


@dataclass
class FilterStage(Stage):
    """Stage that filters items based on a predicate.

    Attributes:
        predicate: Callable that returns True to keep item.
        name: Stage name.
    """

    predicate: callable
    name: str

    def process(self, input_data: T) -> Optional[T]:
        return input_data if self.predicate(input_data) else None
