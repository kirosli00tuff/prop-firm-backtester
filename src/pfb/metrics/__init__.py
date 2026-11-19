"""Performance metrics and statistics (ADR-001 section 11, A8). Imports nothing from other pfb packages."""

from pfb.metrics.core import MetricsInputError, summarize

__all__ = ["MetricsInputError", "summarize"]
