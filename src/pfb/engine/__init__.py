"""Backtest engine and fill simulation (ADR-001 sections 7 to 9, amended by section 16)."""

# 0.2.0: fill convention for a must-be-flat interval without a bar; 0.3.0: the price-limit and
# locked-market rules run, exit reason PRICE_LIMIT, six new kernel counters; 0.3.1: a stop or flat-by exit
# met on a locked bar waits and fills at the open of the first bar that trades through, a session-end exit
# on a locked bar refuses the run, a PRICE_LIMIT exit pays the cost model's event-window slippage. No trial
# ran at 0.3.0.
ENGINE_VERSION = "pfb-engine/0.3.1"

__all__ = ["ENGINE_VERSION"]
