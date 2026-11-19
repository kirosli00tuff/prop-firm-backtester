"""Refusals raised by the bar data layer.

Every refusal subclasses GatewayRefusal (a RuntimeError). A refusal always raises; the data layer never
filters a closed or suspect row out silently.
"""


class GatewayRefusal(RuntimeError):
    """Base class of every refusal raised by pfb.data."""


class BadRequest(GatewayRefusal):
    """The request itself is malformed (unknown root, first after last, wrong argument types)."""


class PathsConfigError(GatewayRefusal):
    """config/paths.toml is missing, malformed or names a store root that does not exist."""


class PathRefused(GatewayRefusal):
    """A file path is outside the store roots, in a forbidden directory, or not a known store name."""


class WindowTableMismatch(GatewayRefusal):
    """The closed-window table's bytes do not hash to the pinned sha256, or the table is malformed."""


class ClosedWindowRefused(GatewayRefusal):
    """A request or a row touches a closed window."""

    def __init__(self, window_id: str, message: str) -> None:
        super().__init__(message)
        self.window_id = window_id


class NoBarsFound(GatewayRefusal):
    """No catalogued store file of the root overlaps the requested trade dates."""


class HashMismatch(GatewayRefusal):
    """A store file's bytes no longer hash to the sha256 the catalog recorded."""


class BoundBreached(GatewayRefusal):
    """A row about to be returned lies past the request (label outside the range, or after the bound)."""


class StoreRefused(GatewayRefusal):
    """A store's rows break an invariant (schema, order, duplicate trade date, null label, bad label)."""


class OffGridPrice(GatewayRefusal):
    """A stored price is not a whole number of vendor ticks (no snapping is ever done)."""


class TickMismatch(GatewayRefusal):
    """The tick table and the store's own metadata disagree on the vendor tick."""


class PinMismatch(GatewayRefusal):
    """A catalog entry's bytes differ from the sha256 its producer pinned for that file (ADR A5 a)."""


class EraNotSupported(GatewayRefusal):
    """The request selects a store era this version does not read (the MES confirmation era, ADR A5 b)."""


class CatalogError(GatewayRefusal):
    """The store catalog is missing, malformed, or cannot be built."""
