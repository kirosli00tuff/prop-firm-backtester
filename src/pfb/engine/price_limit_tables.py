"""CME price-limit periods per root, as data. Source ids name the source records each period was read
from; no source text is copied.

Kinds: HARD_DAILY (a fixed amount in exchange price units, initial limit only), EQUITY_BANDS (eth_pct
up and down outside RTH; 7 % then 20 % down only in RTH; the first RTH level kept), DCB_ONLY (dynamic
circuit breakers only; under the default reading NO_LOCK_LIMIT; the rule does not apply).
Status: sourced, bracketed, carried or pending (pending raises LimitPending).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from enum import Enum

D = Decimal


class LimitKind(Enum):
    HARD_DAILY = "hard_daily"
    EQUITY_BANDS = "equity_bands"
    DCB_ONLY = "dcb_only"


@dataclass(frozen=True)
class LimitPeriod:
    start: date
    end: date  # inclusive
    kind: LimitKind
    status: str  # sourced | bracketed | carried | pending
    n_observations: int
    sources: tuple[str, ...]
    amount: Decimal | None = None  # HARD_DAILY, exchange price units
    eth_pct: int | None = None  # EQUITY_BANDS
    rth_7pct_until: time | None = None  # EQUITY_BANDS
    dcb_kind: str | None = None  # DCB_ONLY: "pct" or "points"
    dcb_value: Decimal | None = None


Row = tuple[str, str, str | None, str, int, tuple[str, ...]]


def _hard(rows: tuple[Row, ...]) -> tuple[LimitPeriod, ...]:
    return tuple(
        LimitPeriod(date.fromisoformat(a), date.fromisoformat(b), LimitKind.HARD_DAILY, status, n, src,
                    amount=None if amt is None else D(amt))
        for a, b, amt, status, n, src in rows
    )  # fmt: skip


def _equity(
    rows: tuple[tuple[str, str, int, str, str, int, tuple[str, ...]], ...],
) -> tuple[LimitPeriod, ...]:
    return tuple(
        LimitPeriod(date.fromisoformat(a), date.fromisoformat(b), LimitKind.EQUITY_BANDS, status, n, src,
                    eth_pct=eth, rth_7pct_until=time.fromisoformat(until))
        for a, b, eth, until, status, n, src in rows
    )  # fmt: skip


def _dcb(
    rows: tuple[tuple[str, str, str | None, str | None, str, int, tuple[str, ...]], ...],
) -> tuple[LimitPeriod, ...]:
    return tuple(
        LimitPeriod(date.fromisoformat(a), date.fromisoformat(b), LimitKind.DCB_ONLY, status, n, src,
                    dcb_kind=kind, dcb_value=None if value is None else D(value))
        for a, b, kind, value, status, n, src in rows
    )  # fmt: skip


FAQ, LIVE = "cme_grain_faq", "cme_livestock_notices"

ZC = _hard((
    ("2019-05-01", "2021-05-02", "0.25", "sourced", 49, ("pl_20190408215002", "pl_20210411030219")),
    ("2021-05-03", "2021-10-31", "0.40", "sourced", 10, (FAQ, "pl_20210519160902", "pl_20211005224651")),
    ("2021-11-01", "2022-05-01", "0.35", "sourced", 13, (FAQ, "pl_20211203073758", "pl_20220324054144")),
    ("2022-05-02", "2022-10-31", "0.50", "sourced", 8, (FAQ, "pl_20220521112448", "pl_20221008031710")),
    ("2022-11-01", "2023-10-31", "0.45", "sourced", 11, (FAQ, "pl_20230323140643", "pl_20230707204329")),
    ("2023-11-01", "2024-04-30", "0.35", "sourced", 7, (FAQ, "pl_20230928165949", "pl_20240413052451")),
    ("2024-05-01", "2025-04-30", "0.30", "sourced", 31, (FAQ, "pl_20240519144823", "pl_20250428093001")),
    ("2025-05-01", "2025-11-02", "0.35", "sourced", 21, (FAQ, "pl_20250513061945", "pl_20251018123959")),
    ("2025-11-03", "2026-06-19", "0.30", "sourced", 13, (FAQ, "pl_20251112101804", "pl_20260907145625")),
))  # fmt: skip
ZW = _hard((
    ("2019-05-01", "2020-04-30", "0.35", "sourced", 18, ("pl_20190408215002", "pl_20200422053715")),
    ("2020-05-01", "2021-05-02", "0.40", "sourced", 32, (FAQ, "pl_20200531052856", "pl_20210411030219")),
    ("2021-05-03", "2021-10-31", "0.45", "sourced", 12, (FAQ, "pl_20210519160902", "pl_20211005224651")),
    ("2021-11-01", "2022-01-28", "0.50", "sourced", 7, (FAQ, "pl_20211203073758", "pl_20220128182742")),
    ("2022-01-29", "2022-03-06", "0.50", "bracketed", 1, (FAQ, "pl_20220307144843")),
    ("2022-03-07", "2022-05-01", "0.85", "sourced", 4, ("pl_20220307144843", "pl_20220324054144")),
    ("2022-05-02", "2022-10-31", "0.70", "sourced", 8, (FAQ, "pl_20220521112448", "pl_20221008031710")),
    ("2022-11-01", "2023-04-30", "0.65", "sourced", 7, (FAQ, "pl_20230323140643", "pl_20230326155246")),
    ("2023-05-01", "2023-10-31", "0.60", "sourced", 4, (FAQ, "pl_20230523200617", "pl_20230707204329")),
    ("2023-11-01", "2024-04-30", "0.50", "sourced", 7, (FAQ, "pl_20230928165949", "pl_20240413052451")),
    ("2024-05-01", "2025-11-02", "0.40", "sourced", 52, (FAQ, "pl_20240519144823", "pl_20251018123959")),
    ("2025-11-03", "2026-04-30", "0.35", "sourced", 10, (FAQ, "pl_20251112101804", "pl_20260418140621")),
    ("2026-05-01", "2026-06-19", "0.45", "sourced", 3, (FAQ, "pl_20260516231307", "pl_20260907145625")),
))  # fmt: skip
ZS = _hard((
    ("2019-05-01", "2019-10-31", "0.65", "sourced", 4, (FAQ, "pl_20190509132638", "pl_20190921105213")),
    ("2019-11-01", "2020-11-01", "0.60", "sourced", 26, (FAQ, "pl_20191108140542", "pl_20201005092639")),
    ("2020-11-02", "2021-05-02", "0.70", "sourced", 18, (FAQ, "pl_20201028173809", "pl_20210411030219")),
    ("2021-05-03", "2021-10-31", "1.00", "sourced", 10, (FAQ, "pl_20210519160902", "pl_20211005224651")),
    ("2021-11-01", "2022-05-01", "0.90", "sourced", 13, (FAQ, "pl_20211203073758", "pl_20220324054144")),
    ("2022-05-02", "2022-10-31", "1.15", "sourced", 8, (FAQ, "pl_20220521112448", "pl_20221008031710")),
    ("2022-11-01", "2023-04-30", "1.00", "sourced", 7, (FAQ, "pl_20230323140643", "pl_20230326155246")),
    ("2023-05-01", "2023-10-31", "1.05", "sourced", 4, (FAQ, "pl_20230523200617", "pl_20230707204329")),
    ("2023-11-01", "2024-04-30", "0.95", "sourced", 7, (FAQ, "pl_20230928165949", "pl_20240413052451")),
    ("2024-05-01", "2024-10-31", "0.85", "sourced", 12, (FAQ, "pl_20240519144823", "pl_20241007192846")),
    ("2024-11-01", "2025-04-30", "0.70", "sourced", 19, (FAQ, "pl_20241106132409", "pl_20250428093001")),
    ("2025-05-01", "2025-11-02", "0.75", "sourced", 20, (FAQ, "pl_20250513061945", "pl_20251018123959")),
    ("2025-11-03", "2026-04-30", "0.70", "sourced", 10, (FAQ, "pl_20251112101804", "pl_20260418140621")),
    ("2026-05-01", "2026-06-19", "0.85", "sourced", 3, (FAQ, "pl_20260516231307", "pl_20260907145625")),
))  # fmt: skip
ZM = _hard((
    ("2019-05-01", "2020-11-01", "20.00", "sourced", 31, ("pl_20190408215002", "pl_20201005092639")),
    ("2020-11-02", "2021-05-02", "25.00", "sourced", 18, (FAQ, "pl_20201028173809", "pl_20210411030219")),
    ("2021-05-03", "2021-10-31", "30.00", "sourced", 10, (FAQ, "pl_20210519160902", "pl_20211005224651")),
    ("2021-11-01", "2022-05-01", "25.00", "sourced", 13, (FAQ, "pl_20211203073758", "pl_20220324054144")),
    ("2022-05-02", "2023-10-31", "30.00", "sourced", 19, (FAQ, "pl_20220521112448", "pl_20230707204329")),
    ("2023-11-01", "2024-10-31", "25.00", "sourced", 19, (FAQ, "pl_20230928165949", "pl_20241007192846")),
    ("2024-11-01", "2026-06-19", "20.00", "sourced", 52, (FAQ, "pl_20241106132409", "pl_20260907145625")),
))  # fmt: skip
ZL = _hard((
    ("2019-05-01", "2020-11-01", "0.020", "sourced", 31, ("pl_20190408215002", "pl_20201005092639")),
    ("2020-11-02", "2021-05-02", "0.025", "sourced", 18, (FAQ, "pl_20201028173809", "pl_20210411030219")),
    ("2021-05-03", "2021-10-31", "0.035", "sourced", 10, (FAQ, "pl_20210519160902", "pl_20211005224651")),
    ("2021-11-01", "2022-05-01", "0.040", "sourced", 13, (FAQ, "pl_20211203073758", "pl_20220324054144")),
    ("2022-05-02", "2022-10-31", "0.050", "sourced", 8, (FAQ, "pl_20220521112448", "pl_20221008031710")),
    ("2022-11-01", "2023-04-30", "0.045", "sourced", 7, (FAQ, "pl_20230323140643", "pl_20230326155246")),
    ("2023-05-01", "2024-04-30", "0.040", "sourced", 11, (FAQ, "pl_20230523200617", "pl_20240413052451")),
    ("2024-05-01", "2024-10-31", "0.035", "sourced", 12, (FAQ, "pl_20240519144823", "pl_20241007192846")),
    ("2024-11-01", "2025-11-02", "0.030", "sourced", 39, (FAQ, "pl_20241106132409", "pl_20251018123959")),
    ("2025-11-03", "2026-04-30", "0.035", "sourced", 10, (FAQ, "pl_20251112101804", "pl_20260418140621")),
    ("2026-05-01", "2026-06-19", "0.045", "sourced", 3, (FAQ, "pl_20260516231307", "pl_20260907145625")),
))  # fmt: skip
LE = _hard((
    ("2019-05-01", "2020-10-04", "0.03", "sourced", 25, ("pl_20190408215002", "pl_20201005092639")),
    ("2020-10-05", "2021-05-31", "0.04", "sourced", 21, (LIVE, "pl_20201028173809", "pl_20210526094607")),
    ("2021-06-01", "2022-05-31", "0.05", "sourced", 23, (LIVE, "pl_20210615145720", "pl_20220521112448")),
    ("2022-06-01", "2023-05-31", "0.0575", "sourced", 15, (LIVE, "pl_20220625225942", "pl_20230523200617")),
    ("2023-06-01", "2024-06-02", "0.0675", "sourced", 12, (LIVE, "pl_20230531231343", "pl_20240520085408")),
    ("2024-06-03", "2024-10-08", "0.0750", "sourced", 10, (LIVE, "pl_20240616110534", "pl_20241007192846")),
    ("2024-10-09", "2024-11-05", "0.0650", "bracketed", 1, (LIVE, "pl_20241106132409")),
    ("2024-11-06", "2025-06-01", "0.0650", "sourced", 23, ("pl_20241106132409", "pl_20250527172431")),
    ("2025-06-02", "2026-05-18", "0.0725", "sourced", 24, (LIVE, "pl_20250604005334", "pl_20260516231307")),
    ("2026-05-19", "2026-06-19", "0.0725", "bracketed", 1, (LIVE, "pl_20260820062901")),
))  # fmt: skip
HE = _hard((
    ("2019-05-01", "2020-04-01", "0.03", "sourced", 13, ("pl_20190408215002", "pl_20200401165707")),
    ("2020-04-02", "2020-04-21", "0.03", "bracketed", 2, ("pl_20200401165707", "pl_20200422053715")),
    ("2020-04-22", "2020-08-19", "0.0375", "sourced", 9, ("pl_20200422053715", "pl_20200819160616")),
    ("2020-08-20", "2020-09-21", "0.03", "bracketed", 2, ("pl_20200819160616", "pl_20200922013357")),
    ("2020-09-22", "2021-08-31", "0.03", "sourced", 32, ("pl_20200922013357", "pl_20210731164704")),
    ("2021-09-01", "2023-08-31", "0.0475", "sourced", 33, (LIVE, "pl_20210918003230", "pl_20230707204329")),
    ("2023-09-01", "2024-09-03", "0.0375", "sourced", 14, (LIVE, "pl_20230928165949", "pl_20240901090928")),
    ("2024-09-04", "2024-09-12", "0.0375", "bracketed", 1, (LIVE, "pl_20240913105737")),
    ("2024-09-13", "2025-08-29", "0.0400", "sourced", 37, ("pl_20240913105737", "pl_20250829060023")),
    ("2025-08-30", "2025-09-01", "0.0400", "bracketed", 2, ("pl_20250829060023", "pl_20250830034814")),
    ("2025-09-02", "2026-06-19", "0.0475", "sourced", 20, ("pl_20250830034814", "pl_20260820062901")),
))  # fmt: skip
EQUITY = _equity((
    ("2019-05-01", "2019-09-23", 5, "15:00", "sourced", 5, ("pl_20190408215002", "pl_20190921105213")),
    ("2019-09-24", "2019-11-07", 5, "15:00", "bracketed", 2, ("pl_20190921105213", "pl_20191108140542")),
    ("2019-11-08", "2020-09-30", 5, "14:25", "sourced", 26, ("pl_20191108140542", "pl_20201005092639")),
    ("2020-10-01", "2020-11-01", 5, "14:25", "bracketed", 2, ("pl_20201005092639", "pl_20201028173809")),
    ("2020-11-02", "2026-06-19", 7, "14:25", "sourced", 135, ("pl_20201028173809", "pl_20260907145625")),
))  # fmt: skip
_RATES_SRC = ("pl_20260516231307", "spfl_20260310032723")


def _rates(points: str) -> tuple[LimitPeriod, ...]:
    return _dcb((
        ("2019-05-01", "2020-10-09", None, None, "pending", 0, ()),
        ("2020-10-10", "2026-06-19", "points", points, "sourced", 15, _RATES_SRC),
    ))  # fmt: skip


S22, S26 = "spfl_20220124234130", "spfl_20260310032723"
FX = _dcb((
    ("2019-05-01", "2022-01-23", None, None, "pending", 0, ()),
    ("2022-01-24", "2026-03-10", "pct", "4", "sourced", 10, (S22, S26)),
    ("2026-03-11", "2026-06-19", "pct", "4", "carried", 1, (S26,)),
))  # fmt: skip
S19, S21, P20 = "spfl_20190718234942", "spfl_20210724172159", "pl_20200325160849"


def _oil(switch_end: str, switch_start: str) -> tuple[LimitPeriod, ...]:
    """CL, MCL, QM (2020-03-18/19) and RB, HO (2020-03-12/13)."""
    return _dcb((
        ("2019-05-01", "2019-07-17", None, None, "pending", 0, ()),
        ("2019-07-18", switch_end, "pct", "7", "sourced", 1, (S19,)),
        (switch_start, "2021-07-24", "pct", "15", "sourced", 5, (P20, S21)),
        ("2021-07-25", "2022-01-23", "pct", "10", "bracketed", 2, (P20, S22)),
        ("2022-01-24", "2026-03-10", "pct", "10", "sourced", 10, (S22, S26)),
        ("2026-03-11", "2026-06-19", "pct", "10", "carried", 1, (S26,)),
    ))  # fmt: skip


GAS = _dcb((
    ("2019-05-01", "2019-07-17", None, None, "pending", 0, ()),
    ("2019-07-18", "2021-07-24", "pct", "7", "sourced", 5, (S19, S21)),
    ("2021-07-25", "2022-01-23", "pct", "7", "bracketed", 2, (S21, S22)),
    ("2022-01-24", "2026-03-10", "pct", "10", "sourced", 10, (S22, S26)),
    ("2026-03-11", "2026-06-19", "pct", "10", "carried", 1, (S26,)),
))  # fmt: skip
S20 = "spfl_20200618002851"
METALS = _dcb((
    ("2019-05-01", "2019-07-17", None, None, "pending", 0, ()),
    ("2019-07-18", "2019-07-18", "pct", "5", "sourced", 1, (S19,)),
    ("2019-07-19", "2020-06-17", "pct", "5", "bracketed", 2, (S19, S20)),
    ("2020-06-18", "2026-03-10", "pct", "10", "sourced", 14, (S20, S26)),
    ("2026-03-11", "2026-06-19", "pct", "10", "carried", 1, (S26,)),
))  # fmt: skip
CRYPTO = _dcb((("2021-05-03", "2026-06-19", "pct", "10", "sourced", 16, ("pl_20260516231307", S26)),))

LIMITS: dict[str, tuple[LimitPeriod, ...]] = {
    "ZC": ZC, "ZW": ZW, "ZS": ZS, "ZM": ZM, "ZL": ZL, "LE": LE, "HE": HE,
    **{r: EQUITY for r in ("NQ", "MNQ", "RTY", "M2K", "YM", "MYM", "ES", "MES")},
    "ZT": _rates("0.75"), "ZF": _rates("1.50"), "ZN": _rates("2.00"), "TN": _rates("3.00"),
    "ZB": _rates("4.50"), "UB": _rates("8.00"),
    **{r: FX for r in ("6E", "M6E", "E7", "6A", "M6A", "6B", "M6B", "6C", "6J", "6S", "6N")},
    **{r: _oil("2020-03-18", "2020-03-19") for r in ("CL", "MCL", "QM")},
    **{r: GAS for r in ("NG", "MNG", "QG")},
    **{r: _oil("2020-03-12", "2020-03-13") for r in ("RB", "HO")},
    **{r: METALS for r in ("GC", "MGC", "SI", "SIL", "HG", "MHG")},
    "MBT": CRYPTO,
}  # fmt: skip
TABLE_WINDOW = (date(2019, 5, 1), date(2026, 6, 19))
HARD_LIMIT_PRODUCTS: frozenset[str] = frozenset(
    r for r, ps in LIMITS.items() if ps[0].kind is not LimitKind.DCB_ONLY
)

__all__ = ["HARD_LIMIT_PRODUCTS", "LIMITS", "TABLE_WINDOW", "LimitKind", "LimitPeriod"]
