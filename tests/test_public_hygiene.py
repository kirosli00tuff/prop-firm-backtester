"""Public-hygiene gate: no tracked file may carry market data, secrets or saved third-party pages.

`find_violations` is a pure checker over a repo-relative path and the file text;
`test_tracked_files_are_clean` feeds it every file from `git ls-files`.
Regexes are built from pieces so this file never matches itself.
"""

import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 2 * 1024 * 1024
BINARY_SNIFF_BYTES = 8192
MARKET_BLOCK_LINES = 20
MARKET_MIN_DECIMALS = 4
MIN_SECRET_LITERAL = 16

BANNED_SUFFIXES = frozenset(
    {".parquet", ".dbn", ".zst", ".npy", ".npz", ".feather", ".arrow", ".h5", ".hdf5", ".pkl", ".pickle"}
)
THIRD_PARTY_SUFFIXES = frozenset({".html", ".htm", ".mhtml", ".pdf"})
BANNED_ROOT_DIRS = ("data", "runs")
COST_TABLE_NAME = "real" + "_book_costs" + ".json"  # a made-up name for a private table
ENV_EXAMPLE_NAME = ".env" + ".example"
ALLOWED_EMAIL = "noreply" + "@" + "anthropic.com"
ALLOWED_EMAIL_DOMAINS = frozenset({"example." + "com", "example." + "org"})
PLACEHOLDER_MARKERS = ("xxx", "changeme", "example")

_DASH = "-"
KEY_RE = re.compile("db" + _DASH + "[A-Za-z0-9]{20,}")
PEM_RE = re.compile(_DASH * 5 + "BEGIN [A-Z ]*PRIVATE KEY" + _DASH * 5)
_SECRET_NAMES = "(?:api[_-]?key|secret|token|pass" + "word|pass" + "wd)"
SECRET_RE = re.compile(
    "(?i)" + _SECRET_NAMES + "[\"']?\\s*[:=]\\s*[\"']([^\\s\"']{" + str(MIN_SECRET_LITERAL) + ",})[\"']"
)
EMAIL_RE = re.compile("[A-Za-z0-9._%+-]+" + "@" + "([A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)*\\.[A-Za-z]{2,})")
_ISO_TS = "\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:Z|[+-]\\d{2}:?\\d{2})?"
_EPOCH_NS = "(?<!\\d)\\d{19}(?!\\d)"
TIMESTAMP_RE = re.compile(_ISO_TS + "|" + _EPOCH_NS)
DECIMAL_RE = re.compile("(?<![\\w.])-?\\d+\\.\\d+(?![\\w.])")
ENV_VALUE_RE = re.compile(r"^[ \t]*(?:export[ \t]+)?[A-Za-z_][A-Za-z0-9_]*[ \t]*=[ \t]*[^\s#]", re.MULTILINE)


def _is_placeholder(literal: str) -> bool:
    lowered = literal.lower()
    return literal.startswith("<") or any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _name_violations(path: str) -> list[str]:
    pure = PurePosixPath(path)
    found: list[str] = []
    if pure.suffix.lower() in BANNED_SUFFIXES:
        found.append(f"{path}: banned data-file suffix {pure.suffix}")
    if pure.suffix.lower() in THIRD_PARTY_SUFFIXES:
        found.append(f"{path}: saved third-party page or paper ({pure.suffix})")
    if pure.parts and pure.parts[0] in BANNED_ROOT_DIRS and len(pure.parts) > 1:
        found.append(f"{path}: lives under {pure.parts[0]}/ at the repo root")
    if pure.name == COST_TABLE_NAME:
        found.append(f"{path}: cost-table file name is never tracked")
    return found


def _env_violations(path: str, text: str) -> list[str]:
    name = PurePosixPath(path).name
    if name != ".env" and not name.startswith(".env."):
        return []
    if name == ENV_EXAMPLE_NAME and not ENV_VALUE_RE.search(text):
        return []
    return [f"{path}: environment file with values must not be tracked"]


def _secret_violations(path: str, text: str) -> list[str]:
    found: list[str] = []
    if KEY_RE.search(text):
        found.append(f"{path}: vendor-style API key")
    if PEM_RE.search(text):
        found.append(f"{path}: PEM private key header")
    if any(not _is_placeholder(m.group(1)) for m in SECRET_RE.finditer(text)):
        found.append(f"{path}: literal secret assignment")
    return found


def _email_violations(path: str, text: str) -> list[str]:
    for match in EMAIL_RE.finditer(text):
        address = match.group(0)
        if address == ALLOWED_EMAIL or match.group(1).lower() in ALLOWED_EMAIL_DOMAINS:
            continue
        return [f"{path}: e-mail address in tracked text"]
    return []


def _is_market_line(line: str) -> bool:
    match = TIMESTAMP_RE.search(line)
    if match is None:
        return False
    rest = line[: match.start()] + " " + line[match.end() :]
    return len(DECIMAL_RE.findall(rest)) >= MARKET_MIN_DECIMALS


def _market_block_violations(path: str, text: str) -> list[str]:
    streak = 0
    for line in text.splitlines():
        streak = streak + 1 if _is_market_line(line) else 0
        if streak >= MARKET_BLOCK_LINES:
            return [f"{path}: block of row-level market data ({MARKET_BLOCK_LINES}+ timestamped rows)"]
    return []


def find_violations(path: str, text: str) -> list[str]:
    """Return one message per hygiene rule that `path` (repo-relative, posix) with `text` breaks."""
    return [
        *_name_violations(path),
        *_env_violations(path, text),
        *_secret_violations(path, text),
        *_email_violations(path, text),
        *_market_block_violations(path, text),
    ]


def _tracked_files() -> list[str]:
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("git or the repository is unavailable")
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=False, timeout=60
    )
    if result.returncode != 0:
        pytest.skip("git ls-files failed")
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


def _scan_tracked(rel: str) -> list[str]:
    file = REPO_ROOT / rel
    if not file.is_file():
        return []
    found: list[str] = []
    size = file.stat().st_size
    if size > MAX_FILE_BYTES:
        found.append(f"{rel}: larger than {MAX_FILE_BYTES} bytes ({size})")
    raw = file.read_bytes()
    is_binary = b"\0" in raw[:BINARY_SNIFF_BYTES]
    text = "" if is_binary else raw.decode("utf-8", errors="replace")
    return [*found, *find_violations(rel, text)]


def test_tracked_files_are_clean() -> None:
    violations = [v for rel in _tracked_files() for v in _scan_tracked(rel)]
    assert not violations, "\n".join(violations)


# ---- self-tests: each rule catches a synthetic offender, a clean string passes ----

CLEAN_TEXT = "Plain prose.\nprice_ticks = 4\ntoken_count = 3\nContact: noreply" + "@" + "anthropic.com\n"


def _market_rows(count: int) -> str:
    row = "2024-01-02T03:04:05.123456Z 1.25 2.50 3.75 4.00 10"
    return "\n".join([row] * count)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("a) vendor key", "key = " + "db" + "-" + "A1" * 12),
        ("b) PEM header", _DASH * 5 + "BEGIN RSA PRIVATE KEY" + _DASH * 5),
        ("c) secret literal", "api" + "_key" + ' = "' + "Q9" * 10 + '"'),
        ("c) secret literal colon", "'" + "to" + "ken'" + ": '" + "z" * 24 + "'"),
        ("d) e-mail", "mail me at someone" + "@" + "corp-mail." + "net"),
        ("f) market block", _market_rows(MARKET_BLOCK_LINES)),
        ("f) epoch-ns block", "\n".join(f"{1700000000000000000 + i} 1.5 2.5 3.5 4.5" for i in range(25))),
    ],
)
def test_content_rule_catches_offender(label: str, text: str) -> None:
    assert find_violations("src/pfb/example.py", text), label


@pytest.mark.parametrize(
    "path",
    [
        "stuff/" + COST_TABLE_NAME,
        "x.parquet",
        "d/x.DBN",
        "a.zst",
        "a.npy",
        "a.pkl",
        "data/anything.txt",
        "runs/run1/out.json",
        ".env",
        ".env.local",
        "sub/.env.prod",
        "saved/page.html",
        "papers/x.pdf",
    ],
)
def test_name_rule_catches_offender(path: str) -> None:
    assert find_violations(path, "")


def test_rule_e_names_the_cost_table_file() -> None:
    assert any("cost-table" in v for v in find_violations(COST_TABLE_NAME, ""))


def test_clean_text_and_paths_pass() -> None:
    assert find_violations("src/pfb/cli/main.py", CLEAN_TEXT) == []
    assert find_violations("docs/data/notes.md", CLEAN_TEXT) == []
    assert find_violations("config/fees/topstep.toml", "tick_value_cents = 1250\n") == []


def test_cost_table_name_mentioned_in_text_is_allowed() -> None:
    assert find_violations("docs/DECISIONS.md", "see " + COST_TABLE_NAME + " in the upstream repo") == []


@pytest.mark.parametrize(
    "text",
    [
        'api_key = "<your-key-goes-here-0000>"',
        'token = "xxxxxxxxxxxxxxxxxxxxxxxx"',
        'password: "changeme-changeme-change"',
        'secret = "example-example-example-1"',
        'api_key = "short"',
        "tokenizer = load()",
    ],
)
def test_secret_placeholders_and_short_values_pass(text: str) -> None:
    assert find_violations("src/pfb/x.py", text) == []


def test_allowed_email_addresses_pass() -> None:
    text = "a@" + "example.com b@" + "example.org " + ALLOWED_EMAIL
    assert find_violations("docs/x.md", text) == []


def test_env_example_without_values_is_allowed() -> None:
    assert find_violations(ENV_EXAMPLE_NAME, "# keys\nSOME_NAME=\nOTHER_NAME=  \n") == []


def test_env_example_with_a_value_is_caught() -> None:
    assert find_violations(ENV_EXAMPLE_NAME, "SOME_NAME=abc\n")


def test_market_block_shorter_than_threshold_passes() -> None:
    assert find_violations("docs/x.md", _market_rows(MARKET_BLOCK_LINES - 1)) == []


def test_market_block_broken_by_an_ordinary_line_passes() -> None:
    rows = _market_rows(MARKET_BLOCK_LINES - 1) + "\nnot a row\n" + _market_rows(MARKET_BLOCK_LINES - 1)
    assert find_violations("docs/x.md", rows) == []
