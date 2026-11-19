"""Step 4 of the Backtest page (docs/design/DASH_V2.md): pick open windows and firm plans,
build the run's hypothesis from the SPEC (pfb.dash.hypgen), show it in full, check it with `pfb hypothesis
register --dry-run` (N unchanged), show whether both files are committed with the exact git command, then,
behind a confirm box that names the N change, register and run one counted trial per picked window.

This module writes nothing and spawns nothing itself: saving, the dry run, the git state and the runs go
through pfb.dash.actions (one at a time on the machine, memory cap, timeout). Registration refuses a file
not committed at HEAD and a trial needs a clean tree, so the user commits the two files: by hand with the
command shown, or with the "Commit these two files" button behind its own confirm box. The
outcome is kept in st.session_state by the strategy's sha256, so a rerun shows it again; render returns the
last trial's run_id for step 5.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from pfb.dash import actions
from pfb.dash.actions import DryRun, StepResult, StrategyState
from pfb.dash.data import DashConfig, LedgerUnreadable, LedgerView, ledger_view
from pfb.dash.hypgen import (
    EVENT_CHOICES,
    HypForm,
    HypGenError,
    PowerSource,
    budget_of,
    build,
    grid_of,
    hypothesis_id,
    missing_fields,
    power_preview,
    power_source,
    shape_error,
    windows_of,
)
from pfb.dash.views import GAIN, LOSS, MUTED
from pfb.registry.api import read_state

STATE_KEY = "pfb_run_step"
FIRMS_REL = "config/firms"
SYNTHETIC = "synthetic"
DOT = "&#9679;"
EDGE_STEP_CENTS = 100
COMMIT_MESSAGE = "hypothesis {hyp_id}: strategy upload {sha8} from the dashboard"
SHA8 = 8
COMMIT_CONFIRM = "Commit the upload and the hypothesis above to git, with the message shown"
# before the commit the dry run's InputNotCommitted is expected, not a fail, and the
# dry run reruns by itself once the two files are committed, so no stale result stays on screen.
NOT_COMMITTED_CODE = "InputNotCommitted"
NOT_COMMITTED_YET = "expected until the two files are committed; Validate reruns after the commit"
NO_SOURCE = "No ok synthetic run of this strategy with power figures in the ledger: run step 2's check."


# --------------------------------------------------------------------------------- read-only ----
def plan_choices(repo_root: Path) -> list[str]:
    """Every "<firm>/<plan stem>" with a plan file config/firms/<firm>/<plan>.toml."""
    firms = repo_root / FIRMS_REL
    if not firms.is_dir():
        return []
    return sorted(f"{p.parent.name}/{p.stem}" for p in firms.glob("*/*.toml"))


def synthetic_source(view: LedgerView, strategy_sha256: str) -> PowerSource | None:
    """The newest ok synthetic run of this strategy with realized power figures (step 2's run)."""
    begun = sorted(view.begun.items(), key=lambda kv: str(kv[1].get("time_utc", "")), reverse=True)
    for run_id, begin in begun:
        if begin.get("kind") != SYNTHETIC or begin.get("strategy_sha256") != strategy_sha256:
            continue
        found = power_source(run_id, view.results.get(run_id))
        if found is not None:
            return found
    return None


def is_registered(config: DashConfig, hyp_id: str) -> bool:
    return hyp_id in read_state(config.registry_config()).hypotheses


def window_label(window: Mapping[str, Any]) -> str:
    return f"{window.get('first')}..{window.get('last')}"


def confirm_text(n: int, trials: int) -> str:
    return f"This run counts as trial N {n} -> {n + trials}"


# ------------------------------------------------------------------------------------ drawing ----
def _dot(ok: bool, title: str, message: str) -> None:
    colour, word = (GAIN, "pass") if ok else (LOSS, "fail")
    st.markdown(
        f'<span style="color:{colour}">{DOT}</span> <b>{html.escape(title)}</b> ({word}): '
        f"<code>{html.escape(message)}</code>",
        unsafe_allow_html=True,
    )


def _session(sha256: str) -> dict[str, Any]:
    store: dict[str, dict[str, Any]] = st.session_state.setdefault(STATE_KEY, {})
    return store.setdefault(sha256, {})


def _picks(
    spec: Mapping[str, Any], windows: Sequence[Mapping[str, Any]], config: DashConfig
) -> tuple[list[Mapping[str, Any]], list[str]]:
    usable = [w for w in windows if w.get("first") and w.get("last")]
    labels = [window_label(w) for w in usable]
    picked = st.multiselect("Open windows (one trial each)", labels, default=labels[-1:], key="run_windows")
    choices = plan_choices(config.repo_root)
    plans = st.multiselect("Firm plans (prop Monte Carlo of the trial)", choices, key="run_plans")
    return [w for w in usable if window_label(w) in picked], list(plans)


GRID_LABEL = "SPEC defaults (budget {defaults})", "full SPEC grid (budget {full})"
TARGET_LABEL = "target_edge_cents (net edge per trade claimed)"
STANCE_LABEL = "event_stance (why event minutes do not matter; required when off)"


def _form(state: StrategyState, source: PowerSource, n_windows: int) -> HypForm | None:
    spec = state.spec or {}
    defaults = budget_of(grid_of(spec, False), n_windows)
    full = budget_of(grid_of(spec, True), n_windows)
    labels = [label.format(defaults=defaults, full=full) for label in GRID_LABEL]
    grid_choice = st.radio("Grid", labels, key="run_grid", horizontal=True)
    statement = st.text_area("Statement (what this run tests)", key="run_statement")
    target = st.number_input(TARGET_LABEL, min_value=0, step=EDGE_STEP_CENTS, key="run_target")
    event_costs = st.radio("event_costs", EVENT_CHOICES, key="run_events", horizontal=True)
    stance = st.text_input(STANCE_LABEL, key="run_stance")
    missing = missing_fields(statement, int(target), str(event_costs), stance)
    if missing:
        st.caption(f"To fill in: {', '.join(missing)}.")
        return None
    return HypForm(
        hypothesis_id=hypothesis_id(state.sha256, datetime.now(UTC)),
        statement=statement,
        target_edge_cents=int(target),
        event_costs=str(event_costs),
        event_stance=stance,
        full_grid=grid_choice == labels[1],
        power=source,
    )


def _power_lines(form: HypForm, windows: Sequence[Mapping[str, Any]], n: int, budget: int) -> None:
    for row in power_preview(form, windows, n + budget):
        edge = "undetectable" if row.min_edge_cents is None else f"{row.min_edge_cents} cents"
        text = f"minimum detectable edge at N {row.n_used}: {edge}; target {form.target_edge_cents}"
        _dot(row.passed, f"power {row.first}..{row.last}", text)


def _build(
    state: StrategyState,
    windows: Sequence[Mapping[str, Any]],
    plans: list[str],
    form: HypForm,
    config: DashConfig,
) -> None:
    try:
        text = build(state, windows, plans, form)
    except HypGenError as exc:
        st.error(f"Not built: {exc}")
        return
    error = shape_error(text, form.hypothesis_id)
    if error is not None:
        st.error(f"Not built: {error}")
        return
    try:
        saved = actions.save_hypothesis(text, form.hypothesis_id, config.repo_root)
    except actions.HypothesisRefused as exc:
        st.error(f"Not saved: {exc}")
        return
    session = _session(state.sha256)
    session.clear()
    session.update(hyp_id=form.hypothesis_id, rel=saved.rel_path, text=text, windows=len(windows_of(windows)))


def _dry_run(session: dict[str, Any], config: DashConfig, ready: bool) -> None:
    clicked = st.button("Validate (pfb hypothesis register --dry-run; N unchanged)", key="run_dry")
    stale = "dry" in session and session.get("dry_ready") != ready
    if clicked or stale:
        with st.spinner("Checking the hypothesis: every registration refusal, nothing appended"):
            session["dry"] = actions.dry_run(config.repo_root / session["rel"], config=config)
            session["dry_ready"] = ready
    dry: DryRun | None = session.get("dry")
    if dry is None:
        return
    if dry.ok:
        _dot(True, "registration check", dry.message)
    elif not dry.refusals:
        _dot(False, "registration check", dry.message)
    for refusal in dry.refusals:
        if refusal.code == NOT_COMMITTED_CODE and not ready:
            _expected(refusal.code, NOT_COMMITTED_YET)
        else:
            _dot(False, refusal.code, refusal.reason)


def _expected(title: str, message: str) -> None:
    st.markdown(
        f'<span style="color:{MUTED}">{DOT}</span> <b>{html.escape(title)}</b> ({html.escape(message)})',
        unsafe_allow_html=True,
    )


def _git_state(state: StrategyState, session: dict[str, Any], config: DashConfig) -> actions.CommitState:
    return actions.commit_state(config.repo_root, (state.rel_path, session["rel"]))


def _commit(
    state: StrategyState, session: dict[str, Any], config: DashConfig, git: actions.CommitState
) -> bool:
    paths = (state.rel_path, session["rel"])
    for rel, ok in git.committed:
        st.markdown(f"`{rel}` committed: **{'yes' if ok else 'no'}**")
    if git.error or git.dirty:
        _dot(False, "git", git.text())
    if not git.ready:
        _commit_button(state, session, config, paths)
    return git.ready


def _commit_button(
    state: StrategyState, session: dict[str, Any], config: DashConfig, paths: tuple[str, str]
) -> None:
    """Commit the two files from here after a confirm box, or by hand with the same command."""
    base = COMMIT_MESSAGE.format(hyp_id=session["hyp_id"], sha8=state.sha256[:SHA8])
    message = actions.commit_message(base)
    st.caption("The two files are committed with this command and message (nothing else is committed):")
    st.code(actions.commit_command(paths, message), language="bash")
    agreed = st.checkbox(COMMIT_CONFIRM, key=f"run_commit_confirm:{session['hyp_id']}")
    left, right = st.columns(2)
    if left.button("Commit these two files", key="run_commit", disabled=not agreed):
        session["commit"] = actions.commit_files(config.repo_root, paths[0], paths[1], message)
        if session["commit"].ok:
            st.rerun()  # redraw with the new HEAD: both files show committed and step 4 opens
    right.button("Refresh", key="run_refresh")
    done: actions.CommitDone | None = session.get("commit")
    if done is not None:
        _dot(done.ok, "commit", done.message)


def _steps(steps: Sequence[StepResult]) -> None:
    for step in steps:
        _dot(step.ok, step.name, step.message)


def _confirm(state: StrategyState, session: dict[str, Any], config: DashConfig, n: int) -> None:
    trials = int(session["windows"])
    agreed = st.checkbox(confirm_text(n, trials), key=f"run_confirm:{session['hyp_id']}")
    if not st.button("Register and run", key="run_go", disabled=not agreed):
        return
    registered = is_registered(config, session["hyp_id"])
    with st.status("Running", expanded=True) as status:
        steps = actions.run_trials(state.rel_path, session["rel"], list(range(1, trials + 1)), config=config,
                                   registered=registered, on_step=status.write)  # fmt: skip
        failed = not all(step.ok for step in steps)
        status.update(label="Failed" if failed else "Done", state="error" if failed else "complete")
    session["steps"] = steps
    run_ids = [step.run_id for step in steps if step.run_id]
    if run_ids:
        session["run_id"] = run_ids[-1]


def render(config: DashConfig, state: StrategyState, windows: list[dict[str, Any]]) -> str | None:
    """Draw step 4; the last trial's run_id once a run was made from this page, else None."""
    if state.spec is None or not state.passed:
        st.caption("Locked until the check in step 2 passes.")
        return None
    try:
        view = ledger_view(config)
    except LedgerUnreadable as exc:
        st.error(f"The ledger cannot be read, so nothing can be run: {exc}")
        return None
    session = _session(state.sha256)
    if session.get("steps"):
        _steps(session["steps"])
        st.code(session["text"], language="toml")
        if st.button("Start a new hypothesis", key="run_reset"):
            session.clear()
        return session.get("run_id")
    source = synthetic_source(view, state.sha256)
    if source is None:
        st.warning(
            "No ok synthetic run of this strategy with power figures in the ledger: run step 2's check."
        )
        return None
    st.markdown(f'<span style="color:{MUTED}">N in the ledger: {view.n}</span>', unsafe_allow_html=True)
    picked, plans = _picks(state.spec, windows, config)
    if not picked:
        st.caption("Pick at least one open window.")
        return None
    st.caption(f"[power] from synthetic run {source.run_id}: {source.trades} trades in {source.days} days, "
               f"sd {source.trade_sd_cents} cents, skew {source.trade_skew:.3f}, "
               f"kurtosis {source.trade_kurtosis:.3f}.")  # fmt: skip
    form = _form(state, source, len(picked))
    if form is not None:
        budget = budget_of(grid_of(state.spec, form.full_grid), len(picked))
        _power_lines(form, picked, view.n, budget)
        if st.button("Build the hypothesis", key="run_build"):
            _build(state, picked, plans, form, config)
    if "rel" not in session:
        return None
    st.markdown(f"Hypothesis `{session['rel']}`")
    st.code(session["text"], language="toml")
    git = _git_state(state, session, config)
    _dry_run(session, config, git.ready)
    if _commit(state, session, config, git):
        _confirm(state, session, config, view.n)
    if session.get("steps"):
        _steps(session["steps"])
    return session.get("run_id")
