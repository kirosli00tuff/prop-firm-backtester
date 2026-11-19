"""Read-only Streamlit dashboard (`pfb dash`).

pfb.dash.data reads the ledger (pfb.registry.api.read_state and status) and the artifacts under runs/; the
view modules render them; app.py and pages/ are the Streamlit scripts. Nothing here writes, begins a run or
reads a bar store.
"""
