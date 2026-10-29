"""Grounded synthetic-item generator: pure code only (facts, render, verify, manifest).

Modules: ``facts`` (typed facts, case identity, template specs), ``render`` (the request text a local model
receives, built from facts only), ``verify`` (fact-presence and banned-token checks with reason codes),
``manifest`` (per-family counts and keep rates, never item text). No model, network call or data file is used here.
"""
