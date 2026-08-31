"""
Offline training code for the innovX cross-domain map representation.

This package is deliberately separate from ``app``: the FastAPI server never
imports it, and nothing here runs at application startup. Datasets and model
weights are never committed to Git.
"""
