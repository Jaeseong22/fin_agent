#!/usr/bin/env python3
"""Simple LangSmith run helper with no-op when API key missing."""
import os
from typing import Any, Dict

try:
    from langsmith import Client
except Exception:
    Client = None


def _get_client():
    key = os.environ.get('LANGSMITH_API_KEY')
    if not key or Client is None:
        return None
    return Client(api_key=key)


def start_run(name: str, input_data: Dict[str, Any]):
    """Start a LangSmith run. Returns run object or None."""
    client = _get_client()
    if client is None:
        return None
    try:
        run = client.create_run(name=name, inputs=input_data, run_type="tool")
    except TypeError:
        # fallback for older/newer client signatures
        run = client.create_run(name=name)
        try:
            run.log_input(input_data)
        except Exception:
            pass
    return run


def finish_run(run, output_data: Dict[str, Any]):
    """Log output and close run. Safe no-op if run is None."""
    if run is None:
        return
    try:
        run.log_output(output_data)
    except Exception:
        pass
    try:
        run.close()
    except Exception:
        pass
