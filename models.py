"""How a model name of the plan is sent to the provider.

The competition proxy has separate models without reasoning, named with the suffix -noreason. Other providers,
such as OpenRouter, have one model and a request option instead. With NOREASON_VIA_REQUEST=1 a name ending in
-noreason is sent as the base model with reasoning turned off, so the same four names can be planned, run and
measured there. Without the variable every name goes out unchanged, as the proxy expects.
"""
import os

NOREASON_SUFFIX = "-noreason"


def request_model(name):
    """The model name to put into the request and the extra request body, or None when there is none."""
    if name.endswith(NOREASON_SUFFIX) and os.getenv("NOREASON_VIA_REQUEST", "0") not in ("", "0"):
        return name[:-len(NOREASON_SUFFIX)], {"reasoning": {"enabled": False}}
    return name, None
