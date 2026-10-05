"""Model action selection for the evaluation loop."""

from __future__ import annotations

import numpy as np
import torch


def select_action(model, obs, *, deterministic: bool, device):
    """Return the flat numpy action for ``obs`` from any supported model.

    * CleanRL agents (``model.model_type == "CleanRL"``): batch the obs, call
      ``get_action`` on ``device`` and flatten the result.
    * Everything else (PyWake / SB3-style): ``model.predict(obs, deterministic)[0]``.
    """
    if hasattr(model, "model_type"):
        if model.model_type == "CleanRL":
            obs = np.expand_dims(obs, 0)
            action, _, _ = model.get_action(
                torch.Tensor(obs).to(device), deterministic=deterministic
            )
            action = action.detach().cpu().numpy()
            return action.flatten()
        raise ValueError(
            f"Unsupported model_type {model.model_type!r}: expected 'CleanRL' "
            "or a model exposing predict(obs, deterministic)."
        )
    # This is for the other models (Pywake and such)
    return model.predict(obs, deterministic=deterministic)[0]
