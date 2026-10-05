"""Building blocks of ``eval_single_fast`` / ``AgentEval``.

* ``recorder``: ``EpisodeRecorder`` collects the per-sim-step arrays and
  turns them into the evaluation ``xarray.Dataset``.
* ``live_figures``: ``LiveFigureWriter`` renders the per-step flow/timeseries
  frames written with ``save_figs=True``.
* ``hawc2_loads``: HAWC2 load-channel extraction and teardown for
  ``return_loads=True``.
* ``selection``: ``select_action`` picks the model API (CleanRL vs
  ``predict``).
"""

from .recorder import EpisodeRecorder
from .live_figures import LiveFigureWriter, resolve_fig_folder
from .hawc2_loads import (
    HAWC2_LOAD_COLUMNS,
    load_hawc2_loads_dataset,
    close_hawc2_and_cleanup,
)
from .selection import select_action

__all__ = [
    "EpisodeRecorder",
    "LiveFigureWriter",
    "resolve_fig_folder",
    "HAWC2_LOAD_COLUMNS",
    "load_hawc2_loads_dataset",
    "close_hawc2_and_cleanup",
    "select_action",
]
