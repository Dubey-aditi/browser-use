"""Action tracing — records what the agent did in a form that can be replayed
deterministically, so a task solved once by an LLM need not be solved again.

Enabled by setting `BROWSER_USE_ACTION_TRACE_PATH` (to a file) or
`BROWSER_USE_ACTION_TRACE_MEMORY=1` (in memory); otherwise nothing here runs.
"""

from browser_use.cache.locators import apply_frame_chain, looks_dynamic, scored_candidates
from browser_use.cache.service import (
	TRACE_MEMORY_ENV,
	TRACE_PATH_ENV,
	VERIFY_ENV,
	ActionTraceRecorder,
	clear_recorded_actions,
	recorded_actions,
	set_trace_context,
)
from browser_use.cache.views import ElementSignals, LocatorCandidate, TracedAction

__all__ = [
	'TRACE_MEMORY_ENV',
	'TRACE_PATH_ENV',
	'VERIFY_ENV',
	'ActionTraceRecorder',
	'ElementSignals',
	'LocatorCandidate',
	'TracedAction',
	'apply_frame_chain',
	'clear_recorded_actions',
	'looks_dynamic',
	'recorded_actions',
	'scored_candidates',
	'set_trace_context',
]
