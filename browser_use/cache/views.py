"""Data model for the action trace.

One `TracedAction` is appended per executed action. The trace is the raw material a
compiler turns into a deterministic automation, so it records more than any single
consumer needs: the element's own signals, every locator candidate derived from
them, and how many elements each candidate actually matched on the live page.

Over-recording here is deliberate. The page is gone by the time anything reads this
file, so a signal not captured at action time cannot be recovered.
"""

from pydantic import BaseModel, ConfigDict, Field


class LocatorCandidate(BaseModel):
	"""One way to address an element, with how trustworthy it looks and how many
	elements it really matched."""

	model_config = ConfigDict(extra='forbid')

	locator: str
	"""Playwright expression relative to a page or frame, e.g. `locator("#email")`."""

	kind: str
	"""Which signal produced it: test-id, id, name, aria-label, role+name, ..."""

	score: int
	"""Stability heuristic, 0-100. Higher means less likely to change between renders.
	Says nothing about whether the locator is unique — that is `match_count`."""

	verify_selector: str | None = None
	"""CSS equivalent used to count matches. None for the `get_by_role` /
	`get_by_text` family, which is counted from `verify_role` / `verify_text`."""

	verify_role: str | None = None
	verify_text: str | None = None
	"""For an exact role / text locator: the role (None for plain text) and the exact
	name or text, counted on the page the way Playwright's exact matching does."""

	match_count: int | None = None
	"""Elements matched on the live page at action time, piercing open shadow roots.
	1 is safe to cache. None means it could not be checked, not that it is unique."""


class ElementSignals(BaseModel):
	"""Everything about the element that outlives this run.

	The DOM index the LLM chose is a position in one serialization and is worthless
	on the next run; these are the properties that are not."""

	model_config = ConfigDict(extra='forbid')

	tag_name: str
	attributes: dict[str, str] = Field(default_factory=dict)
	xpath: str = ''
	role: str | None = None
	accessible_name: str | None = None
	text: str | None = None

	frame_id: str | None = None
	target_id: str | None = None

	in_shadow_dom: bool = False
	"""Open shadow root somewhere above this element. Playwright's CSS engine pierces
	those; XPath does not — so an xpath candidate here is dead on arrival."""

	frame_chain: list[str] = Field(default_factory=list)
	"""`frame_locator(...)` expressions, outermost first, for an element inside one or
	more iframes. Empty for the main frame. A `page.locator(...)` cannot cross an
	iframe boundary, so a non-empty chain must be prefixed onto the final command."""

	anchor_id: str | None = None
	anchor_tag: str | None = None
	"""The nearest close ancestor with a stable id, for an element with no stable
	handle of its own — react-select's control sits inside `<div id="state">`."""

	element_hash: str | None = None
	"""Stable identity across steps within a run, for collapsing repeated actions on
	the same element."""


class TracedAction(BaseModel):
	"""One executed action, with the context needed to replay it without an LLM."""

	model_config = ConfigDict(extra='forbid')

	step: int
	action_name: str
	params: dict = Field(default_factory=dict)
	index: int | None = None

	element: ElementSignals | None = None
	candidates: list[LocatorCandidate] = Field(default_factory=list)

	url_before: str | None = None
	url_after: str | None = None
	"""Differs from `url_before` when the action navigated. A compiler uses this to
	tell a meaningful click from one that did nothing."""

	duration_ms: float = 0.0
	error: str | None = None
	extracted_content: str | None = None

	context: dict[str, str | int] = Field(default_factory=dict)
	"""Whatever the caller had set with `set_trace_context` when the action ran — for
	Optexity, the automation node it belongs to. Lets a consumer map an action back to
	the step that caused it, so one node can be repaired without recompiling the rest."""

	@property
	def succeeded(self) -> bool:
		return self.error is None

	@property
	def best_verified(self) -> LocatorCandidate | None:
		"""Highest-scoring candidate confirmed to match exactly one element.

		Deliberately ignores unverified candidates however well they score: a locator
		that silently matches fourteen table rows is the failure mode this whole
		exercise exists to avoid."""
		unique = [c for c in self.candidates if c.match_count == 1]
		return max(unique, key=lambda c: c.score) if unique else None
