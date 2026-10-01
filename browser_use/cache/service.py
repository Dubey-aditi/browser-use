"""Record every action the agent takes, with enough detail to replay it without an LLM.

Hooked into `Tools.act`, the one point every action passes through. Capturing there
rather than from `AgentHistoryList` afterwards buys two things that cannot be had
later: the full DOM node (the history keeps a reduced copy with no accessibility
data), and a live page to count locator matches against.

Off unless `BROWSER_USE_ACTION_TRACE_PATH` (append to a JSONL file) or
`BROWSER_USE_ACTION_TRACE_MEMORY=1` (keep in this process; read with
`recorded_actions()`) is set. Every recording path is guarded — a failure here logs
and returns, and never touches the action it is observing.
"""

import hashlib
import json
import logging
import os
from contextvars import ContextVar
from pathlib import Path

from browser_use.browser.session import BrowserSession
from browser_use.cache.locators import apply_frame_chain, looks_dynamic, scored_candidates
from browser_use.cache.views import ElementSignals, LocatorCandidate, TracedAction
from browser_use.dom.views import EnhancedDOMTreeNode

logger = logging.getLogger(__name__)

TRACE_PATH_ENV = 'BROWSER_USE_ACTION_TRACE_PATH'
TRACE_MEMORY_ENV = 'BROWSER_USE_ACTION_TRACE_MEMORY'
VERIFY_ENV = 'BROWSER_USE_ACTION_TRACE_VERIFY'

# In-memory traces, shared by every recorder in the process: an agent run and the
# per-step fallbacks around it use different `Tools` instances but one trace.
_recorded: list[TracedAction] = []


def recorded_actions() -> list[TracedAction]:
	"""Every action recorded in this process so far, oldest first."""
	return list(_recorded)


def clear_recorded_actions() -> None:
	_recorded.clear()


# Set by the caller around a unit of work (Optexity: one automation node) and stamped
# onto every action recorded inside it. A ContextVar rather than recorder state because
# the recorder is owned by `Tools`, which the caller neither creates nor holds.
_trace_context: ContextVar[dict[str, str | int] | None] = ContextVar('browser_use_trace_context', default=None)


def set_trace_context(**context: str | int) -> None:
	"""Tag every action recorded from now on, in this task, with `context`."""
	_trace_context.set(dict(context) or None)


# Counts matches the way Playwright's CSS engine does — descending into open shadow
# roots, which `document.querySelectorAll` alone does not. Without this an element
# inside a shadow root reports zero matches and every candidate looks broken.
_COUNT_MATCHES_JS = """
(() => {
	const selector = %s;
	let total = 0;
	const walk = (root) => {
		total += root.querySelectorAll(selector).length;
		for (const element of root.querySelectorAll('*')) {
			if (element.shadowRoot) walk(element.shadowRoot);
		}
	};
	try { walk(document); } catch (error) { return -1; }
	return total;
})()
"""

# Counts what an exact `get_by_role(role, name=..., exact=True)` or
# `get_by_text(..., exact=True)` would match, again piercing open shadow roots. The
# accessible name is approximated (aria-label, then label, then text, value, title)
# and hidden elements are not excluded. Both errors push the count up, never down,
# so an approximation can cost a usable locator but cannot make an ambiguous one
# look unique.
_COUNT_EXACT_JS = """
(() => {
	const role = %s;
	const text = %s;
	const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
	const implicit = {
		button: 'button,input[type=button],input[type=submit],input[type=reset]',
		link: 'a[href]',
		option: 'option',
		checkbox: 'input[type=checkbox]',
		radio: 'input[type=radio]',
		textbox: 'input:not([type]),input[type=text],input[type=email],input[type=search],input[type=tel],input[type=url],input[type=password],textarea',
		combobox: 'select',
		heading: 'h1,h2,h3,h4,h5,h6',
		cell: 'td',
		gridcell: 'td',
	};
	const label = (el) => (el.labels && el.labels.length ? norm(el.labels[0].textContent) : '');
	const names = (el) => [norm(el.getAttribute('aria-label')), label(el), norm(el.textContent), norm(el.value), norm(el.getAttribute('title'))];
	let total = 0;
	const walk = (root) => {
		if (role) {
			const selector = `[role="${role}"]` + (implicit[role] ? ',' + implicit[role] : '');
			for (const el of root.querySelectorAll(selector)) {
				const explicit = el.getAttribute('role');
				if (explicit && explicit !== role) continue;
				if (names(el).includes(text)) total++;
			}
		} else {
			for (const el of root.querySelectorAll('*')) {
				if (['SCRIPT', 'STYLE', 'TEMPLATE'].includes(el.tagName)) continue;
				if (norm(el.textContent) !== text) continue;
				// Playwright resolves text to the innermost element that holds it.
				if ([...el.children].some((child) => norm(child.textContent) === text)) continue;
				total++;
			}
		}
		for (const el of root.querySelectorAll('*')) {
			if (el.shadowRoot) walk(el.shadowRoot);
		}
	};
	try { walk(document); } catch (error) { return -1; }
	return total;
})()
"""

# Actions that carry no element and cannot be replayed from a locator.
_ELEMENTLESS_ACTIONS = frozenset(
	{'done', 'wait', 'screenshot', 'scroll', 'go_back', 'search', 'navigate', 'switch', 'close', 'send_keys', 'find_text'}
)


class ActionTraceRecorder:
	"""Records one `TracedAction` per action: appended to a JSONL file, or, with no
	path, kept in memory for `recorded_actions()`."""

	def __init__(self, path: Path | None = None, verify: bool = True) -> None:
		self.path = path
		self.verify = verify
		self._step = 0
		if self.path is not None:
			self.path.parent.mkdir(parents=True, exist_ok=True)

	@classmethod
	def from_env(cls) -> 'ActionTraceRecorder | None':
		"""The recorder configured by the environment, or None when tracing is off."""
		verify = os.getenv(VERIFY_ENV, '1').lower() not in ('0', 'false', 'no')
		path = os.getenv(TRACE_PATH_ENV)
		if not path:
			if os.getenv(TRACE_MEMORY_ENV, '').lower() not in ('1', 'true', 'yes'):
				return None
			logger.debug(f'recording action trace in memory (verify={verify})')
			return cls(None, verify=verify)
		try:
			recorder = cls(Path(path).expanduser(), verify=verify)
		except OSError as error:
			logger.warning(f'action trace disabled, cannot use {path}: {error}')
			return None
		logger.info(f'📝 recording action trace to {recorder.path} (verify={verify})')
		return recorder

	# region - capture

	async def before(
		self,
		browser_session: BrowserSession,
		action_name: str,
		params: dict,
	) -> TracedAction:
		"""Snapshot the element and page state an action is about to act on."""
		self._step += 1
		traced = TracedAction(
			# Memory mode numbers across recorders, since they share one trace.
			step=self._step if self.path is not None else len(_recorded) + 1,
			action_name=action_name,
			params=_safe_params(params),
			context=_trace_context.get() or {},
		)

		try:
			traced.url_before = await browser_session.get_current_page_url()
		except Exception as error:
			logger.debug(f'trace: no url before {action_name}: {type(error).__name__}: {error}')

		index = params.get('index') if isinstance(params, dict) else None
		if not isinstance(index, int) or action_name in _ELEMENTLESS_ACTIONS:
			return traced
		traced.index = index

		try:
			node = await browser_session.get_dom_element_by_index(index)
		except Exception as error:
			logger.debug(f'trace: index {index} not resolvable: {type(error).__name__}: {error}')
			return traced
		if node is None:
			return traced

		try:
			traced.element = _signals_from_node(node)
			traced.candidates = scored_candidates(traced.element)
			if traced.element.frame_chain:
				for candidate in traced.candidates:
					candidate.locator = apply_frame_chain(candidate.locator, traced.element.frame_chain)
		except Exception as error:
			logger.debug(f'trace: could not derive locators: {type(error).__name__}: {error}')
			return traced

		if self.verify:
			await self._verify(browser_session, node, traced.candidates)

		return traced

	async def after(self, traced: TracedAction, result: object, duration_ms: float, browser_session: BrowserSession) -> None:
		"""Complete the record with the outcome and append it to the trace."""
		traced.duration_ms = round(duration_ms, 1)
		traced.error = getattr(result, 'error', None)
		content = getattr(result, 'extracted_content', None)
		traced.extracted_content = content[:500] if isinstance(content, str) else None

		try:
			traced.url_after = await browser_session.get_current_page_url()
		except Exception as error:
			logger.debug(f'trace: no url after {traced.action_name}: {type(error).__name__}: {error}')

		if self.path is None:
			_recorded.append(traced)
		else:
			try:
				with self.path.open('a', encoding='utf-8') as handle:
					handle.write(traced.model_dump_json(exclude_none=True) + '\n')
			except OSError as error:
				logger.warning(f'trace: failed to append step {traced.step}: {error}')
				return

		best = traced.best_verified
		if best is not None:
			logger.debug(f'trace: step {traced.step} {traced.action_name} -> {best.locator} ({best.kind}, unique)')
		elif traced.candidates:
			counts = ', '.join(f'{c.kind}={c.match_count}' for c in traced.candidates[:3])
			logger.debug(f'trace: step {traced.step} {traced.action_name} -> no unique candidate ({counts})')

	# endregion - capture

	async def _verify(
		self,
		browser_session: BrowserSession,
		node: EnhancedDOMTreeNode,
		candidates: list[LocatorCandidate],
	) -> None:
		"""Count how many elements each verifiable candidate really matches.

		Runs in the node's own frame, so a count for an element inside an iframe is
		taken inside that iframe rather than against the top document.
		"""
		verifiable = [c for c in candidates if c.verify_selector or c.verify_text]
		if not verifiable:
			return

		try:
			cdp_session = await browser_session.cdp_client_for_node(node)
		except Exception as error:
			logger.debug(f'trace: no cdp session for node, skipping verification: {type(error).__name__}: {error}')
			return

		for candidate in verifiable:
			expression = (
				_COUNT_MATCHES_JS % json.dumps(candidate.verify_selector)
				if candidate.verify_selector
				else _COUNT_EXACT_JS % (json.dumps(candidate.verify_role), json.dumps(candidate.verify_text))
			)
			try:
				response = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': expression,
						'returnByValue': True,
						'awaitPromise': True,
					},
					session_id=cdp_session.session_id,
				)
				if response.get('exceptionDetails'):
					continue
				value = response.get('result', {}).get('value')
				if isinstance(value, int) and value >= 0:
					candidate.match_count = value
			except Exception as error:
				logger.debug(f'trace: could not count {candidate.locator!r}: {type(error).__name__}: {error}')


def _safe_params(params: object) -> dict:
	"""Action parameters, JSON-safe and without long payloads."""
	if not isinstance(params, dict):
		return {}
	safe: dict = {}
	for key, value in params.items():
		if isinstance(value, (str, int, float, bool)) or value is None:
			safe[key] = value[:500] if isinstance(value, str) else value
		else:
			safe[key] = str(value)[:500]
	return safe


def _signals_from_node(node: EnhancedDOMTreeNode) -> ElementSignals:
	"""Reduce a DOM node to the properties that survive to the next run."""
	text = ''
	try:
		text = (node.get_meaningful_text_for_llm() or '').strip()
	except Exception:
		text = ''

	in_shadow_dom, frame_chain, anchor_id, anchor_tag = _ancestry(node)

	# Identity of this element, not of its branch: `parent_branch_hash()` is shared by
	# siblings, so two inputs in the same row collapse into one. The document path is
	# unique per element, and qualified by frame it stays unique across iframes.
	element_hash = hashlib.sha1(f'{node.target_id}|{node.frame_id}|{node.xpath}|{node.tag_name}'.encode()).hexdigest()[:16]

	return ElementSignals(
		tag_name=(node.tag_name or '').lower(),
		attributes=dict(node.attributes or {}),
		xpath=node.xpath or '',
		role=(node.ax_node.role if node.ax_node else None) or None,
		accessible_name=(node.ax_node.name if node.ax_node else None) or None,
		text=text[:200] or None,
		frame_id=node.frame_id,
		target_id=node.target_id,
		in_shadow_dom=in_shadow_dom,
		frame_chain=frame_chain,
		anchor_id=anchor_id,
		anchor_tag=anchor_tag,
		element_hash=element_hash,
	)


# How far up to look for a stable id when an element has none of its own. Far enough
# for a custom widget's wrapper (react-select puts `id="state"` four levels above the
# input), close enough that the id still names this control and not the whole form.
_ANCHOR_MAX_DEPTH = 4
# Containers whose id names a region, not a control: clicking one hits whatever sits
# at its centre.
_NOT_ANCHORS = frozenset(
	{'html', 'body', 'form', 'main', 'section', 'article', 'nav', 'header', 'footer', 'table', 'tbody', 'ul', 'ol'}
)


def _ancestry(node: EnhancedDOMTreeNode) -> tuple[bool, list[str], str | None, str | None]:
	"""Walk to the root once, collecting what changes how an element must be
	addressed: whether a shadow root sits above it, which iframes enclose it, and the
	nearest close ancestor with a stable id (with its tag), for elements that have no
	stable handle of their own.

	Returns `(in_shadow_dom, frame_chain, anchor_id, anchor_tag)` with the chain
	ordered outermost-first, matching the order the hops are chained in Playwright.
	"""
	in_shadow_dom = False
	frames: list[str] = []
	anchor_id: str | None = None
	anchor_tag: str | None = None

	current = node.parent_node
	depth = 0
	while current is not None and depth < 100:
		depth += 1
		if getattr(current, 'shadow_root_type', None) is not None:
			in_shadow_dom = True
		is_iframe = (current.node_name or '').upper() == 'IFRAME'
		if is_iframe:
			frames.append(_frame_locator_for(current))
		# The anchor must be in the element's own document, so the search stops at the
		# first frame boundary.
		if anchor_id is None and not frames and depth <= _ANCHOR_MAX_DEPTH:
			tag = (current.tag_name or '').lower()
			element_id = ((current.attributes or {}).get('id') or '').strip()
			if element_id and tag not in _NOT_ANCHORS and not looks_dynamic(element_id):
				anchor_id, anchor_tag = element_id, tag
		current = current.parent_node

	frames.reverse()
	return in_shadow_dom, frames, anchor_id, anchor_tag


def _frame_locator_for(iframe: EnhancedDOMTreeNode) -> str:
	"""A `frame_locator(...)` hop addressing this iframe element.

	Uses the same stability rules as any other element, falling back to the first
	iframe on the page — imprecise, but a caller reading the trace can see the
	selector and judge it, which beats emitting nothing and silently dropping the hop.
	"""
	attributes = iframe.attributes or {}
	for attribute in ('name', 'id', 'title'):
		value = (attributes.get(attribute) or '').strip()
		if value and not looks_dynamic(value):
			return f'frame_locator("iframe[{attribute}=\'{value}\']")'

	source = (attributes.get('src') or '').strip()
	if source and not looks_dynamic(source):
		return f'frame_locator("iframe[src=\'{source}\']")'

	return 'frame_locator("iframe")'
