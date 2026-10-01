"""Derive Playwright locators for a DOM element, ranked by how stable they look.

The ranking is the same idea as Optexity's `LocatorExtraction`, with two changes:

* `looks_dynamic` no longer rejects a value merely for containing two digits. That
  rule discarded `04fullname`, `10address1` and every other hand-numbered form field,
  leaving only the positional xpath (see FINDINGS.md F1). Digits at either end of a
  segment are an ordering convention; digits in the middle are the hash signature.
* the xpath candidate is dropped inside an open shadow root rather than ranked last.
  Playwright's CSS engine pierces open shadow roots and XPath does not, so an xpath
  built by walking `parentNode` across that boundary addresses a node that is not
  reachable from `document`. Ranking a locator that cannot resolve is worse than
  omitting it, because it occupies the fallback slot.

A score is a guess about *stability*. It says nothing about whether the locator is
unique — that is measured separately, against the live page, in `service.py`.
"""

import re

from browser_use.cache.views import ElementSignals, LocatorCandidate

# Tokens frameworks regenerate between renders or builds: React useId, styled-components
# and emotion hashes (`css-13cymwt-control` is a hash with a label), Ember/Radix/
# HeadlessUI ids.
#
# `react-` only when a counter follows, as in react-select's `react-select-3-input`. A
# bare `react-` prefix also caught hand-written BEM classes such as
# `react-datepicker__year-select`, leaving that element with nothing but an xpath.
_DYNAMIC_PREFIX_RE = re.compile(
	r'^(?::r[0-9a-z]*:?$|react-[a-z_-]*\d|ember\d|radix-|headlessui-|jss\d|sc-|css-[a-z0-9]+$|css-(?=[a-z0-9]*\d)[a-z0-9]{5,}-|emotion-)',
	re.IGNORECASE,
)
_UUID_RE = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

_TEST_ID_ATTRS = ('data-testid', 'data-test-id', 'data-test', 'data-cy', 'data-qa')

# Roles `get_by_role` accepts. The accessibility tree also reports Chromium-internal
# roles (LabelText, StaticText, generic, ...) that Playwright rejects; treating those
# as roles produced a locator that can never match and suppressed the text locator
# the element did have — which is how a clicked <label>Music</label> ended up with
# nothing usable.
_ARIA_ROLES = frozenset(
	{
		'alert',
		'button',
		'cell',
		'checkbox',
		'columnheader',
		'combobox',
		'dialog',
		'gridcell',
		'heading',
		'img',
		'link',
		'listbox',
		'listitem',
		'menu',
		'menuitem',
		'menuitemcheckbox',
		'menuitemradio',
		'option',
		'radio',
		'row',
		'rowheader',
		'searchbox',
		'slider',
		'spinbutton',
		'switch',
		'tab',
		'tabpanel',
		'textbox',
		'treeitem',
	}
)

# Attributes whose value is content rather than identity, so a long one is a paragraph
# that will be reworded, not a handle.
_MAX_VALUE_LEN = 120


def _quote(value: str, max_len: int = _MAX_VALUE_LEN) -> str:
	"""Quote a value for embedding in a double-quoted Playwright expression."""
	collapsed = ' '.join(value.split())
	escaped = collapsed.replace('\\', '\\\\').replace('"', '\\"')
	if len(escaped) > max_len:
		escaped = escaped[: max_len - 3] + '...'
	return f'"{escaped}"'


def _css_attr(tag: str, attr: str, value: str) -> str:
	"""A CSS attribute selector using single quotes inside, so it survives being
	wrapped in a double-quoted `locator("...")`."""
	return f"{tag}[{attr}='{value.replace(chr(39), chr(92) + chr(39))}']"


def looks_dynamic(value: str) -> bool:
	"""Does this attribute value look auto-generated, and therefore unsafe to anchor on?

	Conservative by design: a false positive here silently discards the best locator an
	element has, which is exactly how a page ends up with nothing but a broken xpath.
	"""
	if not value:
		return True

	candidate = value.strip()
	if _UUID_RE.search(candidate) or _DYNAMIC_PREFIX_RE.match(candidate):
		return True

	for segment in re.split(r'[\s_\-]+', candidate):
		if len(segment) < 5:
			continue

		# Digits at either end are human ordering conventions — `04fullname`,
		# `address1`, `10address2`. Digits in the *middle* are the hash signature:
		# `3xY7z`, `1d3w5wq`, `a3f9c2b1`. So judge only what is left once both runs
		# are stripped.
		body = segment.lstrip('0123456789')
		if not body:
			return True  # all digits -> a counter
		core = body.rstrip('0123456789')
		if not core:
			return True

		if any(character.isdigit() for character in core):
			return True  # interleaved digits
		if len(core) >= 6 and not any(character in 'aeiouAEIOU' for character in core):
			return True  # consonant soup
		if len(core) >= 8 and re.fullmatch(r'[0-9a-f]+', core, re.IGNORECASE):
			return True  # hex blob

	return False


def scored_candidates(signals: ElementSignals) -> list[LocatorCandidate]:
	"""Every viable locator for the element, best-first.

	Each carries what is needed to count its matches on the page: a CSS
	`verify_selector`, or for the exact `get_by_role` / `get_by_text` family the role
	and text. Only a long name that cannot be matched exactly stays unverifiable.
	"""
	attributes = signals.attributes or {}
	tag = (signals.tag_name or '*').lower()
	text = (signals.text or '').strip()
	if len(text) > 60:
		text = ''
	role = (signals.role or '').strip()
	if role not in _ARIA_ROLES:
		role = ''
	accessible_name = (signals.accessible_name or '').strip()

	candidates: list[LocatorCandidate] = []

	def add(score: int, kind: str, locator: str, verify: str | None = None) -> None:
		candidates.append(LocatorCandidate(locator=locator, kind=kind, score=score, verify_selector=verify))

	def add_exact(score: int, kind: str, role: str | None, value: str) -> None:
		"""A role or text locator with `exact=True`, so it can be counted: a substring
		match would also hit "Uttar Pradesh East". A value too long to quote whole
		cannot be matched exactly, so it is emitted unverifiable rather than truncated."""
		collapsed = ' '.join(value.split())
		quoted = _quote(collapsed)
		if len(collapsed) > _MAX_VALUE_LEN:
			call = f'get_by_role({_quote(role)}, name={quoted})' if role else f'get_by_text({quoted})'
			add(score, kind, call)
			return
		call = f'get_by_role({_quote(role)}, name={quoted}, exact=True)' if role else f'get_by_text({quoted}, exact=True)'
		candidates.append(LocatorCandidate(locator=call, kind=kind, score=score, verify_role=role or None, verify_text=collapsed))

	# Purpose-built test hooks: the most stable thing a page can expose.
	for attr in _TEST_ID_ATTRS:
		value = (attributes.get(attr) or '').strip()
		if not value or looks_dynamic(value):
			continue
		css = _css_attr(tag, attr, value)
		if attr == 'data-testid':
			add(100, 'test-id', f'get_by_test_id({_quote(value)})', css)
		else:
			add(98, 'test-id', f'locator({_quote(css, 400)})', css)

	element_id = (attributes.get('id') or '').strip()
	if element_id and not looks_dynamic(element_id):
		selector = f'#{element_id}' if re.match(r'^[A-Za-z][\w-]*$', element_id) else _css_attr(tag, 'id', element_id)
		add(92, 'id', f'locator({_quote(selector, 400)})', selector)

	name = (attributes.get('name') or '').strip()
	if name and not looks_dynamic(name):
		selector = _css_attr(tag, 'name', name)
		add(84, 'name', f'locator({_quote(selector, 400)})', selector)

	aria_label = (attributes.get('aria-label') or '').strip()
	if aria_label and not looks_dynamic(aria_label):
		# get_by_label also matches <label for=...>, so the CSS is a subset, not an
		# equivalent. Verified separately as an attribute selector.
		add(76, 'aria-label', f'get_by_label({_quote(aria_label)})', _css_attr(tag, 'aria-label', aria_label))

	if role and accessible_name and not looks_dynamic(accessible_name):
		add_exact(72, 'role+name', role, accessible_name)

	placeholder = (attributes.get('placeholder') or '').strip()
	if placeholder and not looks_dynamic(placeholder):
		add(64, 'placeholder', f'get_by_placeholder({_quote(placeholder)})', _css_attr(tag, 'placeholder', placeholder))

	stable_classes = [c for c in (attributes.get('class') or '').split() if c and not looks_dynamic(c)]
	if stable_classes:
		selector = tag + ''.join(f'.{c}' for c in stable_classes[:3])
		score = 50 + min(len(stable_classes), 3) * 3
		if text:
			add(score + 6, 'css+text', f'locator({_quote(selector, 400)}, has_text={_quote(text)})', selector)
		else:
			add(score, 'css', f'locator({_quote(selector, 400)})', selector)

	if text:
		if role and text != accessible_name:
			add_exact(40, 'role+text', role, text)
		elif not role:
			add_exact(38, 'text', None, text)

	# A stable id a few levels up, for a widget whose own element has only generated
	# classes and counter ids (react-select). Ranked below everything the element
	# itself carries. A form field is addressed as the field inside the anchor —
	# filling the wrapper <div> would do nothing — anything else as the anchor,
	# which is what a click on a custom control needs.
	anchor = (signals.anchor_id or '').strip()
	if anchor:
		anchor_css = f'#{anchor}' if re.match(r'^[A-Za-z][\w-]*$', anchor) else _css_attr('*', 'id', anchor)
		if tag in ('input', 'textarea', 'select'):
			selector = f'{anchor_css} {tag}'
			add(30, 'anchor+field', f'locator({_quote(selector, 400)})', selector)
		elif not text and not accessible_name:
			# Only for an anonymous element. One with text of its own (an option inside
			# the dropdown) must be found by that text; the anchor would click the
			# dropdown instead, and nothing would notice.
			add(28, 'anchor', f'locator({_quote(anchor_css, 400)})', anchor_css)

	# Positional xpath: last resort, and only where it can work at all. Inside an open
	# shadow root the path is built by walking parentNode across the shadow boundary,
	# so it addresses a node that does not exist from `document` — it matches zero
	# elements rather than the wrong one, which is at least a loud failure.
	xpath = (signals.xpath or '').strip()
	if xpath and not signals.in_shadow_dom:
		add(10, 'xpath', f'locator({_quote("xpath=" + xpath, 400)})')

	candidates.sort(key=lambda c: c.score, reverse=True)
	return candidates


def apply_frame_chain(locator: str, frame_chain: list[str]) -> str:
	"""Prefix a locator with the `frame_locator(...)` hops needed to reach its frame.

	`page.locator(...)` does not cross an iframe boundary, so an element inside one is
	unreachable without this. Optexity's runner evaluates `page.<command>`, and the
	chain slots in ahead of the locator to give `frame_locator(...).locator(...)`.
	"""
	return ''.join(f'{hop}.' for hop in frame_chain) + locator
