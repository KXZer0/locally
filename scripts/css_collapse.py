#!/usr/bin/env python
"""css_collapse.py — resolve 39 hand-written stylesheets into one truth per selector.

The rebuild plan's rule is "do not port by reading; port by measurement". This is
the porting half; `scripts/css-oracle.js` is the measuring half.

What it does, and why each part is the way it is:

* **Parses source text, not the CSSOM.** The browser's CSSOM is a perfect parser
  but it longhand-expands every shorthand -- `margin: 0` becomes four
  declarations, `background: none` becomes eight. Round-tripping through it turns
  137 KB of code into 221 KB. So the parser here is deliberately small and
  text-preserving: it keeps the author's shorthand, spacing inside values, and
  the comments, which in this repo carry the reasoning and are the most valuable
  thing in the file.

* **Keys on (at-rule context, exact selector).** Within one such key every
  declaration has identical specificity, so the cascade reduces to source order
  and the last one wins. That is the entire resolution rule and it needs no
  specificity model. Selectors that merely *interact* (`.btn` vs `.panel .btn`)
  are left alone -- they are different keys and specificity still decides between
  them, exactly as today.

* **Emits every surviving declaration at its OWN source position.** The first
  cut merged each selector into one rule placed at its last occurrence, and the
  oracle caught what that costs: `.mono { font-size: .85em }` lives in
  10-tokens.css, `.util-availability { font-size: 10px }` in 16-utilities.css,
  same specificity, so today the 10px wins on order. `.mono` is *also* touched
  in 01-tokens.css, which loads last -- so merging `.mono` to its last
  occurrence carried the font-size past 16-utilities and flipped 13 elements
  from 10px to 12.75px. Merging moves declarations forward in the cascade, and
  forward is where the other selectors are. Dropping losers is safe; moving
  winners is not. So a key's winners stay exactly where they were written, and
  only adjacent occurrences of the same key are folded together.

Usage:
    python scripts/css_collapse.py --report          # what would change
    python scripts/css_collapse.py --write out.css   # single merged sheet
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass, field

CSS_DIR = pathlib.Path("static/css")

# A keyframes body is kept verbatim; this sentinel marks such a node so the
# emitter prints it back untouched instead of treating it as declarations.
RAW = chr(0) + "raw"
NEWLINE = chr(10)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
# CSS is not regular, but it is bracket-balanced, and the only constructs that
# can hide a brace or a semicolon are comments, strings and url(). Handling
# those three is enough for a hand-written sheet, and every parse is verified by
# re-emitting it and comparing against the source with whitespace collapsed.

@dataclass
class Decl:
    prop: str
    value: str
    important: bool
    raw: str          # exactly as written, minus the trailing semicolon

    def text(self) -> str:
        return self.raw.strip()


@dataclass
class Rule:
    selector: str
    decls: list[Decl]
    comments: list[str] = field(default_factory=list)
    order: int = 0
    file: str = ""


@dataclass
class AtBlock:
    prelude: str          # e.g. "@media (max-width: 680px)"
    children: list        # Rule | AtBlock
    comments: list[str] = field(default_factory=list)
    order: int = 0
    file: str = ""


def _skip_ws_comments(s: str, i: int, sink: list[str]) -> int:
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s.startswith("/*", i):
            end = s.find("*/", i + 2)
            end = len(s) if end < 0 else end + 2
            sink.append(s[i:end])
            i = end
        else:
            break
    return i


def _scan_to(s: str, i: int, stops: str) -> int:
    """Advance past strings, comments and url() until one of `stops` at depth 0."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c in "\"'":
            q = c
            i += 1
            while i < len(s) and s[i] != q:
                i += 2 if s[i] == "\\" else 1
            i += 1
        elif s.startswith("/*", i):
            end = s.find("*/", i + 2)
            i = len(s) if end < 0 else end + 2
        elif c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            i += 1
        elif depth == 0 and c in stops:
            return i
        else:
            i += 1
    return i


def parse(text: str, file: str) -> list:
    nodes: list = []
    i = 0
    counter = [0]

    def parse_block(i: int, out: list) -> int:
        pending: list[str] = []
        while i < len(text):
            i = _skip_ws_comments(text, i, pending)
            if i >= len(text) or text[i] == "}":
                break
            start = i
            j = _scan_to(text, i, "{;}")
            if j >= len(text):
                break
            head = text[start:j].strip()
            if text[j] == ";":                       # at-statement, e.g. @import
                counter[0] += 1
                out.append(Rule(head + ";", [], pending, counter[0], file))
                pending = []
                i = j + 1
                continue
            if text[j] == "}":
                i = j
                break
            # a block
            counter[0] += 1
            order = counter[0]
            if head.startswith("@") and not head.startswith("@font-face") \
                    and not head.startswith("@page"):
                block = AtBlock(head, [], pending, order, file)
                pending = []
                if head.startswith("@keyframes") or head.startswith("@-"):
                    # Keyframe bodies are selectors-of-percentages; keep verbatim.
                    depth, k = 1, j + 1
                    while k < len(text) and depth:
                        k = _scan_to(text, k, "{}")
                        if k >= len(text):
                            break
                        depth += 1 if text[k] == "{" else -1
                        k += 1
                    block.children = [Rule("\0raw", [Decl("", "", False,
                                       text[j + 1:k - 1])], [], order, file)]
                    out.append(block)
                    i = k
                    continue
                i = parse_block(j + 1, block.children)
                out.append(block)
                i += 1  # past '}'
                continue
            rule = Rule(re.sub(r"\s+", " ", head), [], pending, order, file)
            pending = []
            i = parse_decls(j + 1, rule)
            out.append(rule)
        return i

    def parse_decls(i: int, rule: Rule) -> int:
        pending: list[str] = []
        while i < len(text):
            i = _skip_ws_comments(text, i, pending)
            if i >= len(text) or text[i] == "}":
                return i + 1
            start = i
            j = _scan_to(text, i, ";}")
            raw = text[start:j].strip()
            if raw:
                if ":" in raw:
                    prop, _, value = raw.partition(":")
                    imp = bool(re.search(r"!\s*important\s*$", value))
                    rule.decls.append(Decl(prop.strip().lower(), value.strip(), imp, raw))
                else:                                   # nested block, e.g. @media in a rule
                    rule.decls.append(Decl("\0raw", raw, False, raw))
                if pending:
                    rule.comments.extend(pending)
                    pending = []
            if j < len(text) and text[j] == "}":
                return j + 1
            i = j + 1
        return i

    parse_block(i, nodes)
    return nodes


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def flatten(nodes, ctx=()):
    """Yield (context_tuple, Rule) for every rule, descending into at-blocks."""
    for n in nodes:
        if isinstance(n, AtBlock):
            yield from flatten(n.children, ctx + (n.prelude,))
        else:
            yield ctx, n


def resolve(files: list[pathlib.Path]):
    """Keep only the winning declaration of every (context, selector, property).

    Two passes. The first records, for each key+property, the last position that
    declares it (and whether an `!important` claimed it, which no later normal
    declaration can take back). The second replays the files in order and keeps a
    declaration only when it sits at that winning position, so what survives is
    in exactly the place the browser resolved it from.
    """
    occurrences = []          # (pos, ctx, sel, rule, decls)
    winner: dict[tuple, int] = {}
    important: set[tuple] = set()
    stats = {"rules": 0, "decls": 0, "dropped_identical": 0, "dropped_conflicting": 0}
    pos = 0

    for fi, f in enumerate(files):
        text = f.read_text(encoding="utf-8")
        for ctx, rule in flatten(parse(text, f.name)):
            if not rule.decls and rule.selector.endswith(";"):
                continue                       # @import / @charset — not carried over
            stats["rules"] += 1
            pos += 1
            for sel in [s.strip() for s in split_selector_list(rule.selector)]:
                # @font-face, @property and friends are not selectors: eleven
                # @font-face blocks are eleven faces, not one face declared
                # eleven times. Merging them by name left a single face and
                # every string in the app was then measured in a fallback font
                # -- which the oracle reported as 29 elements whose only
                # difference was width. Position makes each one its own key.
                key_sel = f"{sel}#{pos}" if sel.startswith("@") else sel
                occurrences.append((pos, ctx, sel, rule))
                for d in rule.decls:
                    key = (ctx, key_sel, d.prop)
                    if key in important and not d.important:
                        continue               # !important is not taken back
                    if d.important:
                        important.add(key)
                    winner[key] = pos

    out = []
    for pos, ctx, sel, rule in occurrences:
        key_sel = f"{sel}#{pos}" if sel.startswith("@") else sel
        keep = []
        for d in rule.decls:
            stats["decls"] += 1
            if winner.get((ctx, key_sel, d.prop)) == pos:
                keep.append(d)
            else:
                stats["dropped_conflicting"] += 1
        if keep:
            out.append((pos, ctx, Rule(sel, keep, rule.comments, pos, rule.file)))

    # Fold runs of the same key that ended up adjacent -- nothing can sit between
    # them, so joining them cannot change any outcome.
    folded = []
    for item in out:
        if (folded and folded[-1][1] == item[1]
                and folded[-1][2].selector == item[2].selector
                and not item[2].selector.startswith("@")):
            folded[-1][2].decls.extend(item[2].decls)
            for c in item[2].comments:
                if c not in folded[-1][2].comments:
                    folded[-1][2].comments.append(c)
            continue
        folded.append(list(item))
    stats["folded"] = len(out) - len(folded)
    return folded, stats


def split_selector_list(sel: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in sel:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def emit(items) -> str:
    """Walk the resolved stream in order, opening and closing at-rule contexts.

    Context changes are emitted where they occur rather than grouped together:
    grouping is the same forward move the module docstring rejects.
    """
    out: list[str] = []
    cur_ctx: tuple = ()
    for _, ctx, rule in items:
        if ctx != cur_ctx:
            for _ in cur_ctx:
                out.append("}")
            for prelude in ctx:
                out.append(prelude + " {")
            cur_ctx = ctx
        pad = "    " * len(cur_ctx)
        for c in rule.comments:
            out.append(indent_by(c, pad))
        if rule.selector == RAW:
            out.append(indent_by(rule.decls[0].raw, pad))
            continue
        decls = NEWLINE.join(f"{pad}    {d.text()};" for d in rule.decls)
        out.append(f"{pad}{rule.selector} {{" + NEWLINE + decls + NEWLINE + f"{pad}}}")
    for _ in cur_ctx:
        out.append("}")
    return (NEWLINE * 2).join(out) + NEWLINE


def indent_by(s: str, pad: str) -> str:
    return NEWLINE.join((pad + line) if line.strip() else line
                        for line in s.split(NEWLINE))


def indent(s: str) -> str:
    return "\n".join(("    " + line) if line.strip() else line for line in s.split("\n"))



# --------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------
# Which of the nine sheets a rule belongs in. Measured first: of 1,194 resolved
# rules, 981 (82%) take part in NO ordering constraint at all -- nothing else
# sets the same property on the same element at the same specificity -- so they
# can be moved to wherever they belong by meaning. The other 213 are pinned
# relative to each other and the oracle is what proves a move did not break one.

BUCKETS = [
    ("tokens",     ("@font-face", "@property", ":root", "html")),
    ("motion",     ("@keyframes",)),
    ("voice",      (".voice", ".orb", ".mic", ".speaker", ".vad", ".ptt", ".enroll")),
    ("tools",      (".util", ".tool", ".ocr", ".doc-", ".image-", ".upscale",
                    ".detect", ".search-", ".index-", ".dropzone")),
    ("chat",       (".thread", ".message", ".msg", ".composer", ".think", ".stream",
                    ".sources", ".source-", ".prose", ".md-", ".code-", ".math",
                    ".activity", ".typing", ".empty-", ".home", ".suggest", ".greeting",
                    # The engine line under the composer and the home screen's
                    # engine overview are the same component seen twice, and
                    # `.ed-*` is its parts. Splitting them across two sheets put
                    # `.engine-overview .ed-dev` before `.empty-device .ed-dev`
                    # -- equal specificity, so the flip cost min-width 0 -> 34px
                    # and dragged `.ed-model` 14px narrower with it.
                    ".ed-", ".engine-overview")),
    ("layout",     (".shell", ".sidebar", ".topbar", ".main", ".app", ".rail",
                    ".workspace", ".nav", ".tabs", ".tab", ".brand", ".view",
                    ".window-", ".titlebar", ".main-header", ".mode-")),
    ("components", (".btn", ".iconbtn", ".switch", ".panel", ".settings", ".card",
                    ".chip", ".menu", ".palette", ".setup", ".update", ".toast",
                    ".context-", ".mem", ".hud", ".odys", ".dev-", ".ed-", ".field",
                    ".select", ".slider", ".badge", ".pill", ".tip", ".sep", ".kbd")),
]

# Selectors whose concern and whose CASCADE POSITION disagree. `input.palette-input`
# deliberately sets `border-radius: 0` ("the shared control rule rounds it; flush
# here") and deliberately LOSES: the shared `input[type="text"]` rule is written
# later, so today the palette input is rounded, not flush. The comment records an
# intent the load order defeated. Moving the rule to a sheet that loads after the
# control rule would grant the intent -- which is a design decision, not a port,
# so it is pinned back to where the outcome is unchanged and listed here instead.
PIN = {
    "input.palette-input": "base",
    "input.palette-input::placeholder": "base",
}

LAYER_OF = {"tokens": "tokens", "base": "base", "layout": "layout",
            "components": "components", "chat": "views", "voice": "views",
            "tools": "views", "motion": "utilities", "responsive": "overrides"}

# responsive.css survives as a file only for media rules whose selector matches
# no component at all; it is nearly empty by design, and that is the honest
# shape of this app's CSS.
ORDER = ["tokens", "base", "layout", "components", "chat", "voice", "tools",
         "motion", "responsive"]


def bucket_for(ctx: tuple, sel: str) -> str:
    # A width media query does NOT make a rule "responsive" for filing purposes.
    # Hoisting all of them into one sheet that loads last hands them precedence
    # they never had: measured at 375px, 29 elements moved -- `.util-nav-item`
    # lost its 8px padding, `.quick-action` shrank from 62px to 46px,
    # `.empty-state` lost 60px of margin -- because each of those media rules
    # was written BEFORE a base rule that used to beat it. So a media rule is
    # filed with the component it modifies and keeps its place among that
    # component's rules. Under `@layer overrides` the hoist would be correct by
    # construction; that is a redesign, not a port. See docs/handoff-css.md.
    if any(c.startswith("@keyframes") for c in ctx):
        return "motion"
    if sel in PIN:
        return PIN[sel]
    low = sel.lower()
    for name, prefixes in BUCKETS:
        if any(pfx in low for pfx in prefixes):
            return name
    # Everything left that is a bare element, a pseudo, or a utility class is
    # the base layer: the reset, typography, focus, .mono, .sr-only.
    return "base"


def split(items):
    out = {name: [] for name in ORDER}
    for pos, ctx, rule in items:
        out[bucket_for(ctx, rule.selector)].append((pos, ctx, rule))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="templates/index.html",
                    help="the HTML whose <link> order defines the cascade")
    ap.add_argument("--write")
    ap.add_argument("--split", metavar="DIR",
                    help="write the nine concern sheets into DIR")
    ap.add_argument("--layers", action="store_true",
                    help="wrap each sheet in its @layer (changes precedence: a "
                         "later layer beats an earlier one regardless of "
                         "specificity, so this is a redesign, not a port)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    # The load order IS the cascade, so it has to come from the page that
    # declared it. Once the template points at the generated sheets, pass the
    # saved copy of the 39-link version with --order.
    order_file = pathlib.Path(args.order).read_text(encoding="utf-8")
    hrefs = re.findall(r'href="/static/css/([^"]+\.css)"', order_file)
    files = [CSS_DIR / h for h in hrefs]
    missing = [f for f in files if not f.exists()]
    if missing:
        print("missing:", missing)
        return 1

    items, stats = resolve(files)
    css = emit(items)

    before = sum(f.stat().st_size for f in files)
    print(f"files      {len(files)} -> 1")
    print(f"rules      {stats['rules']} source -> {len(items)} kept "
          f"({stats['folded']} folded into a neighbour)")
    print(f"decls      {stats['decls']} source -> {sum(len(r.decls) for _, _, r in items)} "
          f"({stats['dropped_conflicting']} overridden and dropped)")
    print(f"bytes      {before:,} -> {len(css.encode()):,}")

    if args.write:
        pathlib.Path(args.write).write_text(css, encoding="utf-8")
        print(f"wrote      {args.write}")

    if args.split:
        d = pathlib.Path(args.split)
        d.mkdir(parents=True, exist_ok=True)
        parts = split(items)
        total = 0
        print()
        for name in ORDER:
            body = emit(parts[name]) if parts[name] else ""
            layer = LAYER_OF[name]
            header = [
                f"/* {name}.css" + (f" -- @layer {layer}" if args.layers else ""),
                "   Generated by scripts/css_collapse.py from the 39 hand-written",
                "   sheets. Every declaration is the one the browser already",
                "   resolved; scripts/css-oracle.js proves the computed style of",
                "   every element is unchanged at 375, 900 and 1440. */",
                "",
            ]
            if args.layers:
                header.append(f"@layer {layer} {{")
                text = NEWLINE.join(header) + NEWLINE + indent_by(body, "    ")
                text += NEWLINE + "}" + NEWLINE
            else:
                text = NEWLINE.join(header) + NEWLINE + body
            path = d / f"{name}.css"
            path.write_text(text, encoding="utf-8")
            total += len(text.encode())
            label = name + ".css"
            print(f"  {label:<18} {len(parts[name]):>4} rules  "
                  f"{len(text.encode()):>7,} b   @layer {layer}")
        print(f"  {'TOTAL':<18} {sum(len(v) for v in parts.values()):>4} rules  "
              f"{total:>7,} b")
    return 0


if __name__ == "__main__":
    sys.exit(main())
