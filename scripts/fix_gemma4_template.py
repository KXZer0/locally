"""Make a Gemma 4 chat template parseable by OpenVINO GenAI's Minja engine.

Gemma 4's template uses Python/C-style adjacent string-literal concatenation:

    raise_exception(
        "part one "
        "part two"
    )

Jinja2 accepts that; Minja does not, and fails with
"Expected closing parenthesis in call args". The fix is to merge each run of
adjacent literals into a single literal — a pure syntax change, so the rendered
text is byte-identical.

Usage: python fix_template.py <model_dir> [--check]
"""
import json
import re
import shutil
import sys
from pathlib import Path

# A double-quoted literal (honouring backslash escapes) followed only by
# whitespace/newlines and then another double-quoted literal.
STRING = r'"(?:[^"\\]|\\.)*"'
ADJACENT = re.compile(r'(' + STRING + r')(\s*\n\s*)(?=' + STRING + r')')


def merge_adjacent(text):
    """Collapse runs of adjacent double-quoted literals into one literal."""
    total = 0
    while True:
        # Join one pair at a time: "a" "b"  ->  "ab"
        new, n = re.subn(
            r'(' + STRING + r')(\s*\n\s*)(' + STRING + r')',
            lambda m: '"' + m.group(1)[1:-1] + m.group(3)[1:-1] + '"',
            text,
        )
        if not n:
            return new, total
        total += n
        text = new


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    model_dir = Path(sys.argv[1])
    check_only = "--check" in sys.argv
    changed_any = False

    tpl = model_dir / "chat_template.jinja"
    if tpl.is_file():
        src = tpl.read_text(encoding="utf-8")
        fixed, n = merge_adjacent(src)
        print(f"chat_template.jinja: {n} adjacent-literal join(s)")
        if n and not check_only:
            backup = tpl.with_suffix(".jinja.orig")
            if not backup.exists():
                shutil.copy2(tpl, backup)
                print(f"  backed up -> {backup.name}")
            tpl.write_text(fixed, encoding="utf-8", newline="\n")
            print("  rewritten")
            changed_any = True

    cfg = model_dir / "tokenizer_config.json"
    if cfg.is_file():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        ct = data.get("chat_template")
        if isinstance(ct, str):
            fixed, n = merge_adjacent(ct)
            print(f"tokenizer_config.json chat_template: {n} join(s)")
            if n and not check_only:
                backup = cfg.with_suffix(".json.orig")
                if not backup.exists():
                    shutil.copy2(cfg, backup)
                data["chat_template"] = fixed
                cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                               encoding="utf-8", newline="\n")
                print("  rewritten")
                changed_any = True
        else:
            print("tokenizer_config.json: no inline chat_template")

    # The template GenAI actually uses is baked into the tokenizer IR's
    # rt_info, not chat_template.jinja — fixing only the .jinja changes
    # nothing. Patch the escaped copy in the XML too.
    #
    # In escaped form a concatenation is  &quot;  &#10; spaces  &quot;
    # A genuinely separate argument would carry a comma BEFORE the newline,
    # so this pattern cannot merge distinct arguments.
    tok = model_dir / "openvino_tokenizer.xml"
    if tok.is_file():
        raw = tok.read_text(encoding="utf-8")
        joiner = re.compile(r'&quot;&#10;\s*&quot;')
        n = len(joiner.findall(raw))
        print(f"openvino_tokenizer.xml rt_info chat_template: {n} join(s)")
        if n and not check_only:
            backup = tok.with_suffix(".xml.orig")
            if not backup.exists():
                shutil.copy2(tok, backup)
                print(f"  backed up -> {backup.name}")
            tok.write_text(joiner.sub("", raw), encoding="utf-8", newline="")
            print("  rewritten")
            changed_any = True

    if not check_only and not changed_any:
        print("nothing to change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
