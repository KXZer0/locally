"""How much of the NPU's 8192-token prompt does a tool set actually cost?

The NPU exclusion in _tools_supported is justified on two grounds: a hard
prompt cap, and small models not planning well. The first is arithmetic and
nobody had done it. This does it, on the real tokenizer, for tool sets of the
size an assistant like Odysseus actually sends.
"""
import json
import sys

sys.path.insert(0, r"C:\Projects\Nollama")
import locally
from locally import render_tools_prompt, NPU_MAX_PROMPT_LEN
import openvino_genai as ovg

MODEL = r"C:\Projects\Nollama\model"
tok = ovg.Tokenizer(MODEL)


def count(text):
    return int(len(tok.encode(text).input_ids.data[0])) if text else 0


def tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}


def s(d):
    return {"type": "string", "description": d}


# A realistic assistant tool set: what a Jarvis actually needs to act, not to code.
JARVIS = [
    tool("get_calendar", "List events in a date range from the user's calendar.",
         {"start": s("ISO date, inclusive."), "end": s("ISO date, inclusive.")},
         ["start", "end"]),
    tool("create_event", "Add an event to the user's calendar.",
         {"title": s("Event title."), "start": s("ISO datetime."),
          "end": s("ISO datetime."), "location": s("Optional location.")},
         ["title", "start"]),
    tool("create_task", "Add a task or reminder to the user's task list.",
         {"title": s("What to do."), "due": s("Optional ISO datetime."),
          "priority": {"type": "string", "enum": ["low", "normal", "high"],
                       "description": "Task priority."}},
         ["title"]),
    tool("list_tasks", "List the user's open tasks, optionally filtered.",
         {"due_before": s("Only tasks due before this ISO datetime.")}, []),
    tool("search_notes", "Search the user's notes and documents semantically.",
         {"query": s("What to look for."),
          "limit": {"type": "integer", "description": "Max results, default 5."}},
         ["query"]),
    tool("web_search", "Search the web and return ranked results.",
         {"query": s("The search query.")}, ["query"]),
    tool("fetch_url", "Fetch a web page and return it as clean Markdown.",
         {"url": s("Absolute http(s) URL.")}, ["url"]),
    tool("send_notification", "Send the user a push notification.",
         {"title": s("Short title."), "body": s("Notification body.")},
         ["title", "body"]),
]

# What an unfiltered coding agent sends, for contrast. Deliberately verbose
# descriptions, because that is what real agent schemas look like.
CODING = JARVIS + [
    tool(f"tool_{i}",
         "A code-oriented tool with the kind of long description real agents "
         "ship: it explains when to use the tool, when not to, what the "
         "arguments mean, what the return shape is, and gives two examples "
         "so the model does not have to guess at the calling convention.",
         {"path": s("Absolute path to the file to operate on."),
          "pattern": s("A regular expression, in ripgrep dialect."),
          "content": s("Full replacement contents for the file."),
          "recursive": {"type": "boolean", "description": "Recurse into subdirs."}},
         ["path"])
    for i in range(22)
]

print(f"NPU prompt cap: {NPU_MAX_PROMPT_LEN} tokens\n")

DEFAULT_SYS = (
    "You are a helpful local assistant running on the user's own hardware. "
    "Answer accurately and concisely. Read text exactly as printed. Do not "
    "invent APIs, citations, or file paths. Do not add disclaimers to ordinary "
    "questions. Do not restate the question before answering. Reply in the "
    "user's language."
)

rows = []
for label, tools in (("Jarvis set (8 tools)", JARVIS),
                     ("Jarvis, trimmed to 5", JARVIS[:5]),
                     ("Jarvis, trimmed to 3", JARVIS[:3]),
                     ("Coding agent (30 tools)", CODING)):
    block = render_tools_prompt(tools)
    n = count(block)
    pct = 100.0 * n / NPU_MAX_PROMPT_LEN
    rows.append((label, len(tools), len(block), n, pct))

w = max(len(r[0]) for r in rows)
print(f"{'tool set'.ljust(w)}  {'tools':>5}  {'chars':>7}  {'tokens':>7}  {'% of 8192':>9}")
print("-" * (w + 36))
for label, ntools, chars, n, pct in rows:
    print(f"{label.ljust(w)}  {ntools:5}  {chars:7}  {n:7}  {pct:8.1f}%")

sys_n = count(DEFAULT_SYS)
print(f"\nFor scale:")
print(f"  default system prompt          {sys_n:5} tokens  ({100.0*sys_n/NPU_MAX_PROMPT_LEN:.1f}%)")

jarvis_n = count(render_tools_prompt(JARVIS))
budget_left = NPU_MAX_PROMPT_LEN - jarvis_n - sys_n
print(f"  system + Jarvis tools          {sys_n + jarvis_n:5} tokens  "
      f"({100.0*(sys_n+jarvis_n)/NPU_MAX_PROMPT_LEN:.1f}%)")
print(f"  left for conversation + result {budget_left:5} tokens")
print(f"\n  ~= {budget_left // 750} pages of conversation, or roughly "
      f"{budget_left // 120} short turns.")

# Per-tool cost, so a budget can be enforced by dropping the priciest first.
print("\nPer-tool cost (rendered alone, minus the fixed preamble):")
preamble = count(render_tools_prompt([JARVIS[0]])) - count(json.dumps(JARVIS[0]["function"], ensure_ascii=False))
for t in JARVIS:
    n = count(json.dumps(t["function"], ensure_ascii=False))
    print(f"  {t['function']['name']:<20} {n:4} tokens")
print(f"  {'(fixed preamble)':<20} {preamble:4} tokens")
