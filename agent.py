# agent.py — minimal, dependency-safe agent (works with ChatOllama, no AgentExecutor)
# Keeps your public API: PLANNER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, create_llm, create_agent_executor

from __future__ import annotations

import html
import inspect
import json
import re
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama

# ----------------------------
#  JSON / tool-call helpers
# ----------------------------

def _safe_jsonl(path: str, obj: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _try_json_loads(val: Any) -> Any:
    """Parse JSON strings into Python objects when possible; otherwise return as-is."""
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return val
    return val


# ---- alias-aware coercion & calling ----

_PARAM_ALIASES: Dict[str, List[str]] = {
    "url": ["url", "href", "target", "page"],
    "selector": ["selector", "css", "locator"],
    "text": ["text", "value", "query"],
    "key": ["key", "keys"],
    "file_path": ["file_path", "path", "dest"],
    "javascript_code": ["javascript_code", "expression", "code", "js", "script"],
    "state": ["state"],
    "timeout": ["timeout", "ms"],
    "expected_value": ["expected_value", "expected", "value"],
    "input": ["input"],  # generic wrapper some models emit
}

def _extract_kwargs_for(fn: Callable[..., Any], tool_input: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Given a function and a model-produced tool_input, map inputs to the fn's signature.
    - Strings/numbers → positional
    - Dicts          → kwarg mapping by name or alias (unwrap {"input": {...}} if present)
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    # no params → call with nothing
    if not params:
        return (), {}

    # primitive → pass positionally
    if isinstance(tool_input, (str, int, float, bool)) or tool_input is None:
        return (tool_input,), {}

    # dict → map by param names / aliases; unwrap {"input": {...}} if nested dict
    if isinstance(tool_input, dict):
        if "input" in tool_input and isinstance(tool_input["input"], dict):
            tool_input = {**tool_input, **tool_input["input"]}

        kwargs: Dict[str, Any] = {}
        remaining = dict(tool_input)

        for p in params:
            pname = p.name
            aliases = _PARAM_ALIASES.get(pname, [pname])
            found = None
            for a in aliases:
                if a in remaining:
                    found = remaining.pop(a)
                    break
            if found is not None:
                kwargs[pname] = found

        # single-arg functions: if nothing matched and dict has one value, pass it positionally
        if len(params) == 1 and not kwargs and len(tool_input) == 1:
            only_val = next(iter(tool_input.values()))
            return (only_val,), {}

        return (), kwargs

    # fallback: pass as single positional
    return (tool_input,), {}

def _normalize_tool_input_for_known_tools(tool_name: str, tool_input: Any) -> Any:
    """
    Make common shapes more forgiving for known tools.
    """
    # unwrap {"input": {...}} early
    if isinstance(tool_input, dict) and "input" in tool_input and isinstance(tool_input["input"], dict):
        tool_input = tool_input["input"]

    # browser_snapshot: tolerate junk input by ignoring it
    if tool_name == "browser_snapshot":
        return {}

    # navigation tools: accept {"url": "..."} or {"href": "..."} or a raw string
    if tool_name in ("browser_navigate", "generator_setup_page", "planner_setup_page"):
        if isinstance(tool_input, dict):
            url = tool_input.get("url") or tool_input.get("href") or tool_input.get("target") or tool_input.get("page")
            if url:
                return {"url": url}
        if isinstance(tool_input, str):
            return {"url": tool_input}

    # browser_evaluate: accept dict forms like {"expression": "..."}
    if tool_name == "browser_evaluate" and isinstance(tool_input, dict):
        expr = (tool_input.get("javascript_code") or tool_input.get("expression")
                or tool_input.get("code") or tool_input.get("js") or tool_input.get("script"))
        return expr if expr is not None else tool_input

    return tool_input

def _call_tool(fn: Callable[..., Any], tool_input: Any) -> Any:
    """
    Ultra-robust tool caller:
    - Unwrap {"input": {...}} or {"args": {...}} if present.
    - If dict: try kwargs first (unfiltered), then pass dict positionally.
    - If non-dict: pass positionally.
    - Finally: try calling with no args.
    """
    # 1) unwrap common wrappers
    if isinstance(tool_input, dict):
        if "input" in tool_input and isinstance(tool_input["input"], (dict, list, str, int, float, bool, type(None))):
            tool_input = tool_input["input"]
        elif "args" in tool_input and isinstance(tool_input["args"], (dict, list, str, int, float, bool, type(None))):
            tool_input = tool_input["args"]

    # 2) dict → kwargs
    if isinstance(tool_input, dict):
        try:
            return fn(**tool_input)
        except TypeError:
            pass
        # 3) dict → positional
        try:
            return fn(tool_input)
        except TypeError:
            pass

    # 4) non-dict → positional
    if not isinstance(tool_input, dict):
        try:
            return fn(tool_input)
        except TypeError:
            pass

    # 5) no-arg
    return fn()

# ----------------------------
#  PROMPTS
# ----------------------------

# Drop this into agent.py, replacing the existing PLANNER_SYSTEM_PROMPT.

PLANNER_SYSTEM_PROMPT = r"""
You are the **Planner Agent**.

Your job: for ANY given real web page, you must create a **deep, page-aware Markdown test plan** that will be consumed by a Playwright **Generator Agent** to create automated tests.

Your output must be:
- **Specific to the actual page** you explore with tools (no generic templates).
- **Rich and detailed**, with clear, atomic steps and expected results.
- **Stable and consistent** in structure, so the Generator can reliably parse and map scenarios to tests.
- **Grounded in what you actually see** on the page snapshot. Do **not** invent UI elements that do not exist.

---
## PHASE 1 – Explore the page (MANDATORY)

You MUST explore the page using tools before you write the plan.

1. Call `planner_setup_page({ "url": "<URL>" })` **once at the start**.
2. Call `browser_snapshot()` at least once to inspect page text and structure.
3. If needed, use other `browser_*` tools (navigate, click, type, press_key, wait_for, evaluate, etc.) to:
   - Open obvious links (e.g. "More Offices", "See All Open Positions", login / signup / cart, etc.).
   - Reveal hidden sections (drawers, modals, accordions).
   - Trigger simple state changes relevant for the plan.

From this exploration you must build a concrete mental model of:
- Visible **sections** (e.g. hero, navigation, content blocks, forms, contact area, footer).
- Concrete **interactive elements**:
  - Inputs / textareas / dropdowns / toggles
  - Buttons / links (internal & external)
  - Icons that act like buttons (e.g. X close icon, hamburger menu)
- Any visible **copy** around them (labels, headings, hints, error text).

### Important grounding rule
- Only describe and test flows that are supported by what you saw via tools.
- For example, on a Code and Theory contact page that shows:
  - Columns such as "Find Us", "Work With Us", "Press", "Careers";
  - Each column containing email links or a link like "See All Open Positions";
  - A newsletter block (e.g. "DECODE") with a single "Sign Up" email field and submit arrow;
  …you should model **exactly those** elements:
  - Contact directory + mailto links;
  - Links to careers / more offices;
  - A single newsletter/signup input (likely email) + submit control;
  - Any footer / social links that actually exist.

No extra fields, no hidden multi-step forms unless you exposed them via tools.

---
## PHASE 2 – Analyse flows & risks

Using what you observed, identify:

- **Primary happy-path journeys** (per page), e.g.
  - Open and close overlays / dialogs.
  - Submit search or signup forms.
  - Navigate via key links and CTAs.
- **Edge / negative cases**:
  - Empty/invalid field submission.
  - Extreme values (long strings, whitespace, special characters).
  - Broken or disabled links/buttons.
- **Validation rules** that can be inferred from the UI (required fields, email format, length restrictions).
- **Accessibility & usability**:
  - Keyboard navigation and focus order.
  - Screen-reader friendliness (labels, headings, roles) where inferable.
  - Escape/close behaviours for dialogs and overlays.
- **Responsiveness & layout** for at least desktop and one mobile viewport.
- **Basic security & resilience** that can be tested from the front-end:
  - Simple XSS-style payloads in text inputs.
  - Basic header-injection attempts in email-like fields.
  - Behaviour on simulated network failure (where relevant to forms).

---
## PHASE 3 – Emit the Markdown test plan

You MUST output the **entire** test plan strictly between the markers:

<<<BEGIN_PLAN_MD>>>
... (complete markdown plan) ...
<<<END_PLAN_MD>>>

Then you MUST finish with a single line containing the save path:

  <final_answer>specs/<auto-name>.md</final_answer>

If you do not know the filename, auto-name as `{host}-{path-slug}-plan.md`, under the `specs/` directory.
Example: `codeandtheory.com-contact-plan.md`.

### REQUIRED PLAN STRUCTURE (ADAPTED PER PAGE)

Always follow this high-level structure. The **headings must exist** in this order, but the internal content must be tailored to the current page.

<<<BEGIN_PLAN_MD>>>
# {SITE_OR_FEATURE_NAME} — Comprehensive Test Plan

## Application Overview
1–3 sentences describing what this specific page does, based on exploration.

## Scope & Objectives
- **In scope:** concrete flows/features you will validate (based on observed elements).
- **Out of scope:** anything not visible or testable from the frontend.

## Assumptions / Preconditions
- Page under test: {URL}
- Cookies/consent banners handled as needed.
- Network is stable enough for basic user journeys.

## Information Architecture (Observed)
- Entry routes / deep links you verified (e.g. `/contact`, `/login`, `/search?q=...`).
- Major sections or content blocks.

## UI Inventory (From Snapshot & Probes)
List only **real** controls you observed. For example, for a contact overlay page this might include:
- Contact directory sections with headings and links.
- Specific email links, tel links, and navigation links.
- A newsletter/signup field + submit control.
- Any footer / legal / social links.

Adapt this list to the **actual** page you are testing.

## Test Data
Enumerate **realistic test data** relevant to the page (especially for forms):
- **Valid values** (e.g., valid email addresses).
- **Invalid values** (missing pieces, clearly malformed input).
- **Edge cases** (very long strings, leading/trailing spaces, special characters).

## Test Scenarios

Write concrete scenarios with IDs. Use this format:

### C01 – Short title
**Type:** happy/edge/negative/a11y/security/performance  
**Area:** e.g. overlay, newsletter, navigation, login_form, etc.  
**Priority:** High/Medium/Low  
**Seed:** `tests/seed.spec.ts` (include when applicable)

**Preconditions:**
- Bullet list of any important preconditions (or `- None`).

**Steps:**
1. Step written in imperative form.
2. One user action or assertion per step.

**Expected Results:**
- Bullet list of clear, observable outcomes that a Playwright test can assert.

You MUST cover at least the following **categories**, when they make sense for the page:
- Overlay / page entry & exit (open/close behaviour, deep links, back navigation).
- Core happy paths for the main purpose of the page.
- Validation & error handling (empty, invalid, and edge-case inputs).
- Accessibility (keyboard, focus, basic screen-reader semantics).
- Responsive behaviour for at least one mobile viewport.
- Basic security / resilience checks for any inputs or actions.

Prefer **fewer, high-value scenarios** over many shallow ones. Every scenario should be testable and valuable to automation.

## Accessibility (WCAG 2.1 AA) Notes
Summarise key a11y considerations and which scenarios cover them (e.g. "Keyboard-only navigation covered by C07").

## Risks & Open Questions
Call out anything you could not verify from the frontend or from the current environment.

## Test Environment & Execution Notes
List browsers, devices, and tools assumed for execution.

## Success Criteria
Define what it means for this page to "pass" (e.g. all High-priority scenarios pass in target browsers and viewports, no blocking a11y issues, etc.).
<<<END_PLAN_MD>>>

---
## Guardrails & Style

- Be **page-specific**: reference the actual headings, labels, and link texts you saw.
- Do **not** hallucinate forms or flows that were not observed.
- Keep steps **atomic** so that a Playwright test can map them one-to-one to actions.
- Always include both happy-path and negative/edge coverage when inputs exist.
- Keep language concise but precise; this is documentation for engineers.
"""


GENERATOR_SYSTEM_PROMPT = """
You are a Playwright Test Generator, an expert in browser automation and end-to-end testing.
Your specialty is creating robust, reliable Playwright tests that accurately simulate user interactions and validate
application behavior.

# For each test you generate
- Obtain the test plan with all the steps and verification specification
- Run the `generator_setup_page` tool to set up page for the scenario
- For each step and verification in the scenario, do a-label='Write steps that are'
  - Use Playwright tool to manually execute it in real-time.
  - Use the step description as the intent for each PlayLoop'
- Retrieve generator log via `generator_read_log`
- Immediately after reading the test log, invoke `generator_write_test` with the generated source code
  - File should contain single test
  - File name must be fs-friendly scenario name
  - Test must be placed in a describe matching the top-level test plan item
  - Test title must match the scenario name
  - Includes a comment with the step text before each step execution. Do not duplicate comments if step requires
    multiple actions.
  - Always use best practices from the log when generating tests.

   <example-generation>
   For following plan:

   ```markdown file=specs/plan.md
   ### 1. Adding New Todos
   **Seed:** `tests/seed.spec.ts`

   #### 1.1 Add Valid Todo
   **Steps:**
   1. Click in the "What needs to be done?" input field

   #### 1.2 Add Multiple Todos
   ...
   ```

   Following file is generated:

   ```ts file=add-valid-todo.spec.ts
   // spec: specs/plan.md
   // seed: tests/seed.spec.ts

   test.describe('Adding New Todos', () => {
     test('Add Valid Todo', async { page } => {
       // 1. Click in the "What needs to be done?" input field
       await page.click(...);

       ...
     });
   });
   ```
   </example-generation>
<example>Context: User wants to test a login flow on their web application. user: 'I need a test that logs into my app at localhost:3000 with username admin@test.com and password 123456, then verifies the dashboard page loads' assistant: 'I'll use the generator agent to create and validate this login test for you'Services/A/G/P/S/G/C' <commentary> The user needs a specific browser automation test created, which is exactly what the generator agent is designed for. </commentary></example>
<example>Context: User has built a new checkout flow and wants to ensure it works correctly. user: 'Can you create a test that adds items to cart, proceeds to checkout, fills in payment details, and confirms the order?' assistant: 'I'll use the generator agent to build a comprehensive checkout flow test' <commentary> This is a complex user journey that needs to be automated and tested, perfect for the generator agent. </commentary></G/P/S/G/C'</example>
"""


# ----------------------------
#  Tool-calling protocol text
# ----------------------------

INSTRUCTIONS = """
You can either:
1) Call a tool by emitting:

<tool_call>
{"tool": "<tool_name>", "input": {"key": "value", ...}}
</tool_call>

…then wait for an <observation> before continuing.

2) Or finish with a final answer:

<final_answer>
...your final response here...
</final_answer>

Hard rules:
- Use ONLY tools listed in TOOLS. Do NOT invent tools like 'extract_links' or 'json_parser'.
- Emit exactly one <tool_call> per step.
- When emitting JSON, it MUST be valid JSON (double quotes, escaped inner quotes).
- Prefer an OBJECT for "input" (e.g. {"url": "..."}), do not pass plain strings.
- For planning: you MUST invoke `planner_setup_page` exactly once at the beginning.
- For output: you MUST save the final Markdown test plan to disk using `edit_createFile`
  at path "specs/search-plan.md" (create directories as needed), then end with <final_answer>
  summarizing where the file was written.

Notes:
- To explore the page, use: browser_snapshot (no args), browser_navigate, browser_click, browser_type, etc.
- Do NOT call tools that are not in the provided list.
"""


# ----------------------------
#  Prompt builder & parser
# ----------------------------

def _normalize_tool(t: Any) -> Tuple[str, Callable[..., Any], str]:
    """
    Return a stable (name, call_fn, desc) for any callable or Tool-like object.
    Handles bound methods, functions, callables, and LangChain-style tools.
    """
    desc = getattr(t, "description", "") or getattr(t, "desc", "")
    name = getattr(t, "name", None) or getattr(t, "tool_name", None)

    if name is None:
        if isinstance(t, types.MethodType):
            name = t.__name__                         # bound method name
        else:
            name = getattr(t, "__name__", None) or getattr(t, "__qualname__", None)
            if not name:
                name = t.__class__.__name__

    # Preferred entry points for LangChain tools: .invoke / .run / __call__ / .func
    for attr in ("invoke", "run", "__call__", "func"):
        fn = getattr(t, attr, None)
        if callable(fn):
            return name, fn, desc

    if callable(t):
        return name, t, desc

    raise ValueError(f"Unsupported tool object: {t!r}")


def _render_tool_list(tools: List[Any]) -> str:
    items = []
    for t in tools:
        n, _, d = _normalize_tool(t)
        items.append(f"- {n}: {d or 'No description'}")
    return "\n".join(items)


def _build_prompt(system_prompt: str, tools: List[Any], transcript: List[Dict[str, str]], user_input: str) -> str:
    tool_list = _render_tool_list(tools)

    # Harden against any non-dict residues in transcript
    convo_lines: List[str] = []
    for turn in transcript:
        try:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                role = str(turn.get("role", "ASSISTANT")).upper()
                content = str(turn.get("content", ""))
                convo_lines.append(f"{role}: {content}")
            else:
                convo_lines.append(f"ASSISTANT: {str(turn)}")
        except Exception as e:
            convo_lines.append(f"ASSISTANT: [transcript format error: {e}] {repr(turn)}")

    convo_text = "\n".join(convo_lines)

    return f"""SYSTEM:
{system_prompt.strip()}

TOOLS (you can call these):
{tool_list}

INSTRUCTIONS:
{INSTRUCTIONS.strip()}

CONVERSATION SO FAR:
{convo_text}

USER:
{user_input}

ASSISTANT:
Remember: either output a single <tool_call>...</tool_call> block OR a <final_answer>...</final_answer> block.
"""


def _parse_response(text: str) -> Dict[str, Any]:
    """Detect <tool_call> or <final_answer>."""
    text = text.strip()

    # standard JSON block
    if "<tool_call>" in text and "</tool_call>" in text:
        block = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        try:
            payload = json.loads(block)
            return {"type": "tool_call", "tool": payload.get("tool"), "input": payload.get("input")}
        except Exception as e:
            return {"type": "final", "output": f"[parser error: {e}] RAW: {text}"}

    # permissive: <tool_call name="X" input_string="..."/>
    if "<tool_call" in text and "/>" in text:
        # very light extraction; if it fails, treat as final text
        try:
            name_match = re.search(r'<tool_call[^>]*name="([^"]+)"', text)
            input_match = re.search(r'<tool_call[^>]*input_string="([^"]+)"', text)
            if name_match:
                tool = name_match.group(1)
                raw = html.unescape(input_match.group(1)) if input_match else ""
                try:
                    inp = json.loads(raw)
                except Exception:
                    inp = raw
                return {"type": "tool_call", "tool": tool, "input": inp}
        except Exception:
            pass

    if "<final_answer>" in text and "</final_answer>" in text:
        block = text.split("<final_answer>", 1)[1].split("</final_answer>", 1)[0].strip()
        return {"type": "final", "output": block}

    return {"type": "final", "output": text}


# ----------------------------
#  Executor (minimal loop)
# ----------------------------

class _Executor:
    def __init__(self, llm: ChatOllama, tools: List[Any], system_prompt: str, max_steps: int = 8, verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.transcript: List[Dict[str, str]] = []
        self.max_steps = max_steps
        self.verbose = verbose
        self.last_raw_output: str = ""

        # Build tool index once (with duplicate-name handling)
        self._tool_index: Dict[str, Tuple[str, Callable[..., Any], str]] = {}
        for t in tools:
            name, fn, desc = _normalize_tool(t)
            base = name
            i = 2
            while name in self._tool_index:
                name = f"{base}_{i}"
                i += 1
            self._tool_index[name] = (name, fn, desc)

        if self.verbose:
            print("[debug] Tools registered:", ", ".join(sorted(self._tool_index.keys())))

    def _observe(self, text: Any):
        self.transcript.append({"role": "observation", "content": str(text)})
        if self.verbose:
            print(str(text))

    def _assistant(self, text: Any):
        self.transcript.append({"role": "assistant", "content": str(text)})
        if self.verbose:
            print(str(text))

    def _user(self, text: Any):
        self.transcript.append({"role": "user", "content": str(text)})
        if self.verbose:
            print(str(text))

    def _one_turn(self, user_input: str) -> Optional[str]:
        """Runs one reasoning + tool step."""
        try:
            prompt = _build_prompt(self.system_prompt, self.tools, self.transcript, user_input)
            result = self.llm.invoke(prompt)
            text = getattr(result, "content", None) or str(result)

            self.last_raw_output = text

            if self.verbose:
                print("\n----- RAW LLM OUTPUT -----")
                print(text)
                print("--------------------------\n")

            parsed = _parse_response(text)

            # FINAL ANSWER
            if parsed["type"] == "final":
                final_text = parsed.get("output", "")
                self._assistant(final_text)
                return final_text

            # TOOL CALL
            if parsed["type"] == "tool_call":
                tool_name = parsed.get("tool")
                tool_input = parsed.get("input")

                if not tool_name or tool_name not in self._tool_index:
                    msg = f"[tool error] Unknown tool '{tool_name}'. Available: {', '.join(self._tool_index.keys())}"
                    self._observe(msg)
                    return None

                _, fn, _ = self._tool_index[tool_name]
                try:
                    # First normalize the input for known tools
                    tool_input = _normalize_tool_input_for_known_tools(tool_name, tool_input)
                    # Then apply smart_call if available from main
                    if hasattr(fn, '_smart_call'):
                        tool_input = fn._smart_call(tool_name, tool_input)
                    observation = _call_tool(fn, tool_input)
                except Exception as e:
                    observation = f"[tool exception] {e}"

                self._observe(f"<observation tool='{tool_name}'>\n{str(observation)}\n</observation>")
                return None  # continue agent loop

            # FALLBACK — unexpected agent output
            self._assistant(text)
            return text

        except Exception as e:
            err = f"[agent error] {e}"
            self._observe(err)
            return err

    def invoke(self, inputs: Dict[str, Any]):
        text = inputs.get("input", "")
        out = self.run(text)
        return {"output": out}

    def run(self, text: str) -> str:
        """Run a short tool-use loop until <final_answer> or max_steps."""
        self._user(text)
        last_final = ""
        for _ in range(self.max_steps):
            out = self._one_turn(user_input="")
            if out is not None:
                last_final = out
                break
        return last_final or "[no final answer produced within max steps]"


# ----------------------------
#  Public factories
# ----------------------------
def create_llm(model: str = "llama3.1:8b-instruct-q5_K_M", temperature: float = 0.2) -> Optional[ChatOllama]:

    try:
        llm = ChatOllama(model=model, temperature=temperature)
        llm.invoke("hello")  # health check
        return llm
    except Exception as e:
        print(f"[error] Ollama initialization failed: {e}")
        print("Ensure Ollama is running and the model is pulled, e.g.:")
        print(f"  ollama pull {model}")
        return None


def create_agent_executor(tools: List[Any], system_prompt: str):
    llm = create_llm()
    if llm is None:
        sys.exit("LLM initialization failed. Cannot create agent.")
    # verbose=True so you can see tools registered + raw model output
    return _Executor(llm=llm, tools=tools, system_prompt=system_prompt, verbose=True)