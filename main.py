# main.py

import os
import re
import sys
from typing import Any
from playwright.sync_api import sync_playwright
from agent import create_agent_executor, PLANNER_SYSTEM_PROMPT, create_llm
from browser_tools import PlaywrightBrowserTools
import file_system_tools as fs_tools
import html
import textwrap

PLAN_START = "<<<BEGIN_PLAN_MD>>>"
PLAN_END = "<<<END_PLAN_MD>>>"


def extract_plan_md(text: str) -> str | None:
    """
    Robustly extract content between <<<BEGIN_PLAN_MD>>> and <<<END_PLAN_MD>>>.
    Falls back to heading-based slice if markers missing.
    """
    if not isinstance(text, str):
        text = str(text)

    m = re.search(
        r"<<<BEGIN_PLAN_MD>>>\s*([\s\S]*?)\s*<<<END_PLAN_MD>>>",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # Fallback: start from first markdown H1
    m2 = re.search(r"(?m)^# .+", text)
    return text[m2.start():].strip() if m2 else None


def extract_final_path(text: str) -> str:
    """
    Robustly extract <final_answer>...</final_answer> OR a bare 'specs/*.md' line.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1) Well-formed tag with newlines/whitespace
    m = re.search(
        r"<\s*final_answer\s*>\s*([\s\S]*?)\s*<\s*/\s*final_answer\s*>",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        cand = m.group(1).strip()
        if cand:
            return cand

    # 2) Look for a standalone specs/*.md line at the end of the output
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.endswith(".md") and ("specs/" in line or line.count("/") >= 1):
            return line

    return ""


# --- Sanitizers -----------------------------

TOOL_BLOCK_RE = re.compile(
    r"<tool_call\b[^>]*>.*?</tool_call>", re.DOTALL | re.IGNORECASE
)
OBS_BLOCK_RE = re.compile(
    r"<observation\b[^>]*>.*?</observation>", re.DOTALL | re.IGNORECASE
)
TAG_RE = re.compile(
    r"</?(tool_call|observation|final_answer)[^>]*>", re.IGNORECASE
)


def sanitize_to_markdown(raw: str, url: str | None) -> str:
    """
    Convert noisy LLM transcript (with <tool_call>, <observation>, etc.)
    into a clean, human-readable Markdown test plan.
    Strategy:
      1) Strip tool/observation XML-ish blocks
      2) Try to keep from 'Designing Comprehensive Scenarios' or first '#' heading
      3) If nothing looks like a plan, synthesize a solid default plan
    """
    if not isinstance(raw, str):
        raw = str(raw)

    # 1) strip tool/observation blocks entirely
    s = TOOL_BLOCK_RE.sub("", raw)
    s = OBS_BLOCK_RE.sub("", s)
    # remove dangling tags
    s = TAG_RE.sub("", s)
    s = html.unescape(s).strip()

    # 2) try to capture an obvious plan section
    anchors = [
        "Designing Comprehensive Scenarios",
        "Analyzing User Flows",
        "### 1. Valid Search",
        "# Codeandtheory Search",
        "# Test Plan",
        "# Search Page",
        "# Search",
    ]
    for a in anchors:
        start_idx = s.find(a)
        if start_idx != -1:
            s = s[start_idx:].strip()
            break

    # If it still has no markdown heading, we'll synthesize.
    has_heading = bool(re.search(r"^\s*#{1,6}\s+\S", s, re.MULTILINE))
    if not has_heading:
        title = f"Test Plan: {url}" if url else "Test Plan"
        s = textwrap.dedent(
            f"""
        # {title}

        ## Scope
        Validate key flows on the page:
        - Happy paths for primary user actions
        - Guardrails for empty/invalid inputs
        - Visibility/enablement of critical controls

        ## Assumptions / Pre-conditions
        - User can access {url or 'the target page'}
        - Network stable; no server errors
        - Cookies/consent banner accepted if blocking interactions

        ## Out of Scope
        - Backend relevance/scoring correctness
        - Non-functional concerns unless specified

        ## Accessibility (a11y) Smoke
        - Interactive elements are reachable by keyboard
        - Controls have accessible names/roles
        - Enter/Escape behaviors are sensible

        ## Functional Assertions (Examples)
        - Inputs visible and enabled
        - Action buttons visible and enabled
        - Submitting valid data updates the UI appropriately
        - Submitting invalid/empty data yields informative feedback

        ## Risks & Edge Cases
        - Trimming whitespaces
        - Unicode/emoji handling
        - Very long input
        - Slow network behavior / retries

        ## References
        - Page: {url or 'N/A'}
        """
        ).strip()

    # Final minimal cleanup: drop remaining double blank lines > 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


# --- Helpers ------------------------------------------------------------------


def _extract_url_from_task(task: str) -> str | None:
    m = re.search(r"https?://[^\s'\" )>]+", task)
    return m.group(0) if m else None


def _normalize_write_input(inp: dict, fallback_md: str | None) -> dict:
    # Accept "path" as alias for "file_path"
    if "file_path" not in inp and "path" in inp:
        inp["file_path"] = inp.pop("path")
    # Ensure content
    if "content" not in inp or inp.get("content") in (None, ""):
        inp["content"] = fallback_md or ""
    return inp


def smart_call(tool_name: str, input_obj):
    """
    Fill missing URL for planner_setup_page/browser_navigate calls based on TARGET_URL.
    """
    if tool_name in {"planner_setup_page", "browser_navigate"}:
        if not isinstance(input_obj, dict) or "url" not in input_obj:
            if TARGET_URL:
                return {"url": TARGET_URL}
    return input_obj


# --- Main runner --------------------------------------------------------------


def run_planner_agent(task: str):
    print("Initializing Planner Agent...")

    global TARGET_URL
    TARGET_URL = _extract_url_from_task(task)

    llm = create_llm()
    if llm is None:
        print("Failed to initialize LLM. Exiting.")
        return

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            print("Browser launched.")
        except Exception as e:
            print(f"Error launching Playwright: {e}")
            print(
                "Please ensure Playwright is installed correctly (`playwright install`)"
            )
            return

        try:
            browser_tool_instance = PlaywrightBrowserTools(page)

            def wrap_tool(fn_name: str, fn):
                def _wrapped(*args, **kwargs):
                    # normalize planner_write_markdown calls that pass {"path": ...}
                    if fn_name in {
                        "planner_write_markdown",
                        "wrapped__planner_write_markdown",
                    }:
                        if args and isinstance(args[0], dict):
                            d = dict(args[0])
                            if "file_path" not in d and "path" in d:
                                d["file_path"] = d.pop("path")
                            # ensure content key exists (may be filled by runner auto-save)
                            if "content" not in d:
                                d["content"] = ""
                            args = (d,) + tuple(args[1:])

                    # write-protect until snapshot exists
                    if fn_name in {
                       "planner_write_markdown",
                        "edit_createFile",
                        "generator_write_test",
                        "wrapped__planner_write_markdown",
                    }:
                        if hasattr(browser_tool_instance, "_require_snapshot"):
                            msg = browser_tool_instance._require_snapshot()
                            if msg is not None:
                                return msg

                    return fn(*args, **kwargs)

                _wrapped._smart_call = smart_call
                _wrapped.__name__ = f"wrapped__{fn_name}"
                return _wrapped

            # Wrap browser tools
            wrapped_browser_tools = []
            for t in browser_tool_instance.get_tools():
                t_name = getattr(t, "__name__", "tool")
                wrapped_browser_tools.append(wrap_tool(t_name, t))

            file_tools = fs_tools.get_tools()
            all_tools = wrapped_browser_tools + file_tools

            print(f"Agent initialized with {len(all_tools)} tools.")
            print(
                "[debug] Tools registered:",
                ", ".join(getattr(t, "__name__", "tool") for t in wrapped_browser_tools),
                " + file tools",
            )

            # Prime snapshot so writes are allowed
            if TARGET_URL:
                try:
                    boot1 = browser_tool_instance.planner_setup_page(
                        {"url": TARGET_URL}
                    )
                    print("[boot] planner_setup_page ->", boot1)
                    boot2 = browser_tool_instance.browser_snapshot()
                    print(
                        "[boot] browser_snapshot ->",
                        (boot2[:80] + "...")
                        if isinstance(boot2, str)
                        else boot2,
                    )
                except Exception as e:
                    print("[boot] priming failed (non-fatal):", e)

            agent_executor = create_agent_executor(all_tools, PLANNER_SYSTEM_PROMPT)

            print(f"\n--- Running Planner Agent with Task ---\n{task}\n")
            try:
                result = agent_executor.invoke({"input": task})
                text = result["output"]
                transcript = getattr(agent_executor, "transcript", [])

                # ✅ NEW: get the real RAW LLM output that includes BEGIN_PLAN_MD
                raw_output = getattr(agent_executor, "last_raw_output", text)

                # -------------------------
                # AUTO-SAVE / POST-PROCESS (FORCE WRITE TO final_answer)
                # -------------------------

                # 1) Path: prefer <final_answer> from RAW, else infer
                final_path = extract_final_path(raw_output).strip()
                if not final_path:
                    final_path = infer_save_name(TARGET_URL)
                print(f"[runner] Resolved final_path -> {final_path}")

                # 2) Plan body: prefer markers in RAW; else attempted write body; else sanitize
                plan_md = extract_plan_md(raw_output)
                if not plan_md or not plan_md.strip():
                    for ev in transcript:
                        if isinstance(ev, dict) and ev.get("tool") in {
                            "planner_write_markdown",
                            "wrapped__planner_write_markdown",
                            "edit_createFile",
                            "generator_write_test",
                        }:
                            inp = ev.get("input") or {}
                            if isinstance(inp, dict):
                                body = inp.get("content")
                                if isinstance(body, str) and body.strip():
                                    plan_md = body.strip()
                                    break

                if not plan_md or not plan_md.strip():
                    print("[runner] WARN: No plan extracted; using sanitized fallback.")
                    plan_md = sanitize_to_markdown(raw_output, TARGET_URL)

                # Clean residual tool tags if any
                low = plan_md.lower()
                if (
                    "<tool_call" in low
                    or "<observation" in low
                    or "<final_answer" in low
                ):
                    plan_md = sanitize_to_markdown(raw_output, TARGET_URL)

                # 3) Ensure dir exists
                try:
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                except Exception:
                    pass

                # 4) FORCE WRITE exactly to final_path
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(plan_md)
                print(f"[runner] FORCE-WROTE -> {final_path}")

                # 5) Preview (only final_path)
                try:
                    with open(final_path, "r", encoding="utf-8") as f:
                        preview = f.read(300)
                    print(f"[runner] Preview ({final_path}):\n", preview)
                except Exception as e:
                    print("[runner] Preview read failed:", e)

                print("\n--- Planner Agent Finished ---")
                print(f"Final Output:\n{text}")

            except Exception as e:
                print("\nAn error occurred while running the agent:", e)
                print("The agent loop has been terminated.")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        finally:
            print("Closing browser...")
            browser.close()
            print("Planner agent run complete.")


def infer_save_name(url: str | None) -> str:
    # e.g., https://mystore.com/checkout -> specs/mystore.com-checkout-plan.md
    if not url:
        return "specs/plan.md"
    try:
        from urllib.parse import urlparse

        u = urlparse(url)
        host = (u.netloc or "site").replace(":", "-")
        path = (u.path or "/").strip("/").replace("/", "-") or "home"
        fname = f"{host}-{path}-plan.md"
        return os.path.join("specs", fname)
    except Exception:
        return "specs/plan.md"


if __name__ == "__main__":
    TASK = (
        "I need a comprehensive test plan for 'https://www.codeandtheory.com/contact'. "
        "Cover happy paths, edge cases, validation, a11y, and clear expected outcomes."
    )
    if len(sys.argv) > 1:
        TASK = " ".join(sys.argv[1:])
    run_planner_agent(TASK)
