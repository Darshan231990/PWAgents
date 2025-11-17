import re
import os
from typing import Any, List
from playwright.sync_api import Page, Dialog, sync_playwright
from dom_utils import get_simplified_dom


class PlaywrightBrowserTools:
    """
    A set of tools for controlling a Playwright browser page,
    generating test code, and logging actions.
    """

    def __init__(self, page: Page):
        self.page = page
        self.action_log: List[str] = []
        self.dialog_open = False
        self.dialog_message = ""

        self._ready = False
        self._snapshots = 0
        self._last_url = None

        self.page.on("dialog", self._handle_dialog)

    def _require_ready(self) -> str | None:
        return None if self._ready else "Error: Call planner_setup_page or generator_setup_page first."

    def _require_snapshot(self) -> str | None:
        if not self._ready:
            return "Error: Not ready. Call planner_setup_page/generator_setup_page first."
        if self._snapshots < 1:
            return "Error: Take at least one browser_snapshot before writing files."
        return None

    def _log_action(self, action_description: str, code_snippet: str = ""):
        """Internal method to log an action and its code representation."""
        log_entry = f"Action: {action_description}"
        if code_snippet:
            log_entry += f" | Code: {code_snippet}"
        self.action_log.append(log_entry)

    def _handle_dialog(self, dialog: Dialog):
        """Internal handler for dialogs (alerts, confirms, prompts)."""
        self.dialog_open = True
        self.dialog_message = dialog.message
        # We don't accept or dismiss here; we wait for the agent to call a tool.

    # --- Tool Definitions (plain callables; NO @tool) ---

    def planner_setup_page(self, input: Any) -> str:
        """
        Initializes the planning context and navigates to the starting page.
        Accepts either:
          - a URL string: "https://site.com/page"
          - a JSON object with a URL: {"url": "..."} (other keys are tolerated)
          - nested: {"input": {"url": "..."}}
        """
        self.action_log = []  # clear previous scenario log

        # Unwrap {"input": {...}} if present
        if isinstance(input, dict) and "input" in input and isinstance(input["input"], dict):
            inp = dict(input)
            inp.update(inp.pop("input"))
            input = inp

        # Resolve URL from various shapes the agent may send
        url = None
        if isinstance(input, dict):
            url = (
                input.get("url")
                or input.get("target")
                or input.get("page")
                or input.get("href")
            )
        else:
            url = str(input) if input is not None else None

        if not url or not isinstance(url, str) or not url.strip():
            return (
                "Error: planner_setup_page requires a URL string or a JSON object "
                "with a 'url' field. Example: {\"url\": \"https://example.com\"}"
            )

        url = url.strip()
        try:
            self.page.goto(url, timeout=10000)
            self._log_action(f"Planner setup at {url}", f"page.goto(\"{url}\")")
            self._ready = True
            self._last_url = url
            self._snapshots = 0
            return f"Planner setup complete. Navigated to {url}"
        except Exception as e:
            return f"Error during planner setup for {url}: {e}"

    def generator_setup_page(self, url: str) -> str:
        """
        Navigates to a URL to set up the page for a new test scenario.
        This must be the first browser tool called for any new scenario.
        """
        self.action_log = []  # Clear log for new scenario
        try:
            self.page.goto(url, timeout=10000)
            self._log_action(f"Navigate to {url}", f"page.goto(\"{url}\")")
            self._ready = True
            self._last_url = url
            self._snapshots = 0
            return f"Successfully navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    def browser_navigate(self, url: Any) -> str:
        """
        Navigates the browser to the specified URL.
        Also tolerates dicts like {"url": "..."}.
        """
        # --- require ready so model can't navigate before setup
        if (msg := self._require_ready()) is not None:
            return msg

        if isinstance(url, dict):
            url = url.get("url") or url.get("href") or url.get("target") or url.get("page")
        if not isinstance(url, str) or not url.strip():
            return f"Error navigating to {url}: expected a URL string."
        try:
            self.page.goto(url.strip(), timeout=10000)
            self._last_url = url.strip()     # NEW
            self._log_action(f"Navigate to {url}", f"page.goto(\"{url}\")")
            return f"Successfully navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {e}"


    def browser_navigate_back(self) -> str:
        """Navigates the browser back to the previous page in history."""
        try:
            self.page.go_back(timeout=5000)
            self._log_action("Go back", "page.go_back()")
            return f"Successfully navigated back. New URL: {self.page.url}"
        except Exception as e:
            return f"Error navigating back: {e}"

    def browser_click(self, selector: str) -> str:
        """
        Clicks on an element found by its CSS selector.
        """
        self._log_action(f"Click on '{selector}'", f"page.locator(\"{selector}\").click()")
        try:
            self.page.locator(selector).click(timeout=5000)
            return f"Successfully clicked element with selector: {selector}"
        except Exception as e:
            return f"Error clicking element {selector}: {e}"

    def browser_type(self, selector: str, text: str) -> str:
        """
        Types the specified text into an input field (found by CSS selector).
        """
        self._log_action(
            f"Type '{text}' into '{selector}'",
            f"page.locator(\"{selector}\").fill(\"{text}\")",
        )
        try:
            self.page.locator(selector).fill(text, timeout=5000)
            return f"Successfully typed '{text}' into input {selector}."
        except Exception as e:
            return f"Error typing into input {selector}: {e}"

    def browser_press_key(self, selector: str, key: str) -> str:
        """
        Presses a specific key (like 'Enter', 'ArrowDown', 'Tab') on an element.
        """
        self._log_action(
            f"Press '{key}' on '{selector}'",
            f"page.locator(\"{selector}\").press(\"{key}\")",
        )
        try:
            self.page.locator(selector).press(key, timeout=3000)
            return f"Successfully pressed '{key}' on element {selector}."
        except Exception as e:
            return f"Error pressing key '{key}' on {selector}: {e}"

    def browser_snapshot(self, *args, **kwargs) -> str:
        """
        Gets a simplified DOM snapshot. Accepts and ignores any accidental inputs.
        """
        try:
            snapshot = get_simplified_dom(self.page.content())
            if not snapshot.strip():
             return "Page is blank or contains no interactive elements."
            self._snapshots += 1  
            return snapshot
        except Exception as e:
            return f"Error getting page snapshot: {e}"


    def browser_hover(self, selector: str) -> str:
        """Hovers the mouse over an element found by its CSS selector."""
        self._log_action(
            f"Hover over '{selector}'", f"page.locator(\"{selector}\").hover()"
        )
        try:
            self.page.locator(selector).hover(timeout=3000)
            return f"Successfully hovered over element {selector}."
        except Exception as e:
            return f"Error hovering over element {selector}: {e}"

    def browser_select_option(self, selector: str, value: str) -> str:
        """
        Selects an option from a <select> dropdown by its 'value' attribute.
        """
        self._log_action(
            f"Select option '{value}' in '{selector}'",
            f"page.locator(\"{selector}\").select_option(\"{value}\")",
        )
        try:
            self.page.locator(selector).select_option(value, timeout=3000)
            return f"Successfully selected option '{value}' in dropdown {selector}."
        except Exception as e:
            return f"Error selecting option {value} in {selector}: {e}"

    def browser_file_upload(self, selector: str, file_path: str) -> str:
        """
        Placeholder file upload. Logs the action and returns success (no real file checks).
        """
        self._log_action(
            f"Upload file '{file_path}' to '{selector}'",
            f"page.locator(\"{selector}\").set_input_files(\"{file_path}\")",
        )
        return f"Placeholder: Successfully simulated upload of '{file_path}' to '{selector}'."

    def browser_drag(self, source_selector: str, target_selector: str) -> str:
        """Drags an element (source) to another element (target)."""
        self._log_action(
            f"Drag '{source_selector}' to '{target_selector}'",
            f"page.locator(\"{source_selector}\").drag_to(page.locator(\"{target_selector}\"))",
        )
        try:
            self.page.locator(source_selector).drag_to(
                self.page.locator(target_selector),
                timeout=5000,
            )
            return f"Successfully dragged '{source_selector}' to '{target_selector}'."
        except Exception as e:
            return f"Error dragging element: {e}"

    def browser_evaluate(self, javascript_code: Any) -> str:
        """
        Runs a snippet of JavaScript code in the page context.
        Accepts either a string or a dict like {"expression": "..."}.
        Example: "() => document.title"
        """
        if isinstance(javascript_code, dict):
            javascript_code = (
                javascript_code.get("javascript_code")
                or javascript_code.get("expression")
                or javascript_code.get("code")
                or javascript_code.get("js")
                or javascript_code.get("script")
            )
        if not isinstance(javascript_code, str) or not javascript_code.strip():
            return "Error evaluating JavaScript: expected code string."

        code = javascript_code.strip()
        self._log_action(f"Evaluate JS: '{code}'", f"page.evaluate(\"{code}\")")
        try:
            result = self.page.evaluate(code, timeout=5000)
            return f"JavaScript execution returned: {result}"
        except Exception as e:
            return f"Error evaluating JavaScript: {e}"

    def browser_console_messages(self) -> str:
        """Retrieves and clears console messages (logs, errors, warnings)."""
        # Placeholder; capturing console needs extra wiring.
        self._log_action("Get console messages", "# (Console messages captured)")
        return "Placeholder: Console message retrieval is not implemented in this basic tool."

    def browser_network_requests(self) -> str:
        """Retrieves and clears network requests (XHR, fetch)."""
        # Placeholder; capturing requests needs extra wiring.
        self._log_action("Get network requests", "# (Network requests captured)")
        return "Placeholder: Network request retrieval is not implemented in this basic tool."

    def browser_take_screenshot(self, file_path: str) -> str:
        """Takes a screenshot of the current page and saves it to a file."""
        self._log_action(
            f"Take screenshot: '{file_path}'", f"page.screenshot(path=\"{file_path}\")"
        )
        try:
            self.page.screenshot(path=file_path)
            return f"Screenshot saved to {file_path}"
        except Exception as e:
            return f"Error taking screenshot: {e}"

    # --- Dialog Handling Tools ---

    def browser_handle_dialog(self, action: str) -> str:
        """
        Handles an open dialog (alert, confirm, or prompt).
        'action' must be 'accept' or 'dismiss'.
        """
        if not self.dialog_open:
            return "Error: No dialog is currently open."

        self._log_action(f"Handle dialog: {action}", f"# (Dialog handled: {action})")

        self.dialog_open = False
        message = self.dialog_message
        self.dialog_message = ""

        if action == "accept":
            return f"Successfully accepted dialog with message: '{message}'"
        else:
            return f"Successfully dismissed dialog with message: '{message}'"

    # --- Assertion / Verification Tools (for Generator) ---

    def browser_verify_element_visible(self, selector: str) -> str:
        """
        Verifies that an element (found by CSS selector) is visible.
        This is an ASSERTION.
        """
        self._log_action(
            f"Verify element visible: '{selector}'",
            f"expect(page.locator(\"{selector}\")).to_be_visible()",
        )
        try:
            self.page.locator(selector).wait_for(state="visible", timeout=3000)
            return f"Verification successful: Element '{selector}' is visible."
        except Exception as e:
            error = f"Verification FAILED: Element '{selector}' is NOT visible. {e}"
            self.action_log.append(f"ERROR: {error}")
            return error

    def browser_verify_text_visible(self, text: str) -> str:
        """
        Verifies that a specific text string is visible on the page.
        This is an ASSERTION.
        """
        self._log_action(
            f"Verify text visible: '{text}'",
            f"expect(page.get_by_text(\"{text}\")).to_be_visible()",
        )
        try:
            locator = self.page.get_by_text(text, exact=True)
            locator.wait_for(state="visible", timeout=3000)
            return f"Verification successful: Text '{text}' is visible."
        except Exception as e:
            error = f"Verification FAILED: Text '{text}' is NOT visible. {e}"
            self.action_log.append(f"ERROR: {error}")
            return error

    def browser_verify_value(self, selector: str, expected_value: str) -> str:
        """
        Verifies that an input element (found by CSS selector) has a specific value.
        This is an ASSERTION.
        """
        self._log_action(
            f"Verify value of '{selector}' is '{expected_value}'",
            f"expect(page.locator(\"{selector}\")).to_have_value(\"{expected_value}\")",
        )
        try:
            actual_value = self.page.locator(selector).input_value(timeout=3000)
            if actual_value == expected_value:
                return f"Verification successful: Element '{selector}' has value '{expected_value}'."
            else:
                error = (
                    f"Verification FAILED: Element '{selector}' has value '{actual_value}', "
                    f"not '{expected_value}'."
                )
                self.action_log.append(f"ERROR: {error}")
                return error
        except Exception as e:
            error = f"Verification FAILED: Could not get value for '{selector}'. {e}"
            self.action_log.append(f"ERROR: {error}")
            return error

    def browser_verify_list_visible(self, selector: str) -> str:
        """
        Verifies that a list of elements (e.g., search results) is visible.
        Checks that at least one element matching the selector exists.
        This is an ASSERTION.
        """
        self._log_action(
            f"Verify list visible: '{selector}'",
            f"expect(page.locator(\"{selector}\").first).to_be_visible()",
        )
        try:
            self.page.locator(selector).first.wait_for(state="visible", timeout=3000)
            count = self.page.locator(selector).count()
            return f"Verification successful: List '{selector}' is visible with {count} items."
        except Exception as e:
            error = f"Verification FAILED: List '{selector}' is NOT visible. {e}"
            self.action_log.append(f"ERROR: {error}")
            return error

    def browser_wait_for(self, selector: str, state: str = "visible", timeout: int = 5000) -> str:
        """
        Waits for an element (found by CSS selector) to appear on the page.
        'state' can be 'visible', 'hidden', 'attached', 'detached'.
        """
        self._log_action(
            f"Wait for '{selector}' to be {state}",
            f"page.wait_for_selector(\"{selector}\", state=\"{state}\")",
        )
        try:
            self.page.wait_for_selector(selector, state=state, timeout=timeout)
            return f"Element '{selector}' is now {state}."
        except Exception as e:
            # Clean up the error message for the LLM
            error_lines = str(e).split("\n")
            main_error = [line for line in error_lines if "Error:" in line or "Timeout" in line]
            if not main_error:
                main_error = [error_lines[0]]

            clean_error = (
                f"Error waiting for '{selector}': {' '.join(main_error)}. "
                f"This might be because the element did not enter the '{state}' state within the {timeout}ms timeout."
            )
            self.action_log.append(f"ERROR: {clean_error}")
            return clean_error

    def planner_write_markdown(self, file_path_or_obj: Any, content: str | None = None) -> str:
        """
        Accepts either:
          planner_write_markdown("specs/a.md", "# h1")
        or
          planner_write_markdown({"file_path":"specs/a.md","content":"# h1"})
        or (lenient)
          planner_write_markdown({"path":"specs/a.md","content":"# h1"})
        """
        # Optional gate (if you implemented _require_snapshot):
        if hasattr(self, "_require_snapshot"):
            msg = self._require_snapshot()
            if msg is not None:
                return msg

        try:
            if isinstance(file_path_or_obj, dict):
                file_path = file_path_or_obj.get("file_path") or file_path_or_obj.get("path")
                md = file_path_or_obj.get("content")
            else:
                file_path = file_path_or_obj
                md = content

            if not isinstance(file_path, str) or not file_path.strip():
                return "Error: planner_write_markdown requires a file_path."

            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

            # treat None as empty string to avoid TypeError on len()
            md = "" if md is None else md

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md)

            return f"WROTE {len(md)} chars -> {file_path}"
        except Exception as e:
            return f"Error writing markdown: {e}"


    def fs_read_file(self, file_path: str) -> str:
        """
        Reads a text file and returns its content (or a short error).
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"ERROR reading '{file_path}': {e}"


    # --- Generator-Specific Tools ---

    def generator_read_log(self) -> str:
        """
        Reads the accumulated log of actions and their code snippets for the
        current scenario. This is used by the Generator to build the test.
        """
        if not self.action_log:
            return "Error: No actions have been logged for this scenario."
        return "\n".join(self.action_log)

    def generator_write_test(self, file_path: str, test_title: str, test_plan_path: str, describe_block: str) -> str:
        """
        Writes the final Playwright test file using the log.
        You MUST call `generator_read_log` first.
        """
        if (msg := self._require_snapshot()) is not None:
            return msg
        if not self.action_log:
            return "Error: Cannot write test. The action log is empty."

        # --- Build the test code from the log ---
        setup_code = ""
        steps_code_lines: List[str] = []

        # Sanitize test title into a valid Python function name
        sanitized_title = re.sub(r"\s+", "_", test_title.lower())
        sanitized_title = re.sub(r"[^a-zA-Z0-9_]", "", sanitized_title)
        if not sanitized_title:
            sanitized_title = "test_scenario"

        for entry in self.action_log:
            if "ERROR:" in entry:
                # Add errors as comments, but don't stop generation
                steps_code_lines.append(f"    # {entry}")
                continue

            if " | Code: " not in entry:
                continue

            action_part, code_part = entry.split(" | Code: ", 1)
            action_comment = action_part.replace("Action: ", "").strip()

            # Add step as a comment
            steps_code_lines.append(f"    # {action_comment}")

            # Add code, indented
            steps_code_lines.append(f"    {code_part}")
            steps_code_lines.append("")  # blank line for readability

        # The setup step is the first log entry
        if self.action_log and "page.goto" in self.action_log[0]:
            setup_code = self.action_log[0].split(" | Code: ", 1)[1]

        steps_code = "\n".join(steps_code_lines)

        # Create the full test file content
        test_code = f"""
# Test Plan: {test_plan_path}
# Generated by Playwright AI Agent

from playwright.sync_api import Page, expect
import pytest

@pytest.mark.describe("{describe_block}")
def test_{sanitized_title}(page: Page):
    # 1. Setup (from generator_setup_page)
    {setup_code}

    # --- Test Steps ---
{steps_code}
""".lstrip(
            "\n"
        )

        # --- Save the file ---
        try:
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(test_code)
            return f"Successfully wrote test file to '{file_path}'"
        except Exception as e:
            return f"Error writing file '{file_path}': {e}"

    def get_tools(self) -> List[Any]:
        """Returns a list of all tools in this class (as plain callables)."""
        return [
            self.planner_setup_page,
            self.generator_setup_page,
            self.browser_navigate,
            self.browser_navigate_back,
            self.browser_click,
            self.browser_type,
            self.browser_press_key,
            self.browser_snapshot,
            self.browser_hover,
            self.browser_select_option,
            self.browser_file_upload,
            self.browser_drag,
            self.browser_evaluate,
            self.browser_console_messages,
            self.browser_network_requests,
            self.browser_take_screenshot,
            self.browser_handle_dialog,
            self.browser_verify_element_visible,
            self.browser_verify_text_visible,
            self.browser_verify_value,
            self.browser_verify_list_visible,
            self.browser_wait_for,
            self.generator_read_log,
            self.generator_write_test,
            self.planner_write_markdown,
            self.fs_read_file,
        ]

if __name__ == "__main__":
    # Manual smoke test for tools
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        tools = PlaywrightBrowserTools(page)

        print("Testing browser tools...")
        print(tools.generator_setup_page("http://quotes.toscrape.com/login"))
        print(tools.browser_snapshot())
        print(tools.browser_type("input[name=username]", "admin"))
        print(tools.browser_type("input[name=password]", "admin"))
        print(tools.browser_verify_value("input[name=username]", "admin"))
        print(tools.browser_click("input[type=submit]"))
        print(tools.browser_verify_text_visible("Logout"))

        print("\n--- Action Log ---")
        log = tools.generator_read_log()
        print(log)

        print("\n--- Writing Test File ---")
        print(
            tools.generator_write_test(
                file_path="tests/test_login_generated.py",
                test_title="test_login_success",
                test_plan_path="N/A",
                describe_block="Login Tests",
            )
        )

        browser.close()
