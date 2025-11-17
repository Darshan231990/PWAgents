import os
import sys
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

from browser_tools import BrowserController
import file_system_tools
from agent import create_agent_executor, GENERATOR_SYSTEM_PROMPT

# --- CONFIGURATION ---
# The test plan file created by the Planner Agent
DEFAULT_TEST_PLAN_FILE = "tests/login_test_plan.md" 
# ---------------------

def run_generator_agent_task(plan_file_path: str):
    """
    Launches the Generator Agent to read a test plan and write a test script.
    """
    # Load API keys from .env file
    load_dotenv()
    
    # 1. Read the test plan
    print(f"[GENERATOR] Reading test plan from: {plan_file_path}")
    plan_content = file_system_tools.read_file.invoke({"file_path": plan_file_path})
    
    if "Error: File not found" in plan_content:
        print(f"\nFATAL ERROR: Test plan not found at '{plan_file_path}'.")
        print("Please run the Planner Agent first (python main.py) to create the plan.")
        sys.exit(1)
        
    print(plan_content)
    
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=False,  # Set to False to watch the agent work!
                slow_mo=750      # Slows down actions to make it observable
            )
            page = browser.new_page()

            # 2. Initialize the browser tool controller
            controller = BrowserController(page)
            
            # 3. Get BOTH toolsets
            browser_tools = controller.get_tools()
            fs_tools = file_system_tools.get_tools()
            all_tools = browser_tools + fs_tools
            
            # 4. Create the agent executor
            agent_executor = create_agent_executor(
                tools=all_tools,
                system_prompt=GENERATOR_SYSTEM_PROMPT # Pass the Generator's brain
            )

            # 5. Run the agent
            print(f"\n[GENERATOR AGENT] Starting test generation...\n")
            
            # The input is the content of the test plan
            task_input = f"""
Here is the test plan I need you to generate:
---
{plan_content}
---
Please generate the test script based on these steps.
"""
            
            result = agent_executor.invoke({
                "input": task_input,
                "chat_history": []
            })
            
            print(f"\n[GENERATOR AGENT] Task finished.")
            print("="*50)
            print("GENERATOR AGENT FINAL MESSAGE:")
            print("="*50)
            
            print(result.get('output', ''))

            print("\nBrowser operations complete. Closing browser.")
            browser.close()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            if 'browser' in locals():
                browser.close()

if __name__ == "__main__":
    
    # You can pass a test plan file as an argument
    # e.g., python run_generator.py "my_other_plan.md"
    plan_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_PLAN_FILE
    
    run_generator_agent_task(plan_file)
