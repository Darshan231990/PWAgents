# PWAgents — Local Playwright Planner & Test Generator (Powered by Ollama)

PWAgents is a fully local, privacy-preserving alternative to Playwright’s AI-powered test planner and test generator.  
It uses **Ollama local LLMs** (Llama 3.1 / Qwen / GPT-OSS 20B, etc.) and executes browser actions using Playwright tool bindings.

This project recreates the Playwright Agents architecture:

- **Planner Agent** → explores the webpage, inspects DOM, and generates a **comprehensive Markdown test plan**
- **Generator Agent** → reads the plan and generates **Playwright `.spec.ts` tests** using real browser tooling
- **100% local**, no API keys, no cloud calls
- Highly extensible tool set (browser navigation, type, click, snapshot, DOM text, fs write, etc.)

---

## 🚀 Features Completed So Far

### ✅ 1. Local LLM-backed Planner Agent
- Accepts a URL and user task → produces a **detailed Markdown test plan**.
- Uses real browser tools:
  - `planner_setup_page`
  - `browser_snapshot`
  - navigation, click, type, evaluate, press, wait, etc.
- Extracts:
  - Page structure
  - UI inventory
  - Interactive components
- Identifies:
  - Happy paths
  - Negative & validation cases
  - Edge cases
  - Accessibility checks
  - Performance/responsiveness
- Saves output automatically to:
specs/<domain>-<page>-plan.md


### ✅ 2. Local Playwright Test Generator Agent
- Reads the Markdown test plan
- Runs:
- `generator_setup_page`
- All browser actions to simulate steps
- `generator_read_log`
- `generator_write_test`
- Generates clean Playwright tests:
- Scenario names
- Steps as comments
- Real browser tool actions
- Proper `.spec.ts` file naming

### ✅ 3. Fully Local Execution (Ollama)
No OpenAI. No cloud APIs.  
Works with:
- Llama 3.1 8B
- GPT-OSS 20B
- Qwen 2.5 Coder 7B
- Any `ollama pull` model.

### ✅ 4. Tooling Layer (30+ Actions)
#### Browser Tools
- Navigate / click / type / press keys
- Snapshot (DOM-text)
- Hover, drag, file upload
- Verify element visibility, text, values
- Console logs, network logs

#### File Tools
- Create directories/files
- Read/write markdown
- Save tests

#### Generator Tools
- Setup page
- Read action logs
- Write Playwright tests

### ✅ 5. Custom Agent Runtime
- Multi-step reasoning executor
- `<tool_call>` + `<final_answer>` parser
- Observation tracking
- Hard rules:
- Planner must call setup exactly once
- Output must appear between:
  ```
  <<<BEGIN_PLAN_MD>>>
  <<<END_PLAN_MD>>>
  ```

### ✅ 6. Output Structure
Generated files live in:

specs/
codeandtheory-search-plan.md
codeandtheory-contact-plan.md

tests/
<scenario-name>.spec.ts


---

## 📂 Project Structure
```
PWAgents/
│── agent.py # Planner+Generator agent core
│── main.py # Run planner
│── run_generator.py # Run Playwright test generator
│── browser_tools.py # Browser actions
│── file_system_tools.py # File operations
│── dom_utils.py # DOM text/snapshot helpers
│── specs/ # Generated test plans
│── tests/ # Generated Playwright tests
│── requirements.txt
│── .venv/
```

---

## 🧩 How It Works

### 1. Run the Planner Agent
```bash
python3 main.py
Opens the browser

Inspects the page

Gathers UI controls

Generates a detailed plan

Saves to /specs/...md
```

## 2. Run the Test Generator
python3 run_generator.py
Reads plan markdown

Generates .spec.ts tests

Saves inside /tests/
----------------------------------------------
### Example Outputs
✔ Test Plan Example

specs/codeandtheory-search-plan.md includes:
Overview
UI inventory
Happy & negative paths
a11y checks
Test data
Scenario details

✔ Generated Playwright Test Example

tests/search-valid.spec.ts

      test.describe('Search Page', () => {
        test('Valid Search Input', async ({ page }) => {
          // 1. Navigate to search page
          await page.goto('https://www.codeandtheory.com/search');
      
          // 2. Type search query
          await page.type('[data-testid="search-input"]', 'react');
      
          // 3. Submit
          await page.press('[data-testid="search-input"]', 'Enter');
        });
      });

### Vision / Next Steps

- Auto-selector extraction from snapshot
- Deep DOM crawling
- Multi-step flows
- Auto-locators (like PW Agents)

- Vercel UI panel for Planner & Generator
- Hybrid mode: Local LLM + OpenAI toggle
- Plugin support for custom tools

### Why PWAgents Is Special

- This is one of the first fully local, AI-driven testing agents with:
- No cloud dependency
- No API key requirement
- Unlimited usage
- Extensible tool ecosystem
- Compatible with any website
- It replicates Playwright’s Agent but experience—locally.

