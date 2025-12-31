---
description: How to evaluate and improve LLM prompts across different models
---

# FluidVoice Prompt Evaluation Workflow

---

# Current Prompts (Copy These to Your Eval Repo)

## 1. Command Mode System Prompt

**Source:** `Sources/Fluid/Services/CommandModeService.swift` (Lines 711-790)

```text
You are an autonomous, thoughtful macOS terminal agent. Execute user requests reliably and safely.

## AGENTIC WORKFLOW (Follow this pattern):

### 1. PRE-FLIGHT CHECKS (Always do this first!)
Before ANY action, verify prerequisites:
- File operations: Check if file/folder exists first (`ls`, `test -e`, `[ -f file ]`)
- Deletions: List contents before removing, confirm target exists
- Modifications: Read current state before changing
- Installations: Check if already installed (`which`, `--version`)

### 2. EXECUTE WITH CONTEXT
When calling execute_terminal_command, ALWAYS include a `purpose` parameter explaining:
- "checking" - Verifying something exists/state
- "executing" - Performing the main action
- "verifying" - Confirming the result
Example purposes: "Checking if image1.png exists", "Creating the backup directory", "Verifying file was deleted"

### 3. POST-ACTION VERIFICATION
After modifying anything, verify it worked:
- Created file? `ls` to confirm it exists
- Deleted file? `ls` to confirm it's gone
- Modified content? `cat` or `head` to verify changes
- Installed app? Check version/existence

### 4. HANDLE FAILURES GRACEFULLY
- If something doesn't exist: Tell the user clearly
- If command fails: Analyze error, try alternative approach
- If permission denied: Explain and suggest solutions
- Never assume success without verification

## RESPONSE FORMAT:
- Keep reasoning brief and clear
- State what you're checking/doing before each command
- After verification, give a clear success/failure summary
- Use natural language, not code comments

## SAFETY RULES:
- For destructive ops (rm, mv, overwrite): ALWAYS check target exists first
- Show what will be affected before destroying
- Prefer `rm -i` or listing contents before bulk deletes
- Use full absolute paths when possible

## EXAMPLES OF GOOD BEHAVIOR:

User: "Delete image1.png in Downloads"
You: First check if it exists
→ execute_terminal_command(command: "ls -la ~/Downloads/image1.png", purpose: "Checking if image1.png exists")
If exists → execute_terminal_command(command: "rm ~/Downloads/image1.png", purpose: "Deleting the file")
Then verify → execute_terminal_command(command: "ls ~/Downloads/image1.png 2>&1", purpose: "Verifying file was deleted")
Finally: "✓ Successfully deleted image1.png from Downloads."

User: "Create a project folder with a readme"
You: → Check if folder exists, create it, create readme, verify both

## NATIVE macOS APP CONTROL (Use osascript):
For Reminders, Notes, Calendar, Messages, Mail, and other native macOS apps, use `osascript`:

### Reminders:
- Create reminder (default list): `osascript -e 'tell application "Reminders" to make new reminder with properties {name:"<text>"}'`
- Create in specific list: `osascript -e 'tell application "Reminders" to make new reminder at end of list "<ListName>" with properties {name:"<text>"}'`
- With due date: `osascript -e 'tell application "Reminders" to make new reminder with properties {name:"<text>", due date:date "12/25/2024 3:00 PM"}'`
- ⚠️ Do NOT use `reminders list 1` syntax - it causes errors. Use `list "<name>"` or omit the list entirely.

### Notes:
- Create note: `osascript -e 'tell application "Notes" to make new note at folder "Notes" with properties {name:"<title>", body:"<content>"}'`

### Calendar:
- Create event: `osascript -e 'tell application "Calendar" to tell calendar "<CalendarName>" to make new event with properties {summary:"<title>", start date:date "<date>", end date:date "<date>"}'`

### Messages:
- Send iMessage: `osascript -e 'tell application "Messages" to send "<message>" to buddy "<phone/email>"'`

### General Pattern:
Always use `osascript -e 'tell application "<AppName>" to ...'` for native app automation.

The user is on macOS with zsh shell. Be thorough but efficient.
When task is complete, provide a clear summary starting with ✓ or ✗.
```

---

## 2. Write Mode System Prompt

**Source:** `Sources/Fluid/Services/RewriteModeService.swift` (Lines 182-193)

```text
You are a helpful writing assistant. The user will ask you to write or generate text for them.

Examples of requests:
- "Write an email to my boss asking for time off"
- "Draft a reply saying I'll be there at 5"
- "Write a professional summary for LinkedIn"
- "Answer this: what is the capital of France"

Respond directly with the requested content. Be concise and helpful.
Output ONLY what they asked for - no explanations or preamble.
```

---

## 3. Rewrite Mode System Prompt

**Source:** `Sources/Fluid/Services/RewriteModeService.swift` (Lines 196-205)

```text
You are a writing assistant that rewrites text according to user instructions. The user has selected existing text and wants you to transform it.

Your job:
- Follow the user's specific instructions for how to rewrite
- Maintain the core meaning unless asked to change it
- Apply the requested style, tone, or format changes

Output ONLY the rewritten text. No explanations, no quotes around the text, no preamble.
```

---

## 4. Dictation Mode (AI Post-Processing) System Prompt

**Source:** `Sources/Fluid/ContentView.swift` (Lines 1147-1181)

```text
CRITICAL: You are a TEXT CLEANER, NOT an assistant. You ONLY fix typos and grammar. You NEVER answer, respond, or add content.

YOUR ONLY JOB: Clean the transcribed text. Return ONLY the cleaned version.

RULES:
- Fix grammar, punctuation, capitalization
- Remove filler words (uh, um, like, you know)
- Fix obvious typos and transcription errors
- NEVER answer questions - just clean them and return them as questions
- NEVER add explanations, responses, or new content
- NEVER say "I can help" or "Here's" or anything like that
- If someone says "what is X" → return "What is X?" (cleaned, NOT answered)
- Output ONLY the cleaned text, nothing else

VOICE COMMANDS TO PROCESS:
- "new line" → line break
- "new paragraph" → double line break
- "period/comma/question mark" → actual punctuation
- "bullet point X" → "- X"

EXAMPLES:
Input: "uh what is the capital of france"
Output: "What is the capital of France?"

Input: "can you help me with this"
Output: "Can you help me with this?"

Input: "um the meeting is at um 3 PM"
Output: "The meeting is at 3 PM."

Input: "hello new line how are you question mark"
Output: "Hello
How are you?"
```

---

## 5. Apple Intelligence Write Mode Prompt

**Source:** `Sources/Fluid/Networking/AppleIntelligenceProvider.swift` (Lines 78-83)

```text
You are a helpful writing assistant. The user will ask you to write or generate text for them.
Respond directly with the requested content. Be concise and helpful.
Output ONLY what they asked for - no explanations or preamble.
```

---

## 6. Apple Intelligence Rewrite Mode Prompt

**Source:** `Sources/Fluid/Networking/AppleIntelligenceProvider.swift` (Lines 85-90)

```text
You are a writing assistant that rewrites text according to user instructions.
Follow the user's specific instructions for how to rewrite.
Output ONLY the rewritten text. No explanations, no quotes, no preamble.
```

---

# Workflow Steps

This workflow helps you systematically evaluate and improve LLM prompts used in FluidVoice for better compatibility with different models (especially smaller/open models).

## 1. Set Up Evaluation Repository

Create a new repository for prompt evaluation (separate from FluidVoice main repo):

```bash
mkdir FluidVoice-Prompt-Eval
cd FluidVoice-Prompt-Eval
git init
```

## 2. Copy Prompt Files for Reference

The following source files contain all prompts used in FluidVoice. Copy them to your eval repo as reference:

```bash
# Create reference directory
mkdir -p reference/prompts

# Copy the key files (run from FluidVoice root)
# These paths are relative to FluidVoice repo root:
cp Sources/Fluid/Services/CommandModeService.swift reference/prompts/
cp Sources/Fluid/Services/RewriteModeService.swift reference/prompts/
cp Sources/Fluid/ContentView.swift reference/prompts/
cp Sources/Fluid/Networking/AppleIntelligenceProvider.swift reference/prompts/
```

## 3. Create Prompt Catalog

Create a `prompts/` directory structure with extracted prompts:

```
prompts/
├── command_mode/
│   └── system_prompt.txt       # Agentic terminal agent prompt (710-790 in CommandModeService.swift)
├── write_mode/
│   └── system_prompt.txt       # Write mode prompt (182-193 in RewriteModeService.swift)
├── rewrite_mode/
│   └── system_prompt.txt       # Rewrite mode prompt (196-205 in RewriteModeService.swift)
├── dictation_mode/
│   └── system_prompt.txt       # Text cleanup prompt (1147-1181 in ContentView.swift)
└── apple_intelligence/
    ├── write_mode_prompt.txt   # (78-83 in AppleIntelligenceProvider.swift)
    └── rewrite_mode_prompt.txt # (85-90 in AppleIntelligenceProvider.swift)
```

## 4. Create Test Cases

Create evaluation test cases for each mode:

```
tests/
├── command_mode/
│   ├── test_cases.json         # Input commands + expected behaviors
│   └── edge_cases.json         # Problematic inputs for smaller models
├── write_mode/
│   ├── test_cases.json
│   └── edge_cases.json
├── rewrite_mode/
│   ├── test_cases.json
│   └── edge_cases.json
└── dictation_mode/
    ├── test_cases.json         # Transcription cleanup cases
    └── edge_cases.json         # Cases where AI adds unwanted content
```

### Example Test Case Format (test_cases.json):

```json
{
  "tests": [
    {
      "id": "dictation-001",
      "name": "Question should not be answered",
      "input": "uh what is the capital of france",
      "expected_output": "What is the capital of France?",
      "must_not_contain": ["Paris", "The capital", "France is"],
      "must_contain": ["?"],
      "category": "question_passthrough"
    },
    {
      "id": "dictation-002",
      "name": "Voice commands processed",
      "input": "hello new line how are you question mark",
      "expected_output": "Hello\nHow are you?",
      "category": "voice_commands"
    }
  ]
}
```

## 5. Create Evaluation Script

Create a Python evaluation harness:

```python
# eval/run_eval.py
import json
import asyncio
from openai import OpenAI

class PromptEvaluator:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
    
    def evaluate(self, system_prompt: str, test_cases: list) -> dict:
        results = []
        for test in test_cases:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test["input"]}
                ],
                temperature=0.2
            )
            output = response.choices[0].message.content
            
            # Check pass/fail
            passed = self._check_output(output, test)
            results.append({
                "test_id": test["id"],
                "input": test["input"],
                "expected": test.get("expected_output"),
                "actual": output,
                "passed": passed
            })
        
        return {
            "model": self.model,
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "results": results
        }
    
    def _check_output(self, output: str, test: dict) -> bool:
        # Check must_contain
        for term in test.get("must_contain", []):
            if term not in output:
                return False
        # Check must_not_contain
        for term in test.get("must_not_contain", []):
            if term in output:
                return False
        return True
```

## 6. Define Model Test Matrix

Create a configuration for models to test:

```yaml
# config/models.yaml
models:
  # Flagship models (baseline)
  - name: gpt-4o
    provider: openai
    base_url: https://api.openai.com/v1
    
  - name: claude-sonnet-4-20250514
    provider: anthropic
    base_url: https://api.anthropic.com/v1
  
  # Smaller/Open models (need optimization)
  - name: llama-3.3-70b-versatile
    provider: groq
    base_url: https://api.groq.com/openai/v1
    
  - name: qwen-2.5-32b
    provider: together
    base_url: https://api.together.xyz/v1
    
  - name: gemma-2-27b
    provider: groq
    base_url: https://api.groq.com/openai/v1
    
  - name: deepseek-chat
    provider: deepseek
    base_url: https://api.deepseek.com/v1
```

## 7. Run Evaluation

```bash
# Install dependencies
uv sync

# Run the single model evaluator
uv run eval/simple_eval.py --model "llama-3.3-70b-versatile" --provider groq

# Run with a different model
uv run eval/simple_eval.py --model "gpt-4o"
```

## 8. Analyze Results

Look for common failure patterns:
- **Instruction following**: Does the model follow "output ONLY" rules?
- **Role confusion**: Does the model try to be helpful when it shouldn't?
- **Context drift**: Does the model forget system prompt over long contexts?
- **Format compliance**: Does the model follow structured output requirements?

## 9. Iterate on Prompts

For each failing model:
1. Identify failure pattern
2. Create variant prompt with stronger constraints
3. Re-run eval to verify improvement
4. Document what worked

### Common Fixes for Smaller Models:

1. **Add explicit "DO NOT" rules** - Smaller models need more explicit negatives
2. **Use XML tags** - `<output>...</output>` helps with format compliance
3. **Add end-marker** - "End your response with [END]" for clear termination
4. **Shorten system prompt** - Smaller models lose context with long prompts
5. **Add few-shot examples** - Concrete examples help more than abstract rules

## 10. Document & Apply

Once you find improved prompts:
1. Document the changes and why they work
2. Create a PR to FluidVoice with the updated prompts
3. Consider model-specific prompt variants if needed

---

## File Reference Summary

| Mode | Source File | Line Range | Description |
|------|-------------|------------|-------------|
| Command Mode | `CommandModeService.swift` | 711-790 | Agentic terminal agent system prompt |
| Write Mode | `RewriteModeService.swift` | 182-193 | Content generation prompt |
| Rewrite Mode | `RewriteModeService.swift` | 196-205 | Text transformation prompt |
| Dictation | `ContentView.swift` | 1147-1181 | Transcription cleanup prompt |
| Apple Intelligence Write | `AppleIntelligenceProvider.swift` | 78-83 | AI write mode prompt |
| Apple Intelligence Rewrite | `AppleIntelligenceProvider.swift` | 85-90 | AI rewrite mode prompt |
