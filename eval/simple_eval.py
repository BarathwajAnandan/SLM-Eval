import os
import json
import yaml
import argparse
import asyncio
import csv
import re
import difflib
from datetime import datetime
from typing import List, Dict, Any
import httpx
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

console = Console()

class PromptEvaluator:
    def __init__(self, model_name: str, provider_info: Dict[str, Any]):
        self.model = model_name
        self.api_key = os.getenv(provider_info.get("api_key_env", ""), "")
        if not self.api_key:
            self.api_key = provider_info.get("api_key", "")
            
        self.client = OpenAI(
            base_url=provider_info.get("base_url"),
            api_key=self.api_key
        )

    def is_reasoning_model(self, model: str) -> bool:
        model_lower = model.lower()
        return (model_lower.startswith("gpt-5") or
                "gpt-5." in model_lower or
                model_lower.startswith("o1") or
                model_lower.startswith("o3") or
                "gpt-oss" in model_lower or
                model_lower.startswith("openai/") or
                "nvidia/" in model_lower or
                ("deepseek" in model_lower and "reasoner" in model_lower))

    def strip_thinking(self, text: str) -> str:
        # Pattern matches both <think>...</think> and <thinking>...</thinking> including multiline
        pattern = r"<think(?:ing)?>([\s\S]*?)</think(?:ing)?>"
        # Also handle orphan closing tags just in case
        orphan_pattern = r"^([\s\S]*?)</think(?:ing)?>"
        
        cleaned = re.sub(pattern, "", text)
        cleaned = re.sub(orphan_pattern, "", cleaned)
        
        # Remove any stray tags
        cleaned = cleaned.replace("</think>", "").replace("</thinking>", "")
        cleaned = cleaned.replace("<think>", "").replace("<thinking>", "")
        
        return cleaned.strip()

    async def run_test(self, system_prompt: str, test_case: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        try:
            is_reasoning = self.is_reasoning_model(self.model)
            
            # Build request body mirroring FluidVoice's LLMClient.swift
            body: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_case["input"]}
                ]
            }
            
            # Reasoning models don't support temperature
            if not is_reasoning:
                body["temperature"] = settings.get("temperature", 0.1)
            
            # Max tokens logic
            max_tokens = settings.get("max_tokens", 1024)
            if is_reasoning:
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens
                
            # 1. Start with defaults or hardcoded logic
            model_lower = self.model.lower()
            reasoning_effort = None
            enable_thinking = None
            
            # 2. Hardcoded defaults (matching FluidVoice)
            if "gpt-oss" in model_lower or model_lower.startswith("openai/"):
                reasoning_effort = "medium"
            elif model_lower.startswith("o1"):
                reasoning_effort = "medium"
            elif "deepseek" in model_lower and "reasoner" in model_lower:
                enable_thinking = True

            # 3. YAML Overrides (highest priority)
            # evaluator was initialized with model_info, can check there
            if "reasoning_effort" in settings:
                reasoning_effort = settings["reasoning_effort"]
            if "enable_thinking" in settings:
                enable_thinking = settings["enable_thinking"]
            
            # NVIDIA specific params
            reasoning_budget = settings.get("reasoning_budget")

            # Apply to body
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
            
            if "nvidia/" in model_lower:
                # Nvidia uses extra_body for these
                eb = {}
                if reasoning_budget:
                    eb["reasoning_budget"] = reasoning_budget
                if enable_thinking is not None:
                    eb["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
                if eb:
                    body["extra_body"] = eb
            else:
                if enable_thinking is not None:
                    body["enable_thinking"] = enable_thinking

            # Check for Responses API (OpenAI GPT-5.1 style)
            if settings.get("use_responses_api"):
                # Use httpx for raw request to /responses
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                # Map messages to input
                resp_body = {
                    "model": self.model,
                    "input": [
                        {"role": m["role"], "content": m["content"]} for m in body["messages"]
                    ],
                    "text": {"format": {"type": "text"}},
                    "reasoning": {},
                    "max_output_tokens": settings.get("max_tokens", 2048),
                    "temperature": settings.get("temperature", 1),
                    "top_p": settings.get("top_p", 1),
                    "store": True,
                    "include": [
                        "reasoning.encrypted_content"
                    ]
                }
                
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        "https://api.openai.com/v1/responses",
                        headers=headers,
                        json=resp_body
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # Response format for /responses typically has the text in a nested field
                    # Based on standard response patterns:
                    raw_output = data.get("output", {}).get("text", "")
                    if not raw_output and "choices" in data:
                         raw_output = data["choices"][0]["message"]["content"]
            else:
                response = self.client.chat.completions.create(**body)
                # Some models (like NVIDIA gpt-oss) return reasoning in a separate attribute
                raw_output = response.choices[0].message.content or ""
                # Optionally log reasoning_content if present (but don't include in output)
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                if reasoning:
                    # Could log this for debugging, but we only use the final content
                    pass
            
            # FluidVoice strips thinking before using the content
            output = self.strip_thinking(raw_output)
            
            # 2. Similarity Score (Fuzzy matching)
            expected = test_case.get("expected_output", "")
            similarity = 0.0
            if expected:
                similarity = difflib.SequenceMatcher(None, output.lower(), expected.lower()).ratio()
            
            # Simple validation logic
            passed = True
            failures = []
            
            for term in test_case.get("must_contain", []):
                if term.lower() not in output.lower():
                    passed = False
                    failures.append(f"Missing: '{term}'")
                    
            for term in test_case.get("must_not_contain", []):
                if term.lower() in output.lower():
                    passed = False
                    failures.append(f"Should not contain: '{term}'")
            
            return {
                "id": test_case["id"],
                "name": test_case["name"],
                "difficulty": test_case["difficulty"],
                "input": test_case["input"],
                "expected": expected,
                "output": output,
                "passed": passed,
                "similarity": f"{similarity*100:.1f}%",
                "error": None,
                "failures": failures
            }
        except Exception as e:
            return {
                "id": test_case["id"],
                "name": test_case["name"],
                "difficulty": test_case["difficulty"],
                "input": test_case["input"],
                "expected": test_case.get("expected_output", ""),
                "output": None,
                "passed": False,
                "similarity": "0%",
                "error": str(e),
                "failures": ["API Error"]
            }

async def main():
    parser = argparse.ArgumentParser(description="FluidVoice Prompt Evaluator")
    parser.add_argument("--model", type=str, help="Model name (e.g., llama-3-8b)")
    parser.add_argument("--provider", type=str, help="Provider name (openai, groq, etc.)")
    parser.add_argument("--difficulty", type=str, help="Filter by difficulty (easy, medium, hard)")
    parser.add_argument("--prompt", type=str, default="system_prompt.txt", help="Prompt file name in prompts/dictation_mode/")
    parser.add_argument("-i", "--input", type=str, default="test", choices=["test", "val"], help="Input test set: 'test' (training) or 'val' (validation)")
    parser.add_argument("--ls", action="store_true", help="List all available models")
    args = parser.parse_args()

    # 1. Load Config
    try:
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        console.print("[red]Error: config/models.yaml not found.[/red]")
        return

    # Handle --ls: list all models and exit
    if args.ls:
        models = config.get("models", [])
        by_provider = {}
        for m in models:
            provider = m.get("provider", "unknown")
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(m["name"])
        
        console.print("[bold cyan]Available Models:[/bold cyan]")
        for provider, model_list in sorted(by_provider.items()):
            console.print(f"\n[yellow]{provider}[/yellow] ({len(model_list)} models)")
            for name in model_list:
                console.print(f"  • {name}")
        return

    # 2. Find model info and merge with provider info
    model_entry = None
    if args.model:
        # Search for model name
        for m in config.get("models", []):
            if m["name"] == args.model:
                model_entry = m
                break
    
    if not model_entry:
        if args.model:
            # Model was specified but not found
            available_models = [m["name"] for m in config.get("models", [])]
            console.print(f"[red]Error: Model '{args.model}' not found in config/models.yaml[/red]")
            console.print(f"[yellow]Available models:[/yellow]")
            for m in available_models[:10]:  # Show first 10
                console.print(f"  • {m}")
            if len(available_models) > 10:
                console.print(f"  ... and {len(available_models) - 10} more")
            return
        else:
            console.print("[red]Error: No model specified. Use --model <name>[/red]")
            return

    # Resolve provider info
    provider_name = model_entry["provider"]
    provider_info = config.get("providers", {}).get(provider_name, {})
    
    if not provider_info:
        console.print(f"[red]Error: Provider '{provider_name}' not defined in config.[/red]")
        return

    # Merge into a single dict for the evaluator
    model_info = {**model_entry, **provider_info}

    # 3. Load Prompt & Tests
    prompt_file = args.prompt
    prompt_path = os.path.join("prompts/dictation_mode", prompt_file)
    prompt_id = os.path.splitext(prompt_file)[0].replace("system_prompt", "v").strip("_")
    if prompt_id == "v": prompt_id = "v0" # Default version
    
    if not os.path.exists(prompt_path):
        console.print(f"[red]Error: Prompt file '{prompt_path}' not found.[/red]")
        return

    with open(prompt_path, "r") as f:
        system_prompt = f.read()
    
    # Map input shorthand to actual filename
    input_files = {"test": "test_cases.json", "val": "val.json"}
    tests_filename = input_files[args.input]
    tests_path = os.path.join("tests/dictation_mode", tests_filename)
    if not os.path.exists(tests_path):
        console.print(f"[red]Error: Test file '{tests_path}' not found.[/red]")
        return
    with open(tests_path, "r") as f:
        tests_data = json.load(f)
    
    # 4. Filter tests by difficulty if specified
    all_tests = tests_data["tests"]
    if args.difficulty:
        diff_filter = args.difficulty.lower()
        tests_to_run = [t for t in all_tests if t.get("difficulty", "").lower() == diff_filter]
        if not tests_to_run:
            available_diffs = sorted(list(set(t.get("difficulty", "") for t in all_tests)))
            console.print(f"[red]Error: No tests found for difficulty '{args.difficulty}'.[/red]")
            console.print(f"[yellow]Available difficulties: {', '.join(available_diffs)}[/yellow]")
            return
        console.print(f"[cyan]Filtering for difficulty: [bold]{args.difficulty}[/bold] ({len(tests_to_run)} tests)[/cyan]")
    else:
        tests_to_run = all_tests
        console.print(f"[cyan]Running all {len(tests_to_run)} tests.[/cyan]")

    evaluator = PromptEvaluator(model_info["name"], model_info)
    
    # 5. Run Tests
    results = []
    
    table = Table(title=f"Evaluation Results: {model_info['name']}")
    table.add_column("ID", style="cyan")
    table.add_column("Difficulty", style="magenta")
    table.add_column("Name", style="white")
    table.add_column("Similarity", justify="right")
    table.add_column("Result", style="bold")
    table.add_column("Feedback", style="dim")

    with Live(table, refresh_per_second=4):
        for test in tests_to_run:
            # Merge global settings with model-specific info for overrides
            full_settings = {**config.get("settings", {}), **model_info}
            res = await evaluator.run_test(system_prompt, test, full_settings)
            results.append(res)
            
            status = "[green]PASS[/green]" if res["passed"] else "[red]FAIL[/red]"
            feedback = ", ".join(res["failures"]) if not res["passed"] else "Clean"
            
            table.add_row(
                res["id"], 
                res["difficulty"].upper(),
                res["name"],
                res["similarity"],
                status,
                feedback
            )

    # 5. Summary Statistics
    stats = {} # [pass, total]
    for r in results:
        diff = r["difficulty"].lower()
        if diff not in stats:
            stats[diff] = [0, 0]
        stats[diff][1] += 1
        if r["passed"]:
            stats[diff][0] += 1
    
    # Sort difficulties for consistent display (easy -> medium -> hard -> others)
    def diff_sort(d):
        order = {"easy": 0, "medium": 1, "hard": 2, "ultimate": 3}
        return order.get(d, 99)
    
    sorted_diffs = sorted(stats.keys(), key=diff_sort)
    console.print("\n[bold]Summary Matrix:[/bold]")
    summary_table = Table()
    summary_table.add_column("Tier")
    summary_table.add_column("Score")
    summary_table.add_column("Percentage")
    
    for tier in sorted_diffs:
        p, t = stats[tier]
        percentage = (p/t)*100 if t > 0 else 0
        color = "green" if percentage > 80 else "yellow" if percentage > 50 else "red"
        summary_table.add_row(tier.upper(), f"{p}/{t}", f"[{color}]{percentage:.1f}%[/{color}]")
    
    console.print(summary_table)

    # 6. Save results
    model_safe_name = model_info['name'].replace('/', '_')
    output_dir = os.path.join("results", prompt_id, model_safe_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort results: Failures first (passed=False < passed=True)
    results.sort(key=lambda x: x["passed"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diff_suffix = args.difficulty.lower() if args.difficulty else "all"
    
    # JSON Report
    json_path = os.path.join(output_dir, f"eval_{diff_suffix}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # CSV Report
    csv_path = os.path.join(output_dir, f"eval_{diff_suffix}_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "difficulty", "name", "similarity", "passed", "input", "expected", "output", "failures", "error"
        ])
        writer.writeheader()
        
        has_split = False
        # Only add a split if we have both failed and passed tests
        failed_count = len([r for r in results if not r["passed"]])
        passed_count = len([r for r in results if r["passed"]])
        
        for r in results:
            # Flatten failures list for CSV
            row = r.copy()
            row["failures"] = "; ".join(row["failures"])
            
            # Add a separator row between failures and passes
            if not has_split and r["passed"] and failed_count > 0:
                writer.writerow({k: "---" for k in row.keys()})
                has_split = True
                
            writer.writerow(row)

    console.print(f"\n[dim]Results saved to:[/dim]")
    console.print(f"  [cyan]• JSON: {json_path}[/cyan]")
    console.print(f"  [cyan]•  CSV: {csv_path}[/cyan]")

    # 7. Update Leaderboard
    leaderboard_path = "results/leaderboard.csv"
    file_exists = os.path.isfile(leaderboard_path)
    
    # Calculate totals
    total_passed = sum(p for p, t in stats.values())
    total_tests = sum(t for p, t in stats.values())
    total_pct = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    with open(leaderboard_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "prompt", "model", "dataset", "difficulty_filter", 
            "easy_score", "medium_score", "hard_score", "ultimate_score",
            "total_passed", "total_tests", "percentage", "report_file"
        ])
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            "timestamp": timestamp,
            "prompt": prompt_id,
            "model": model_info["name"],
            "dataset": args.input,  # 'test' or 'val'
            "difficulty_filter": args.difficulty if args.difficulty else "all",
            "easy_score": f"{stats.get('easy', [0,0])[0]}/{stats.get('easy', [0,0])[1]}",
            "medium_score": f"{stats.get('medium', [0,0])[0]}/{stats.get('medium', [0,0])[1]}",
            "hard_score": f"{stats.get('hard', [0,0])[0]}/{stats.get('hard', [0,0])[1]}",
            "ultimate_score": f"{stats.get('ultimate', [0,0])[0]}/{stats.get('ultimate', [0,0])[1]}",
            "total_passed": total_passed,
            "total_tests": total_tests,
            "percentage": f"{total_pct:.1f}%",
            "report_file": csv_path
        })
    
    console.print(f"  [cyan]• LDB: {leaderboard_path}[/cyan]")

if __name__ == "__main__":
    asyncio.run(main())
