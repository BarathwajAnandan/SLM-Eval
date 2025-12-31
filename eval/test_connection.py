import os
import yaml
import argparse
import sys
from openai import OpenAI
import httpx
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

def test_model(model_name: str, custom_prompt: str = None):
    # 1. Load Config
    try:
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        console.print("[red]Error: config/models.yaml not found.[/red]")
        sys.exit(1)

    # 2. Find model entry
    model_entry = next((m for m in config.get("models", []) if m["name"] == model_name), None)
    if not model_entry:
        available_models = [m["name"] for m in config.get("models", [])]
        console.print(f"[red]Error: Model '{model_name}' not found in config/models.yaml.[/red]")
        console.print(f"[yellow]Available models: {', '.join(available_models)}[/yellow]")
        sys.exit(1)

    # 3. Resolve provider
    provider_name = model_entry["provider"]
    provider_info = config.get("providers", {}).get(provider_name)
    if not provider_info:
        console.print(f"[red]Error: Provider '{provider_name}' for model '{model_name}' is not defined in providers list.[/red]")
        sys.exit(1)

    # 4. Get API Key
    api_key_env = provider_info.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else provider_info.get("api_key")
    
    if not api_key:
        console.print(f"[red]Error: No API key found for {provider_name}.[/red]")
        console.print(f"[yellow]Please set the '{api_key_env}' environment variable.[/yellow]")
        sys.exit(1)

    # 5. Initialize Client
    client = OpenAI(
        base_url=provider_info.get("base_url"),
        api_key=api_key
    )

    prompt = custom_prompt if custom_prompt else "Hi, how are you? Please respond with a short sentence."
    
    console.print(Panel(f"Testing [bold cyan]{model_name}[/bold cyan] via [bold magenta]{provider_name}[/bold magenta]\n"
                        f"Endpoint: [dim]{provider_info.get('base_url')}[/dim]\n"
                        f"Prompt: [italic]\"{prompt}\"[/italic]", 
                        title="Connection Test", border_style="cyan"))

    # 6. Make Call
    try:
        # Check if it's a reasoning model (same logic as simple_eval.py)
        model_lower = model_name.lower()
        is_reasoning = (model_lower.startswith("gpt-5") or 
                        "gpt-5." in model_lower or 
                        model_lower.startswith("o1") or 
                        model_lower.startswith("o3") or 
                        "gpt-oss" in model_lower or 
                        model_lower.startswith("openai/") or
                        "nvidia/" in model_lower or
                        ("deepseek" in model_lower and "reasoner" in model_lower))

        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if is_reasoning:
            body["max_completion_tokens"] = 100
            
            # Default reasoning effort logic
            reasoning_effort = None
            enable_thinking = None
            
            if "gpt-oss" in model_lower or model_lower.startswith("openai/"):
                reasoning_effort = "medium"
            elif model_lower.startswith("o1"):
                reasoning_effort = "medium"
            elif "deepseek" in model_lower and "reasoner" in model_lower:
                enable_thinking = True

            # YAML Overrides (highest priority)
            if "reasoning_effort" in model_entry:
                reasoning_effort = model_entry["reasoning_effort"]
            if "enable_thinking" in model_entry:
                enable_thinking = model_entry["enable_thinking"]

            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
            
            if "nvidia/" in model_lower:
                # Nvidia specific extra_body
                eb = {}
                if "reasoning_budget" in model_entry:
                    eb["reasoning_budget"] = model_entry["reasoning_budget"]
                if "enable_thinking" in model_entry:
                    eb["chat_template_kwargs"] = {"enable_thinking": model_entry["enable_thinking"]}
                if eb:
                    body["extra_body"] = eb
            else:
                if enable_thinking is not None:
                    body["enable_thinking"] = enable_thinking
        else:
            body["max_tokens"] = 100
            body["temperature"] = 0.7

        if model_entry.get("use_responses_api"):
            # Use httpx for raw request to /responses
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            resp_body = {
                "model": model_name,
                "input": [{"role": "user", "content": prompt}],
                "text": {"format": {"type": "text"}},
                "reasoning": {},
                "max_output_tokens": 100,
                "temperature": 1,
                "store": True,
                "include": ["reasoning.encrypted_content"]
            }
            
            with httpx.Client(timeout=30.0) as h_client:
                resp = h_client.post(
                    "https://api.openai.com/v1/responses",
                    headers=headers,
                    json=resp_body
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("output", {}).get("text", "")
                if not content and "choices" in data:
                    content = data["choices"][0]["message"]["content"]
        else:
            response = client.chat.completions.create(**body)
            content = response.choices[0].message.content
        
        console.print("[bold green]✓ Connection Successful![/bold green]")
        console.print(Panel(content, title=f"Response from {model_name}", border_style="green"))

    except Exception as e:
        console.print("[bold red]✗ Connection Failed![/bold red]")
        
        # Friendly error mapping
        err_msg = str(e)
        if hasattr(e, "response") and hasattr(e.response, "text"):
             err_msg += f" - {e.response.text}"
        
        if "401" in err_msg:
            hint = "Incorrect API key. Please check your .env or environment variables."
        elif "404" in err_msg:
            hint = f"Model '{model_name}' was not found at this endpoint. Check the model name string."
        elif "connection error" in err_msg.lower():
            hint = "Could not reach the server. Check your internet connection or the base_url."
        else:
            hint = "Check the console output below for raw error details."
            
        console.print(f"[red][bold]Reason:[/bold] {err_msg}[/red]")
        console.print(f"[yellow][bold]Hint:[/bold] {hint}[/yellow]")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickly test LLM connection and chat.")
    parser.add_argument("model", help="Model name as defined in models.yaml")
    parser.add_argument("--prompt", help="Optional custom prompt to send")
    
    args = parser.parse_args()
    test_model(args.model, args.prompt)
