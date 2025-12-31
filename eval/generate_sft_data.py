import json
import os
import argparse

def generate_sft_data(tests_path: str, prompt_path: str, output_path: str):
    # 1. Load System Prompt
    try:
        with open(prompt_path, "r") as f:
            system_prompt = f.read().strip()
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return

    # 2. Load Test Cases
    try:
        with open(tests_path, "r") as f:
            data = json.load(f)
            tests = data.get("tests", [])
    except Exception as e:
        print(f"Error loading tests: {e}")
        return

    # 3. Convert to OpenAI/ChatML JSONL format
    dataset = []
    for test in tests:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test["input"]},
                {"role": "assistant", "content": test["expected_output"]}
            ]
        }
        dataset.append(entry)

    # 4. Save to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"Successfully generated {len(dataset)} SFT examples at {output_path}")
    print(f"Note: To effectively fine-tune (LoRA), you should aim for 200-500+ similar examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert evaluation test cases to SFT training data.")
    parser.add_argument("--tests", default="tests/dictation_mode/test_cases.json", help="Path to test cases JSON")
    parser.add_argument("--prompt", default="prompts/dictation_mode/system_prompt.txt", help="Path to system prompt TXT")
    parser.add_argument("--output", default="training/sft_dataset.jsonl", help="Path to output JSONL file")
    
    args = parser.parse_args()
    generate_sft_data(args.tests, args.prompt, args.output)
