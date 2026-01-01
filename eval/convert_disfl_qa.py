import json
import os
import re

def convert_disfl_qa(input_path, output_path, limit=None):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    converted_tests = []
    
    # Common fillers to definitely exclude
    filler_words = ['uh', 'um', 'sorry', 'wait', 'scratch that', 'rather', 'actually']
    
    count = 0
    for qid, content in data.items():
        if limit and count >= limit:
            break
            
        disfluent = content['disfluent']
        original = content['original']
        
        # Skip weird data like #VALUE!
        if disfluent == "#VALUE!":
            continue
            
        # Create a basic test case
        test_case = {
            "id": f"disfl-qa-{qid[:8]}",
            "name": f"Disfl-QA: {original[:30]}...",
            "input": disfluent,
            "expected_output": original,
            "must_contain": [],
            "must_not_contain": [],
            "difficulty": "ultimate"
        }
        
        # Simple keyword extraction for must_contain (longer words from original)
        words = re.findall(r'\w+', original)
        unique_long_words = sorted(list(set([w for w in words if len(w) > 4])), key=len, reverse=True)
        test_case["must_contain"] = unique_long_words[:3]
        
        # Extract must_not_contain (words that are in disfluent but not in original)
        d_words = set(re.findall(r'\w+', disfluent.lower()))
        o_words = set(re.findall(r'\w+', original.lower()))
        difference = d_words - o_words
        
        # Filter difference for common fillers we definitely want gone
        killers = [w for w in difference if w in filler_words or len(w) > 3]
        test_case["must_not_contain"] = killers[:3]
        
        converted_tests.append(test_case)
        count += 1

    output_data = {
        "mode": "dictation",
        "description": f"Disfl-QA dataset ({len(converted_tests)} cases)",
        "tests": converted_tests
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Successfully converted {len(converted_tests)} cases to {output_path}")

if __name__ == "__main__":
    import sys
    # Full dev dataset
    convert_disfl_qa('tests/disfl_qa/dev.json', 'tests/dictation_mode/disfl_qa_dev_full.json')
    # 50-case dev subset
    convert_disfl_qa('tests/disfl_qa/dev.json', 'tests/dictation_mode/disfl_qa_dev_subset.json', limit=50)
    # 20-case training sample
    convert_disfl_qa('tests/disfl_qa/train.json', 'tests/dictation_mode/disfl_qa_train_sample.json', limit=20)


