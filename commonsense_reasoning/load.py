import json

lines = []
with open('commonsense_170k.json', 'r') as f:
    for line in f:
        try:
            lines.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            print(f"Error decoding line: {line}")
            continue

print(len(lines))