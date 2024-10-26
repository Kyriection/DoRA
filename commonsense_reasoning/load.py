import json

with open('commonsense_170k.json') as f:
    lines = json.loads(f)

print(len(lines))