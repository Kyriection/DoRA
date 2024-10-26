import json

with open('commonsense_170k.json') as f:
    lines = json.load(f)

print(len(lines))