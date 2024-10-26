import json


with open('commonsense_170k.json') as f:
    data = json.load(f)

print(len(data))