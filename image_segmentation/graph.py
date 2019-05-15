import json

with open('seg_5_15_7a.json', 'r') as f:
    json_obj = json.load(f)
    for obj in json_obj:
        print(obj) #key
        # print(json_obj[obj]) val
