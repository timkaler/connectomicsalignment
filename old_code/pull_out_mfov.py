import json
import sys
d = json.loads(open(sys.argv[1]).read())
new_list = []
for x in d:
  if x["mfov"] == int(sys.argv[2]):
    new_list.append(x)

print json.dumps(new_list)
