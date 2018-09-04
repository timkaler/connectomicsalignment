lines = open('tmp/pts-match-0001-0000.txt').readlines()

items_list = []
for line in lines[1:]:
  items=line.strip().split(' ')
  items = items[2]+" " + items[3]
  items_list.append(items + " 25") 
locations= ";".join(items_list)
import random
random.seed(42)
colors= ";".join([" ".join([str(random.randint(0,255)), str(random.randint(0,50)), str(random.randint(0,255))]) for x in items_list])


print "RGB=insertShape(ans, 'FilledCircle', ["+locations+"], 'Color', ["+colors+"]);"
