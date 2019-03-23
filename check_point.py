import json
from pprint import pprint

def point_inside_polygon(x,y,poly):

    n = len(poly)
    inside =False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def get_jason_return_lis():
	path ="data.json"

	# with open('data.json') as f:
	with open(path) as f:

	    data = json.load(f)

	lis = []
	get_geometry = data[0]["Label"]["Cavity"][0]["geometry"]
	for i in range(len(get_geometry)):
		x = data[0]["Label"]["Cavity"][0]["geometry"][i]["x"]  
		y = data[0]["Label"]["Cavity"][0]["geometry"][i]["y"]
		lis.append((x,y))

	print(lis)
	return lis
lis = get_jason_return_lis()
print(point_inside_polygon(1300,1100,lis))
