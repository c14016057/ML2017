import sys
import Image
a = Image.open (sys.argv[1])
b = Image.open (sys.argv[2])
w, h = a.size
c = Image.new("RGBA",(w, h))
for i in range(0, w, 1):
	for j in range(0, h, 1):
		if a.getpixel ((i, j)) != b.getpixel ((i, j)):
			c.putpixel ((i, j), b.getpixel ((i, j)))
c.save("ans_two.png")			
	
