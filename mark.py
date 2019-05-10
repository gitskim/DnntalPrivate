import numpy as np
import matplotlib.pyplot as plt

def imageshow(img, dpi=200):
	if dpi > 0:
	    F = plt.gcf()
	    F.set_dpi(dpi)
	plt.imshow(img)


def checking(image):
    array2 = image.copy()
    length,wide, z = image.shape
    size = 128 
    length = int(length/size)*size
    wide = int(wide/size)*size

    for x in range(0,wide,size):
        for y in range(0,length,size):
            
            crop = array2[y:y+size,x:x+size]
            # Send to predcit 
            # number = model.predict(crop)
            number = np.random.randint(0,2)
            if number == 1:
                array2[y:y+size,x:x+size]=0
    return array2


image = plt.imread('puppy.jpg')
plt.imshow(checking(image))



