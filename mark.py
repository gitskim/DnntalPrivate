import numpy as np
import matplotlib.pyplot as plt

def imageshow(img, dpi=200):
	if dpi > 0:
	    F = plt.gcf()
	    F.set_dpi(dpi)
	plt.imshow(img)


def checking(image):
    array2 = image.copy()
    a,b,c = image.shape

    
    for x in range(0,b,30):
        for y in range(0,a,30):
            
            crop = array2[y:y+20,x:x+20]
            # Send to predcit 
            number = np.random.randint(0,2)
            ## the result of the predict 
            if number == 1:
                array2[y:y+20,x:x+20]=0
    return array2

	#return array2 

image = plt.imread('puppy.jpg')
plt.imshow(checking(image))



