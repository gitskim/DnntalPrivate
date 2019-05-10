### Take mask and orginal image and return the orginal with the mask on it - color 

import numpy as np
import matplotlib.pyplot as plt

def imageshow(img, dpi=200):
	if dpi > 0:
	    F = plt.gcf()
	    F.set_dpi(dpi)
	plt.imshow(img)




def checking(image_orginal,image_mask):
    orginal_array = image_orginal.copy()
    mask_array = image_mask.copy()
    
    length, width, z = image_orginal.shape
    black_px = np.asarray([(0,255,0)])

    for x in range(0,width,1):
        for y in range(0,length,1):
            
            a,b,c,d = mask_array[y:y+1,x:x+1][0][0]
            if (a,b,c,d) != (0,0,0,255):
                orginal_array[y:y+1,x:x+1][0][0] = black_px
    return orginal_array


image = plt.imread('file2.jpg')
image2 =plt.imread('org_file2.jpg')
plt.imshow(checking(image2,image))