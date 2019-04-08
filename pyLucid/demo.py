from patchPaint import paint
import cv2
from PIL import Image
import numpy as np
from lucidDream import dreamData

Iorg=cv2.imread('img.jpg')
Morg=Image.open('gt.png')
palette=Morg.getpalette()

bg=paint(Iorg,np.array(Morg),False)
cv2.imwrite('bg.jpg',bg)
# bg=cv2.imread('bg.jpg')

# im_1,gt_1,bb_1,im_2,gt_2,bb_2,fb,ff=dreamData(Iorg,np.array(Morg),bg,True)
im_1,gt_1,bb_1=dreamData(Iorg,np.array(Morg),bg,False)
# Image 1 in this pair.
cv2.imwrite('gen1.jpg',im_1)

# Mask for image 1.
gtim1=Image.fromarray(gt_1,'P')
gtim1.putpalette(palette)
gtim1.save('gen1.png')

# Deformed previous mask for image 1.
bbim1=Image.fromarray(bb_1,'P')
bbim1.putpalette(palette)
bbim1.save('gen1bb.png')

# # Image 2 in this pair.
# cv2.imwrite('gen2.jpg',im_2)
#
# # Mask for image 2.
# gtim2=Image.fromarray(gt_2,'P')
# gtim2.putpalette(palette)
# gtim2.save('gen2.png')
#
# # Deformed previous mask for image 2.
# bbim2=Image.fromarray(bb_2,'P')
# bbim2.putpalette(palette)
# bbim2.save('gen2bb.png')
#
# # Optical flow from Image 2 to Image 1.
# # Its magnitude can be used as a guide to get mask of Image 2.
# flowmag=np.sqrt(np.sum(fb**2,axis=2))
# flowmag_norm=(flowmag-flowmag.min())/(flowmag.max()-flowmag.min())
# flowmagim=(flowmag_norm*255+0.5).astype('uint8')
# flowim=Image.fromarray(flowmagim,'L')
# flowim.save('gen2fb.png')
#
# # Optical flow from Image 1 to Image 2.
# # Its magnitude can be used as a guide to get mask of Image 1.
# flowmag=np.sqrt(np.sum(ff**2,axis=2))
# flowmag_norm=(flowmag-flowmag.min())/(flowmag.max()-flowmag.min())
# flowmagim=(flowmag_norm*255+0.5).astype('uint8')
# flowim=Image.fromarray(flowmagim,'L')
# flowim.save('gen1ff.png')