import os
import numpy as np
from pyLucid.patchPaint import paint
import cv2
from PIL import Image
from pyLucid.lucidDream import dreamData
from mypath import Path

db_root_dir = Path.db_root_dir()
save_dir = os.path.join(db_root_dir,'first_frame')
with open(os.path.join(db_root_dir, 'ImageSets/2016/', 'val.txt')) as f:
	seqnames = f.readlines()

for i in range(len(seqnames)):
	seq_name = seqnames[i].strip()
	save_path = os.path.join(save_dir,seq_name)
	dream_dir = os.path.join(save_path,'dream')
	if not os.path.exists(dream_dir):
		os.makedirs(os.path.join(dream_dir))

	names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
	img_list = list(map(lambda x: os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name), x), names_img))
	name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
	labels = list(map(lambda x: os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name), x), name_label))
	img_path = img_list[0]
	label_path = labels[0]
	Iorg = cv2.imread(img_path)
	Morg = Image.open(label_path)
	palette = Morg.getpalette()
	Morg = np.array(Morg)
	Morg[Morg>0]=1
	bg = paint(Iorg, np.array(Morg), False)
	cv2.imwrite(os.path.join(save_path,'bg.jpg'), bg)
	# bg = cv2.imread(os.path.join(save_path,'bg.jpg'))
	for i in range(100):
		im_1, gt_1, bb_1 = dreamData(Iorg, np.array(Morg), bg, False)
		# Image 1 in this pair.
		cv2.imwrite(os.path.join(dream_dir,'%03d.jpg'%i), im_1)

		# Mask for image 1.
		gtim1 = Image.fromarray(gt_1, 'P')
		gtim1.putpalette(palette)
		gtim1.save(os.path.join(dream_dir,'%03d.png'%i))
