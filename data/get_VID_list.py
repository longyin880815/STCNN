import os
import numpy as np

dirs = ['/home/xk/Dataset/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/',
        '/home/xk/Dataset/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0001/',
        '/home/xk/Dataset/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002/',
        '/home/xk/Dataset/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0003/',
        '/home/xk/Dataset/ILSVRC2015/Data/VID/val/',
        '/home/xk/Dataset/ILSVRC2015/Data/VID/test/'
        ]



file_write_obj = open('VID_seqs_list.txt','w')
for dir in dirs:
    seqs = np.sort(os.listdir(dir))
    for seq in seqs:
        seq_path = os.path.join(dir,seq)
        file_write_obj.writelines(seq_path)
        file_write_obj.write('\n')

file_write_obj.close()