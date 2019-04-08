import os
import numpy as np

dirs = '/home/xk/Dataset/DAVIS/JPEGImages/480p/'
anno_dir = '/home/xk/Dataset/DAVIS/Annotations/480p/'
seqs_file = '/home/xk/Dataset/DAVIS/ImageSets/2016/train.txt'
f = open(seqs_file,'r')
seq_names = f.readlines()
file_write_obj = open('DAVIS16_samples_list_4.txt','w')
for seq in seq_names:
    seq = seq.strip()
    seq_path = os.path.join(dirs,seq)
    anno_seq_path = os.path.join(anno_dir,seq)
    frame_name_list = sorted([f for f in os.listdir(seq_path) if f.endswith(".jpg")])
    frame_path_list = sorted([os.path.join(seq_path,x) for x in frame_name_list])
    anno_name_list = sorted([f for f in os.listdir(anno_seq_path) if f.endswith(".png")])
    anno_path_list = sorted([os.path.join(anno_seq_path, x) for x in anno_name_list])
    # order
    for i in range(4,len(frame_path_list)):
        line = frame_path_list[i-4]+','+frame_path_list[i-3]+','+frame_path_list[i-2]+','+frame_path_list[i-1]\
             + ';'+frame_path_list[i]+';'+anno_path_list[i]
        file_write_obj.writelines(line)
        file_write_obj.write('\n')
    # border
    line = frame_path_list[0] + ',' + frame_path_list[0] + ',' + frame_path_list[0] + ',' + frame_path_list[
        0] + ';' + frame_path_list[1] + ';' + anno_path_list[1]
    file_write_obj.writelines(line)
    file_write_obj.write('\n')

    line = frame_path_list[0] + ',' + frame_path_list[0] + ',' + frame_path_list[0] + ',' + frame_path_list[
        1] + ';' + frame_path_list[2] + ';' + anno_path_list[2]
    file_write_obj.writelines(line)
    file_write_obj.write('\n')
    line = frame_path_list[0] + ',' + frame_path_list[0] + ',' + frame_path_list[1] + ',' + frame_path_list[
        2] + ';' + frame_path_list[3] + ';' + anno_path_list[3]
    file_write_obj.writelines(line)
    file_write_obj.write('\n')
    # reverse order
    for i in range(0,len(frame_path_list)-4):
        line = frame_path_list[i+4]+','+frame_path_list[i+3]+','+frame_path_list[i+2]+','+frame_path_list[i+1]\
             + ';'+frame_path_list[i]+';'+anno_path_list[i]
        file_write_obj.writelines(line)
        file_write_obj.write('\n')



file_write_obj.close()