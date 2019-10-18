import os
import os.path as osp

def create_path(path):
    if not osp.exists(path):
        os.makedirs(path)

color_video_root = "D:\datasets\SLR_dataset\S500_color_video"
train_list = open("../input/train_list.txt","w")
val_list = open("../input/val_list.txt","w")

color_video_path_list = os.listdir(color_video_root)
color_video_path_list.sort()
n = len(color_video_path_list)
for i,color_video_path in enumerate(color_video_path_list):
    print("%d/%d"%(i,n))
    label = color_video_path
    abs_color_video_path = osp.join(color_video_root,color_video_path)
    color_video_list = os.listdir(abs_color_video_path)
    color_video_list.sort()
    for color_video in color_video_list:
        abs_color_video = osp.join(abs_color_video_path,color_video)
        if(osp.isdir(abs_color_video)):
            p = color_video.split('_')
            person = int(p[0].lstrip('P'))
            num_frames = len(os.listdir(abs_color_video))
            record = osp.join(color_video_path,color_video)+"\t"+\
                                str(num_frames)+"\t"+color_video_path+"\n"
            if person<=25:
                train_list.write(record)
            else:
                val_list.write(record)
