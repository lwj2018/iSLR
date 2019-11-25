import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import imghdr

color_video_root = "/home/liweijie/SLR_dataset/S500_color_video"

def is_valid_image(filename):
    valid = True
    # print(imghdr.what(filename))
    if imghdr.what(filename)!='jpeg':
        valid = False
    return valid

color_video_path_list = os.listdir(color_video_root)
color_video_path_list.sort()
n = len(color_video_path_list)
for i,color_video_path in enumerate(color_video_path_list):
    print("%d/%d"%(i,n))
    label = color_video_path
    abs_color_video_path = osp.join(color_video_root,color_video_path)
    color_video_list = os.listdir(abs_color_video_path)
    color_video_list.sort()
    index = int(label)
    for color_video in color_video_list:
        abs_color_video = osp.join(abs_color_video_path,color_video)
        if(osp.isdir(abs_color_video)):
            img_list = os.listdir(abs_color_video)
            img_list.sort()
            for img_name in img_list:
                abs_img_name = osp.join(abs_color_video,img_name)
                if not is_valid_image(abs_img_name):
                    print("Wrong! %s",abs_img_name)