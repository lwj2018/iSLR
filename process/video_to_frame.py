import os
import os.path as osp

def create_path(path):
    if not osp.exists(path):
        os.makedirs(path)

color_video_root = "E:\datasets\SLR_dataset\S500_color_video"
# 切换工作目录
os.chdir(color_video_root)
color_video_path_list = os.listdir(color_video_root)
color_video_path_list.sort()
n = len(color_video_path_list)
for i,color_video_path in enumerate(color_video_path_list):
    print("%d/%d"%(i,n))
    abs_color_video_path = osp.join(color_video_root,color_video_path)
    color_video_list = os.listdir(color_video_path)
    color_video_list.sort()
    index = int(color_video_path)
    if color_video_path == "000238":
        for color_video in color_video_list:
            abs_color_video = osp.join(color_video_path,color_video)
            out_color_video_path = abs_color_video.rstrip(".avi")
            create_path(out_color_video_path)
            cmd = "ffmpeg -i "+abs_color_video+" "+out_color_video_path+"/%06d.jpg" 
            os.system(cmd)
