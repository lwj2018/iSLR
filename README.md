# ISLR
## 使用
### 预处理
- 修改process/video_to_frame.py中color_video_root参数
- cd process 运行video_to_frame.py，将中科大islr数据集中的视频转换为图片
- 根据数据集路径，改变opts.py中video_root, train_file, val_file参数
- 修改process/generate_list.py开头color_video_root参数,代表视频的根目录
- 修改process/generate_list.py开头num_class参数，代表手语词汇总数
- 运行generate_list.py，生成训练列表和测试列表
### 训练
- 根据数据集路径修改opts.py中video_root参数
- 设置opts.py中代表词汇总数的num_class参数
- 设置opts.py中代表backbone类型的arch参数