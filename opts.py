import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_root",type=str,
                    default="E:/datasets/SLR_dataset/S500_color_video")
parser.add_argument("--train_file",type=str,
                    default="input/train_list.txt")
parser.add_argument("--val_file",type=str,
                    default="input/val_list.txt")
parser.add_argument('--root_model', type=str, 
                    default='models')

# parser.add_argument('--arch', type=str, default='resnet34')
# parser.add_argument('--arch', type=str, default='BNInception')
parser.add_argument('--arch', type=str, default='Resnet_cbam')

parser.add_argument('--modality', type=str, default='RGB')
parser.add_argument('--num_class', type=int, default=5)
parser.add_argument('--hidden_unit', type=int, default=1024)
# ========================= Learning Configs ==========================
parser.add_argument('--start_epoch',default=0, type=int)
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
# parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
#                     metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')

# ========================= Runtime Configs ==========================
# workers 原默认值为30
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--val_resume', 
        default='models\iSLR_RGB_Resnet_cbam_class5_hidden1024_best.pth.tar', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
