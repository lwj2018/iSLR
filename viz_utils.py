import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torchvision import models

def attentionmap_visualize(input, attention_map):
    '''
        visualize attention map
    Args:
        input (torch.Tensor[NxC(Lx3)xHxW]): input tensor
        attention_map (torch.Tensor[LX1XhXw]): attention map
    '''
    input_mean = np.array([0.485, 0.456, 0.406])
    input_std = np.array([0.229, 0.224, 0.225])
    input = input.detach().cpu().numpy()
    attention_map = attention_map.detach().cpu().numpy()
    input = input.reshape([-1,3,input.shape[2],input.shape[3]])
    input = input.transpose([0,2,3,1])
    attention_map = attention_map.transpose([0,2,3,1])
    for i,(image, att_map) in enumerate(zip(input, attention_map)):
        # image反标准化
        image = image*input_std+input_mean
        # 缩放attention map
        att_map = np.squeeze(att_map)
        att_map = normalize(att_map)
        att_map = Image.fromarray(att_map)
        height = image.shape[0]
        width = image.shape[1]
        att_map = att_map.resize((width, height), Image.BICUBIC)
        att_map = np.array(att_map)
        no_trans_map,  map_on_image = apply_colormap_on_image(image, att_map, 'hsv')
        plt.subplot(4,4,i+1)
        plt.imshow(map_on_image)
    plt.show()

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_im (numpy arr): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    org_im = (org_im*255).astype(np.uint8)
    org_im = Image.fromarray(org_im)
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def normalize(att_map):
    att_map = (att_map-att_map.min())/(att_map.max()-att_map.min())
    return att_map