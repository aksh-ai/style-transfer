import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms, models

def load_img(img_path, img_size = 512, shape = None):
  img = Image.open(img_path).convert('RGB')

  if max(img.size) < img_size:
    img_size = max(img.size)
  
  if shape is not None:
    img_size = shape

  transform = transforms.Compose([transforms.Resize(img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  return transform(img).unsqueeze(0)

def np_convert(tensor):
  img = tensor.cpu().clone().detach().numpy()
  
  img = img.squeeze()
  
  img = img.transpose(1, 2, 0)
  
  img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  
  return img.clip(0, 1)

def get_features(img, model):
  layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
  
  features = {}

  for name, layer in model._modules.items():
    img = layer(img)
    
    if name in layers:
      features[layers[name]] = img
    
  return features

def gram_matrix(tensor):
  _, c, h, w = tensor.size()

  tensor = tensor.view(c, h*w)

  gram= torch.mm(tensor, tensor.t())

  return gram