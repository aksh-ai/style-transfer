import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
import PIL.Image as Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, models
from utils import *

parser = argparse.ArgumentParser(description='Perform Style Transfer')
parser.add_argument('--content', default=None, type=str, help='The path to the content image over which the style should be transferred')
parser.add_argument('--style', default=None, type=str, help='The path to the style image with which stye transfer should be done')
parser.add_argument('--size', default=512, type=int, help='Size for resizing the image, Ex: 512')
parser.add_argument('--steps', default=6000, type=int, help='Number of steps to optimize the target image with style')
parser.add_argument('--verbose', default=300, type=int, help='Interval to display the training results')
parser.add_argument('--save_vid', default=False, type=bool, help='Specify True or False to save the style transferred image')
parser.add_argument('--output_folder', default='./', type=str, help='Specify the output folder to save the image/video')
parser.add_argument('--output_name', default='output', type=str, help='Specify the name with which the image/video should be saved')

if __name__ == "__main__":
	opt = parser.parse_args()

	CONTENT = opt.content
	STYLE = opt.style
	SIZE = opt.size 
	STEPS = opt.steps 
	VERBOSE = opt.verbose
	SAVE_VID = opt.save_vid 
	OUTPUT_FOLDER = opt.output_folder
	OUTPUT_NAME = opt.output_name

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	vgg = models.vgg19(pretrained=True).features

	for param in vgg.parameters():
		param.requires_grad = False

	vgg = vgg.to(device)

	content = load_img(CONTENT, shape=(SIZE, SIZE)).to(device)
	style = load_img(STYLE, shape=(SIZE, SIZE)).to(device)

	target = content.clone().requires_grad_(True).to(device)

	# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

	# ax1.imshow(np_convert(content))
	# ax1.axis('off')

	# ax2.imshow(np_convert(style))
	# ax2.axis('off')

	# ax3.imshow(np_convert(target))
	# ax3.axis('off')

	# plt.show()

	content_features = get_features(content, vgg)
	style_features = get_features(style, vgg)

	style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

	style_weights = {'conv1_1': 1.0, 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

	content_weight = 1
	style_weight = 1e6 

	verbose = VERBOSE
	optimizer = optim.Adam([target], lr=0.003)
	num_steps = STEPS

	height, width, channels = np_convert(target).shape
	img_array = np.empty(shape=(VERBOSE, height, width, channels))
	cap_frame = num_steps/VERBOSE

	ctr = 0

	for i in range(1, num_steps+1):
		target_features = get_features(target, vgg)

		content_loss = torch.mean(torch.square(target_features['conv4_2'] - content_features['conv4_2']))

		style_loss = 0.0

		for layer in style_weights:
			target_feature = target_features[layer]
			target_gm = gram_matrix(target_feature)
			style_gm = style_grams[layer]
			layer_style_loss = style_weights[layer] * torch.mean(torch.square(target_gm - style_gm))
			_, d, h, w = target_feature.shape
			style_loss += layer_style_loss / (d * h * w)

		total_loss = content_loss * content_weight + style_loss * style_weight

		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()

		if i % verbose == 0:
			print(f'Step {i}, Total Loss: {total_loss.item()}')

			# plt.imshow(np_convert(target))
			# plt.axis('off')
			# plt.show()

		if i % cap_frame == 0:
			img_array[ctr] = np_convert(target)
			ctr += 1

	image = target
	img = image.detach().numpy()* 255
	img = img.astype(np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	cv2.imwrite(os.path.join(OUTPUT_FOLDER, OUTPUT_NAME + '.jpg'), img)

	if SAVE_VID:
		frame_height, frame_width, _ = np_convert(target).shape
		writer = cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, OUTPUT_NAME + '.mp4'), cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

		for i in range(0, 300):
			img = img_array[i] * 255
			img = img.astype(np.uint8)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			writer.write(img)

		writer.release()

	exit()
