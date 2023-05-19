import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
import os
from skimage.transform import resize


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)


def generate_animation(path, epochs):
    images = []
    for epoch in range(epochs):
        img_name = path + '_epoch%03d' % (epoch+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(G_losses, D_losses, path, model_name = ''):
    x = range(len(D_losses))

    plt.plot(x, G_losses, label='GeneratorLoss', color='r')
    plt.plot(x, D_losses, label='DiscriminatorLoss', color='b')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(path):
        os.makedirs(path)
    
    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.show()

    plt.close()

def is_plot(inception_scores):
    x = range(len(inception_scores))

    plt.plot(x, inception_scores, label='InceptionScore', )

    plt.xlabel('Iteration')
    plt.ylabel('Inception Score')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    plt.close()


def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')