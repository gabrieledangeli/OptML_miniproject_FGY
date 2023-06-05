import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
import os
import torch
from skimage.transform import resize


def initialize_weights(net):
    '''
    The initialization method used in the initialize_weights function is 
    known as "normal initialization" or "Gaussian initialization."
    It sets the initial weights of the neural network layers 
    from a normal distribution with a mean of 0 and a standard deviation of 0.02.
    This initialization is common in the majority of the papers related to GANs.
    '''

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
    '''
    The scale_images function resizes a list of images to a new 
    shape using nearest neighbor interpolation.

    images: A list of input images.
    new_shape: The desired shape of the output images.
    '''
    images_list = list()
    for image in images:
		# resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
		# store
        images_list.append(new_image)
        return np.asarray(images_list)


def generate_animation(path, epochs):
    '''
    Function that generates an animation (GIF) from a series of image files. 

    path: The base path where the image files are located.
    epochs: The number of epochs (or iterations) for which the image files exist.
    '''
    images = []
    for epoch in range(epochs):
        img_name = path + '_epoch%03d' % (epoch+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(G_losses, D_losses, path, model_name = ''):
    '''
    Function used to plot and save the losses of 
    the generator and discriminator during training.
    '''
    x = range(len(D_losses))

    plt.plot(x, G_losses, label='GeneratorLoss', color='r')
    plt.plot(x, D_losses, label='DiscriminatorLoss', color='b')

    plt.xlabel('Epoch')
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


def loss_plot_ACGD(losses, path, model_name=''):
    x = range(len(losses))

    plt.plot(x, losses, label='Loss')
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



def score_plot(path, model_name, score_name, scores):
    '''
    Function used to plot inception score during the training.
    '''
    x = range(len(scores))

    plt.plot(x, scores, label=score_name, )

    plt.xlabel('Epoch')
    plt.ylabel(score_name)

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, model_name + '_' + score_name + '.png')

    plt.savefig(path)

    plt.show()

    plt.close()


def save_images(images, size, image_path):
    '''
    This function serves as a wrapper around the imsave function
    
    images: The generated images to be saved.
    size: The desired size of the output image grid in terms of rows and columns.
    image_path: The file path where the image will be saved.
    '''
    return imsave(images, size, image_path)

def imsave(images, size, path):
    '''
    images: The generated images to be saved.
    size: The desired size of the output image grid in terms of rows and columns.
    path: The file path where the image will be saved.
    '''
    image = np.squeeze(merge(images, size))
    # image = image.astype(np.uint8)  
    return imageio.imwrite(path, image)

def merge(images, size):
    '''
    The function handles merging the images by creating an empty canvas (img) with 
    the appropriate dimensions based on the size of the images and the desired grid size.
    It then iterates over the images and places each image at the corresponding position on the canvas. 
    The resulting merged image is returned.

    images: The generated images to be merged and arranged in a grid.
    size: The desired size of the output image grid in terms of rows and columns.
    '''
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

def save_scores(score, save_dir, dataset, gan_type, IS=False, FID=False):
    if not IS and not FID:
        raise ValueError('Cannot save different scores except IS score and FID score.')
    if IS:
        path = os.path.join(save_dir, dataset, gan_type, 'Inception_Scores')
    elif FID:
        path = os.path.join(save_dir, dataset, gan_type, 'FID_Scores')

    return np.save(path, score)


def save_loss(loss, save_dir, dataset, gan_type, Generator=False, Discriminator=False):
    path = os.path.join(save_dir, dataset, gan_type)
    
    if Generator:
        path = os.path.join(path, 'Generator_')
    if Discriminator:
        path = os.path.join(path, 'Discriminator_')
    
    return np.save(path+'loss', loss)

#def IS(images):
    """
    input has shape Nimage N x 3 x H x W
    """
    """
    inception = InceptionScore(normalize=True)
    inception.update(images)
    IS=inception.compute()
    return IS
    """

#def FID(imagesR,imagesG):
    """
    imagesR: minibatch of real image of shape N x 3 x H x W
    imagesG: minibatch of generated image of shape N x 3 x H x W
    """
    """
    fid = FrechetInceptionDistance(feature=64,normalize=True)
    fid.update(imagesR, real=True)
    fid.update(imagesG, real=False)
    FID=fid.compute()
    return FID
    """


