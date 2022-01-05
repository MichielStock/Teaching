"""
Created on Saturday 21 October 2017
Last update: Sunday 22 October 2017

@author: Michiel Stock
michielfmstock@gmail.com

Color transfer using optimal transport.

Takes to color scheme of one image and transforms it to another image
"""

import numpy as np
from optimal_transport import OptimalTransport
from skimage import io
from skimage.color import rgb2hsv, hsv2rgb
from sklearn.neighbors import KNeighborsRegressor
import argparse

import warnings
warnings.simplefilter("ignore", UserWarning)



n_pixels = 1000
lam = 10
n_neighbors = 10
name_from = 'Figures/PB2PC.jpg'
name_to = 'Figures/PC.jpg'
name_out = 'test.jpg'

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-f', '--fr', type=str,default='Figures/PC.jpg',
                help='image to take the color from')
arg_parser.add_argument('-t', '--to', type=str,default='Figures/PB.jpg',
                help='image transform')
arg_parser.add_argument('-o', '--out', type=str,default='test.jpg',
                help='name of the output image')
arg_parser.add_argument('-n_pixels', type=int, default=1000,
                help='number of pixels to sample (default: 1000)')
arg_parser.add_argument('-lam', type=float, default=10,
                help='value for entropic regularization (default: 10)')
arg_parser.add_argument('-n_neighbors', type=int, default=10,
                help='number of neighbors in the KNN (default: 10)')
arg_parser.add_argument('-metric', type=str, default='mahalanobis',#mahalanobis
                help='distance metric used for cost matrix (default: Mahalanobis)')
arg_parser.add_argument('-save_color_distribution', type=bool, default=False,
                help='save plots of the to and from color distribition')
arg_parser.add_argument('--hsv', dest='use_hsv', default=False, const=True,
                action='store_const',
                help='process the image in hsv space (default: in RGB space)')
args = arg_parser.parse_args()

# get arguments
name_from = args.fr
name_to = args.to
name_out = args.out
n_pixels = args.n_pixels
lam = args.lam
n_neighbors = args.n_neighbors
distance_metric = args.metric
save_col_distribution = args.save_color_distribution

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)

def minmax(I):
    return np.clip(I, 0, 1)

def main():
    # read the images
    image_from = io.imread(name_from) / 256
    image_to = io.imread(name_to) / 256

    # change to hsv domain (if requested)
    if args.use_hsv:
        image_from[:] = rgb2hsv(image_from)
        image_to[:] = rgb2hsv(image_to)

    # get shapes
    shape_from = image_from.shape
    shape_to = image_to.shape

    # flatten
    X_from = im2mat(image_from)
    X_to = im2mat(image_to)

    # number of pixes
    n_pixels_from = X_from.shape[0]
    n_pixels_to = X_to.shape[0]

    # subsample
    X_from_ss = X_from[np.random.randint(0, n_pixels_from-1, n_pixels),:]
    X_to_ss = X_to[np.random.randint(0, n_pixels_to-1, n_pixels),:]

    if save_col_distribution:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('white')

        fig, axes = plt.subplots(nrows=2, figsize=(5, 10))
        for ax, X in zip(axes, [X_from_ss, X_to_ss]):
            ax.scatter(X[:,0], X[:,1], color=X)
            if args.use_hsv:
                ax.set_xlabel('hue')
                ax.set_ylabel('value')
            else:
                ax.set_xlabel('red')
                ax.set_ylabel('green')
        axes[0].set_title('distr. from')
        axes[1].set_title('distr. to')
        fig.tight_layout()
        fig.savefig('color_distributions.png')

    # optimal tranportation
    ot_color = OptimalTransport(X_to_ss, X_from_ss, lam=lam,
                                    distance_metric=distance_metric)#euclidean distance_metric

    # model transfer
    transfer_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    transfer_model.fit(X_to_ss, n_pixels * ot_color.P @ X_from_ss)
    X_transfered = transfer_model.predict(X_to)

    image_transferd = minmax(mat2im(X_transfered, shape_to))
    if args.use_hsv:
        image_transferd[:] = hsv2rgb(image_transferd)
    io.imsave(name_out, image_transferd)

if __name__ == '__main__':
    main()
