##############################################################################
# Imports

import glob
import inspect
import os
import math
from numbers import Number
from typing import Callable, List, Optional, Iterable, Union, Any
import warnings

from pygments import highlight  # type: ignore
from pygments.lexers import PythonLexer  # type: ignore
from pygments.formatters import HtmlFormatter  # type: ignore
from IPython.display import display, HTML  # type: ignore

from PIL import Image  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore

from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import LinearSegmentedColormap  # type: ignore
import matplotlib.pyplot as plt

import seaborn as sns  # type: ignore

sns.set()
warnings.simplefilter(action="ignore", category=FutureWarning)

##############################################################################
# Traitement d'images

black_red_cmap = LinearSegmentedColormap.from_list(
    "black_red_cmap", ["black", "red"])
black_green_cmap = LinearSegmentedColormap.from_list(
    "black_green_cmap", ["black", "green"]
)
black_blue_cmap = LinearSegmentedColormap.from_list(
    "black_blue_cmap", ["black", "blue"]
)


def load_images(datadir: str, pattern: str = "*.png") -> pd.Series:
    """
    Return all the images in `datadir` whose name match the pattern

    The images are returned as a Panda Series, with the image file
    names as indices.

    Example:

    This returns all png images in `dataset/` whose name starts
    with `a`:

        >>> images = load_images('dataset', 'a*.png')

    The names of the files serve as index:

        >>> images.index
    """
    paths = sorted(glob.glob(os.path.join(datadir, pattern)))
    images = [Image.open(path) for path in paths]
    names = [os.path.basename(path) for path in paths]
    return pd.Series(images, names)


def image_grid(
    images: Iterable[Image.Image], columns: int = 5, titles: Optional[Iterable] = None
) -> Figure:
    """
    Return a figure holding the images arranged in a grid

    Optionally the number of columns and/or image titles can be provided.

    Example:

         >>> image_grid(images)
         >>> image_grid(images, titles=[....])

    """
    rows = math.ceil(1.0 * len(images) / columns)
    fig = Figure(figsize=(10, 10.0 * rows / columns))
    if titles is None:
        titles = range(len(images))
    for k, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(rows, columns, k + 1)
        ax.imshow(img)
        ax.tick_params(axis="both", labelsize=0, length=0)
        ax.grid(b=False)
        ax.set_xlabel(title, labelpad=-4)
    return fig


def show_color_channels(img: Image.Image) -> Figure:
    """
    Return a figure displaying the image together with its red, green, and blue layers
    """
    black_red_cmap   = LinearSegmentedColormap.from_list('black_red_cmap',   ["black", "red"])
    black_green_cmap = LinearSegmentedColormap.from_list('black_green_cmap', ["black", "green"])
    black_blue_cmap  = LinearSegmentedColormap.from_list('black_blue_cmap',  ["black", "blue"])
    m = np.array(img)
    fig = Figure(figsize=(30,5));
    (subplot, subplotr, subplotg, subplotb) = fig.subplots(1, 4)  # Quatre zones de dessin
    # Dessin de l'image et de ses trois couches
    subplot.imshow(m)
    imgr = subplotr.imshow(m[:,:,0], cmap=black_red_cmap,   vmin=0, vmax=255)
    imgg = subplotg.imshow(m[:,:,1], cmap=black_green_cmap, vmin=0, vmax=255)
    imgb = subplotb.imshow(m[:,:,2], cmap=black_blue_cmap,  vmin=0, vmax=255)
    # Ajout des barres d'échelle de couleur aux images
    fig.colorbar(imgr, ax=subplotr);
    fig.colorbar(imgg, ax=subplotg);
    fig.colorbar(imgb, ax=subplotb);
    return fig


def color_histogram(img: Image.Image) -> Figure:
    """
    Return a histogram of the color channels of the image
    """
    M = np.array(img)
    n, p, m = M.shape
    MM = np.reshape(M, (n * p, m))
    if m == 4:  # Discard transparency channel if present
        MM = MM[:, 0:3]
    colors = ["red", "green", "blue"]
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot()
    ax.hist(MM, bins=10, density=True, histtype="bar", color=colors, label=colors)
    ax.set_xlabel("Pixel amplitude in each color channel")
    ax.set_ylabel("Pixel density")
    return fig


def foreground_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 150
) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    # Remplacer la ligne suivante par le code adéquat
    F = np.array(img)
    F = np.min(F[:,:,0:3], axis=2)
    F = F < theta
    return F


def transparent_background_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 150
) -> Image.Image:
    """Create a cropped image with transparent background."""
    F = foreground_filter(img, theta=theta)
    M = np.array(img)
    N = np.zeros([M.shape[0], M.shape[1], 4], dtype=M.dtype)
    N[:, :, :3] = M[:, :, :3]
    N[:, :, 3] = F * 255
    return Image.fromarray(N)


def transparent_background(img: Image.Image) -> Image.Image:
    """Sets the white background of an image to transparent"""
    data = img.getdata()  # Get a list of tuples
    newData = []
    for a in data:
        a = a[:3]  # Shorten to RGB
        if np.mean(np.array(a)) == 255:  # the background is white
            a = a + (0,)  # Put a transparent value in A channel (the fourth one)
        else:
            a = a + (255,)  # Put a non- transparent value in A channel
        newData.append(a)
    img.putdata(newData)  # Get new img ready
    return img

##############################################################################
# Attributs


def redness(img: Image.Image) -> float:
    """Return the redness of a PIL image."""
    # Remplacer la ligne suivante par le code adéquat
    m = np.array(img)
    G = m[:, :, 1]*1.0
    R = m[:, :, 0]*1.0
    F = foreground_filter(m)
    return np.mean(R[F]) - np.mean(G[F])


def elongation(img: Image.Image) -> float:
    """Extract the scalar value elongation from a PIL image."""
    F = foreground_filter(img)
    # Build the cloud of points given by the foreground image pixels
    xy = np.argwhere(F)
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)
    return s[0] / s[1]


def elongation_plot(img: Image.Image, subplot: Axes) -> None:
    """Plot the principal axes of the SVD when computing the elongation"""
    # Build the cloud of points defined by the foreground image pixels
    F = foreground_filter(img)
    xy = np.argwhere(F)
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)

    N = len(xy)
    a0 = s[0] / np.sqrt(N)
    a1 = s[1] / np.sqrt(N)

    # Plot the center
    subplot.plot(
        C[1], C[0], "ro", linewidth=50, markersize=10
    )  # x and y are j and i in matrix coord.
    # Plot the principal axes
    subplot.plot(
        [C[1], C[1] + a0 * V[0, 1]], [C[0], C[0] + a0 * V[0, 0]], "r-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] - a0 * V[0, 1]], [C[0], C[0] - a0 * V[0, 0]], "r-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] + a1 * V[1, 1]], [C[0], C[0] + a1 * V[1, 0]], "g-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] - a1 * V[1, 1]], [C[0], C[0] - a1 * V[1, 0]], "g-", linewidth=3
    )


##############################################################################
# Nouveaux attributs

# Remplacer la ligne suivante par le code adéquat
# Code non implanté


##############################################################################
# Analyse de performance

def error_rate(solutions: pd.Series, predictions: pd.Series) -> Any:
    """
    Return the error rate between two vectors.
    """
    return (solutions != predictions).mean()


def split_data(X, Y, verbose=True, seed=0):
    """Make a 50/50 training/test data split (stratified).
    Return the indices of the split train_idx and test_idx."""
    SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    ((train_index, test_index),) = SSS.split(X, Y)
    if verbose:
        print("TRAIN:", train_index, "TEST:", test_index)
    return (train_index, test_index)


def make_scatter_plot(
    df,
    images,
    train_index=[],
    test_index=[],
    filter=None,
    predicted_labels=[],
    show_diag=False,
    axis="normal",
    feat=None,
    theta=None,
) -> Figure:
    """This scatter plot function allows us to show the images.

    predicted_labels can either be:
                    - None (queries shown as question marks)
                    - a vector of +-1 predicted values
                    - the string "GroundTruth" (to display the test images).
    Other optional arguments:
            show_diag: add diagonal dashed line if True.
            feat and theta: add horizontal or vertical line at position theta
            axis: make axes identical if 'square'."""
    fruit = np.array(["B", "A"])

    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot()

    nsample, nfeat = df.shape
    if len(train_index) == 0:
        train_index = range(nsample)
    # Plot training examples
    x = df.iloc[train_index, 0]
    y = df.iloc[train_index, 1]
    f = images.iloc[train_index]
    ax.scatter(x, y, s=750, marker="o", c="w")

    for x0, y0, img in zip(x, y, f):
        ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # Plot test examples
    x = df.iloc[test_index, 0]
    y = df.iloc[test_index, 1]

    if len(predicted_labels) > 0 and not (predicted_labels == "GroundTruth"):
        label = (predicted_labels + 1) / 2
        ax.scatter(x, y, s=250, marker="s", color="c")
        for x0, y0, lbl in zip(x, y, label):
            ax.text(
                x0 - 0.03,
                y0 - 0.03,
                fruit[int(lbl)],
                color="w",
                fontsize=12,
                weight="bold",
            )
    elif predicted_labels == "GroundTruth":
        f = images.iloc[test_index]
        ax.scatter(x, y, s=500, marker="s", color="c")
        for x0, y0, img in zip(x, y, f):
            ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
            ax.add_artist(ab)
    else:  # Plot UNLABELED test examples
        f = images[test_index]
        ax.scatter(x, y, s=250, marker="s", c="c")
        ax.scatter(x, y, s=100, marker="$?$", c="w")

    if axis == "square":
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(f"$x_1$ = {df.columns[0]}")
    ax.set_ylabel(f"$x_2$ = {df.columns[1]}")

    # Add line on the diagonal
    if show_diag:
        ax.plot([-3, 3], [-3, 3], "k--")

    # Add separating line along one of the axes
    if theta is not None:
        if feat == 0:  # vertical line
            ax.plot([theta, theta], [-3, 3], "k--")
        else:  # horizontal line
            ax.plot([-3, 3], [theta, theta], "k--")

    return fig


##############################################################################
# Revue de code

def show_source(function: Callable) -> None:
    code = inspect.getsource(function)
    lexer = PythonLexer()
    formatter = HtmlFormatter(cssclass="pygments")
    html_code = highlight(code, lexer, formatter)
    css = formatter.get_style_defs(".pygments")
    html = f"<style>{css}</style>{html_code}"
    display(HTML(html))

    
    
#def boucle(img, start_pixel):
#    if(start_pixel == img.shape):
       #plan effectuee, fonction recursive qui par dun point (supposee blanc et qui ssaie de trouver une sortie)
       #en cours d'implementation
