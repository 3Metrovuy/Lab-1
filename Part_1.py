from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

im_path = Path(__file__).parent / 'IMG_20260611_220645_783.jpg'

def image_to_points(path, threshold=100, step=50):
    im = plt.imread(path)
    gray = im.mean(axis=2)
    mask = gray < threshold
    rows, cols = np.nonzero(mask)
    rows = rows[::step]
    cols = cols[::step]

    X = np.vstack([cols, -rows]).astype(float)
    X -= X.mean(axis=1, keepdims=True)

    return X


def plot_points(X):
    plt.scatter(X[0], X[1], s=1, c='black')
    plt.axis('equal')  
    plt.axhline(0, color='gray', linewidth=0.5)  
    plt.axvline(0, color='gray', linewidth=0.5)  
    plt.title('Original')
    plt.show()


def stretch(X, a, b):
    A = np.array([[a, 0],
                  [0, b]], dtype=float)
    X_strethced = X.copy()
    return A @ X_strethced, A


def shear(X, a, b):
    A = np.array([[1, a],
                  [b, 1]], dtype=float)
    X_sheared = X.copy()
    return A @ X_sheared, A


def reflection(X, a, b):
    if a == 0 and b == 0:
        raise ValueError('(a, b) = (0, 0) does not define a line')
    A = np.array([[a**2 - b**2, 2*a*b],
                  [2*a*b, b**2 - a**2]], dtype=float) / (a**2 + b**2)
    X_reflected = X.copy()
    return A @ X_reflected, A


def rotation(X, theta):
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[c, -s],
                  [s, c]], dtype=float)
    X_rotated = X.copy()
    return A @ X_rotated, A


def show_transformation(X, X_trans, A, title):
    print(f'{title}\n{A}\n')

    plt.scatter(X[0], X[1], s=1, c='lightgray', label='original')
    plt.scatter(X_trans[0], X_trans[1], s=1, c='tab:blue', label='transformed') # e 
    plt.axis('equal')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)

    plt.title(f'{title}\n{np.array2string(A, precision=2)}')
    plt.legend(markerscale=8)  #!
    plt.show()


def apply_in_order(X, steps):
    A_total = np.eye(2)
    for func, *params in steps:
        X, A = func(X, *params)
        A_total = A @ A_total
    return X, A_total


def show_orders(X, results):
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5), sharex=True, sharey=True)
    for ax, (title, X_trans, A) in zip(axes, results):
        print(f'{title}\n{A}\n')
        ax.scatter(X[0], X[1], s=1, c='lightgray')
        ax.scatter(X_trans[0], X_trans[1], s=1, c='tab:blue')
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_title(f'{title}\n{np.array2string(A, precision=2)}')
    plt.show()


if __name__ == '__main__':
    X = image_to_points(im_path)
    print('Shape:', X.shape) 
    plot_points(X)

    X_s, A_s = stretch(X, 1.5, 0.7)
    show_transformation(X, X_s, A_s, 'Stretch (1.5, 0.7)')

    X_sh, A_sh = shear(X, 0.5, 0)
    show_transformation(X, X_sh, A_sh, 'Shear (0.5, 0)')

    X_r, A_r = reflection(X, 1, 1)
    show_transformation(X, X_r, A_r, 'Reflection about y = x')

    X_rot, A_rot = rotation(X, np.pi / 4)
    show_transformation(X, X_rot, A_rot, 'Rotation 45°')

    steps = {
        'stretch': (stretch, 1.5, 0.7),
        'shear': (shear, 0.5, 0),
        'rotation': (rotation, np.pi / 4),
    }
    orders = [
        ('stretch', 'shear', 'rotation'),
        ('rotation', 'shear', 'stretch'),
        ('shear', 'rotation', 'stretch'),
    ]
    results = []
    for order in orders:
        X_final, A_total = apply_in_order(X, [steps[name] for name in order])
        results.append((' -> '.join(order), X_final, A_total))
    show_orders(X, results)
