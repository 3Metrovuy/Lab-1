from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

model_path = Path(__file__).parent / 'ModelNet40/airplane/test/airplane_0627.off'


def read_off(path):
    with open(path) as f:
        header = f.readline().strip()
        if not header.startswith('OFF'):
            raise ValueError(f'{path} is not an OFF file')
        counts = header[3:] or f.readline()
        n_verts, n_faces, _ = map(int, counts.split())
        verts = [list(map(float, f.readline().split())) for _ in range(n_verts)]
        faces = [list(map(int, f.readline().split()[1:])) for _ in range(n_faces)]
    return np.array(verts), faces


def model_to_points(path):
    verts, faces = read_off(path)
    X = verts.T
    X -= X.mean(axis=1, keepdims=True)
    return X, faces


def rotate_xy(X, theta):
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]], dtype=float)
    X_rotated = X.copy()
    return A @ X_rotated, A


def rotate_yz(X, theta):
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]], dtype=float)
    X_rotated = X.copy()
    return A @ X_rotated, A


def rotate_xz(X, theta):
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[c, 0, -s],
                  [0, 1, 0],
                  [s, 0, c]], dtype=float)
    X_rotated = X.copy()
    return A @ X_rotated, A


def draw_model(ax, X, faces, title, color):
    V = X.T
    mesh = Poly3DCollection([V[face] for face in faces],
                            facecolor=color, alpha=0.3, edgecolor='k', linewidth=0.1)
    ax.add_collection3d(mesh)

    r = np.linalg.norm(X, axis=0).max()
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)


def show_rotation(X, X_rot, faces, A, title):
    print(f'{title}\n{A}\n')

    fig = plt.figure(figsize=(12, 6))
    draw_model(fig.add_subplot(1, 2, 1, projection='3d'), X, faces, 'Original', 'lightgray')
    draw_model(fig.add_subplot(1, 2, 2, projection='3d'), X_rot, faces,
               f'{title}\n{np.array2string(A, precision=2, suppress_small=True)}', 'tab:blue')
    plt.show()


if __name__ == '__main__':
    X, faces = model_to_points(model_path)
    print('Shape:', X.shape, 'faces:', len(faces))

    for rotate, plane in [(rotate_xy, 'xy'), (rotate_yz, 'yz'), (rotate_xz, 'xz')]:
        X_rot, A = rotate(X, np.pi / 4)
        show_rotation(X, X_rot, faces, A, f'Rotation 45° in the {plane} plane')

    X1, A1 = rotate_xy(X, np.pi / 6)
    X2, A2 = rotate_yz(X1, np.pi / 4)
    X3, A3 = rotate_xz(X2, np.pi / 3)
    A_total = A3 @ A2 @ A1
    show_rotation(X, X3, faces, A_total, 'xy 30° -> yz 45° -> xz 60°')
