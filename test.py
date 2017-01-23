import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gridcut


SOURCE = np.array([8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
SINK = np.array([0, 0, 0, 0, 0, 0, 0, 9, 0], dtype=np.float32)
PW = np.ones(9, dtype=np.float32) * 8.
# UP = np.array([5, 1, 0, 0, 0, 0, 0, 0, 0])
# DOWN = np.array([0, 0, 4,0, 0, 0, 0, 3, 0])
# LEFT = np.array([0, 0, 0, 0, 3, 0, 0, 0, 0])
# RIGHT = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0])


def test_solve_2d_4c_simple(source=SOURCE, sink=SINK, pw=PW):
    result = gridcut.solve_2D_4C(3, 3, source, sink, pw, pw, pw, pw,
                                 n_threads=2, block_size=4)
    assert np.array_equal(result, np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]))

    result = gridcut.solve_2D_4C_potts(3, 3, 8, source, sink,
                                       n_threads=2, block_size=1)
    # FIXME: this should be the same as above
    assert np.array_equal(result, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_solve_2d_4c_image(width=100, height=100):
    """  """
    img = np.random.randn(height, width)
    img[int(0.3 * height):int(0.8 * height),
        int(0.2 * width):int(0.7 * width)] += 2
    img -= 1

    unary = np.c_[img.reshape(img.size, 1), -img.reshape(img.size, 1)].copy()

    fig = plt.figure(figsize=(unary.shape[-1] * 4, 4))
    for i in range(unary.shape[-1]):
        plt.subplot(1, unary.shape[-1], i + 1)
        plt.imshow(unary[:, i].reshape((height, width)), cmap="gray",
                   interpolation="nearest")
    fig.tight_layout(), fig.savefig('./images/grid_unary.png')

    source = img.ravel().astype(np.float32)
    sink = -img.ravel().astype(np.float32)

    labels = gridcut.solve_2D_4C_potts(height, width, 8, source, sink,
                                       n_threads=2, block_size=1)

    fig = plt.figure(figsize=(2 * 4, 4))
    plt.subplot(1, 2, 1), plt.title('image')
    plt.imshow(img, interpolation="nearest")
    plt.subplot(1, 2, 2), plt.title('solved labeling')
    plt.imshow(labels.reshape(height, width), interpolation="nearest")
    fig.tight_layout(), fig.savefig('./images/grid_labels.png')


# if __name__ == '__main__':
#     test_solve_2d_4c_simple()
#     test_solve_2d_4c_image()
