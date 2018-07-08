import time

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gridcut

PW = 2
COST = 8
SOURCE = np.array([[COST, COST, 0, 0],
                   [0, COST, COST, 0],
                   [0, 0, COST, COST]], dtype=np.float32)
SINK = np.array([[0, 0, COST, COST],
                 [COST, 0, 0, COST],
                 [COST, COST, 0, 0]], dtype=np.float32)
FIG_SIZE = 4


def test_maxflow_2d_4c_simple(source=SOURCE, sink=SINK, pw=PW):
    edges = np.ones((source.shape), dtype=np.float32) * pw
    result = gridcut.maxflow_2D_4C(source.shape[1], source.shape[0],
                                 source.ravel(), sink.ravel(),
                                 edges.ravel(), edges.ravel(),
                                 edges.ravel(), edges.ravel(),
                                 n_threads=1, block_size=4)
    assert np.array_equal(result.reshape(source.shape),
                          np.array([[0, 0, 1, 1],
                                    [1, 0, 0, 1],
                                    [1, 1, 0, 0]]))

    result = gridcut.maxflow_2D_4C_potts(source.shape[1], source.shape[0],
                                       source.ravel(), sink.ravel(), pw,
                                       n_threads=2, block_size=1)
    assert np.array_equal(result.reshape(source.shape),
                          np.array([[0, 0, 1, 1],
                                    [1, 0, 0, 1],
                                    [1, 1, 0, 0]]))


def generate_image_unary_term_2cls(width=120, height=80):
    """  """
    annot = np.zeros((height, width))
    annot[int(0.3 * height):int(0.8 * height),
          int(0.2 * width):int(0.7 * width)] = 1
    noise = np.random.randn(height, width) - 0.5

    source = (0.5 + (1 - annot) * 2 + noise).astype(np.float32)
    sink = (0.5 + annot * 2 + noise).astype(np.float32)

    fig = plt.figure(figsize=(2 * FIG_SIZE, FIG_SIZE))
    plt.subplot(1, 2, 1)
    plt.imshow(source, cmap="gray", interpolation="nearest")
    plt.subplot(1, 2, 2)
    plt.imshow(sink, cmap="gray", interpolation="nearest")
    fig.tight_layout()
    fig.savefig('./images/2cls_grid_unary.png')
    plt.close(fig)

    return annot, source, sink


def generate_image_unary_term_3cls(width=120, height=80):
    annot = np.zeros((height, width))
    annot[:, int(0.4 * width)] = 2
    annot[int(0.3 * height):int(0.8 * height),
        int(0.2 * width):int(0.7 * width)] = 1
    img = annot + np.random.randn(100, 100)

    unary = np.tile(img[:, :, np.newaxis], [1, 1, 3])

    tmp = unary[:, :, 1] - 1
    tmp[annot == 0] *= -1
    unary[:, :, 1] = tmp
    unary[:, :, 2] = 2 - unary[:, :, 2]

    fig = plt.figure(figsize=(unary.shape[-1] * FIG_SIZE, FIG_SIZE))
    for i in range(unary.shape[-1]):
        plt.subplot(1, unary.shape[-1], i + 1)
        plt.imshow(unary[:, :, i], cmap="gray", interpolation="nearest")
    fig.tight_layout()
    fig.savefig('./images/3cls_grid_unary.png')
    plt.close(fig)

    return annot, unary


def save_results(img, segm, fig_name=''):
    fig = plt.figure(figsize=(2 * FIG_SIZE, FIG_SIZE))
    plt.subplot(1, 2, 1), plt.title('image')
    plt.imshow(img, interpolation="nearest")
    plt.subplot(1, 2, 2), plt.title('solved labeling')
    plt.imshow(segm, interpolation="nearest")
    fig.tight_layout()
    fig.savefig('./images/%s.png' % fig_name)
    plt.close(fig)


def test_maxflow_2d_image(width=120, height=80, pw=PW):
    """  """
    img, source, sink = generate_image_unary_term_2cls(width, height)

    t = time.time()
    labels = gridcut.maxflow_2D_4C_potts(img.shape[1], img.shape[0],
                                       source, sink, pw)
    print ('elapsed time for "maxflow_2D_4C_potts": %f' % (time.time() - t))

    save_results(img, labels.reshape(img.shape),
                 fig_name='2cls_grid_labels_4c')

    t = time.time()
    labels = gridcut.maxflow_2D_8C_potts(img.shape[1], img.shape[0],
                                       source, sink, pw)
    print ('elapsed time for "maxflow_2D_8C_potts": %f' % (time.time() - t))

    save_results(img, labels.reshape(img.shape),
                 fig_name='2cls_grid_labels_8c')


# if __name__ == '__main__':
#     test_maxflow_2d_4c_simple()
#     test_maxflow_2d_image()
