import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt


mnist = {
    'train_images': "train-images-idx3-ubyte.gz",
    'train_labels': "train-labels-idx1-ubyte.gz",
    'test_images': "t10k-images-idx3-ubyte.gz",
    'test_labels': "t10k-labels-idx1-ubyte.gz",}

fashion_mnist = {
    'train_images': "train-images-idx3-ubyte.gz",
    'train_labels': "train-labels-idx1-ubyte.gz",
    'test_images': "t10k-images-idx3-ubyte.gz",
    'test_labels': "t10k-labels-idx1-ubyte.gz",}

extended_mnist = dict(
    balanced = {
        'train_images':'emnist-balanced-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-balanced-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-balanced-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-balanced-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-balanced-mapping.txt',},
    byclass = {
        'train_images':'emnist-byclass-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-byclass-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-byclass-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-byclass-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-byclass-mapping.txt',},
    bymerge = {
        'train_images':'emnist-bymerge-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-bymerge-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-bymerge-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-bymerge-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-bymerge-mapping.txt',},
    digits = {
        'train_images':'emnist-digits-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-digits-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-digits-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-digits-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-digits-mapping.txt',},
    letters = {
        'train_images':'emnist-letters-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-letters-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-letters-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-letters-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-letters-mapping.txt',},
    mnist = {
        'train_images':'emnist-mnist-train-images-idx3-ubyte.gz',
        'train_labels':'emnist-mnist-train-labels-idx1-ubyte.gz',
        'test_images':'emnist-mnist-test-images-idx3-ubyte.gz',
        'test_labels':'emnist-mnist-test-labels-idx1-ubyte.gz',
        'mapping':'emnist-mnist-mapping.txt',})


def getfile(file_url, target_dir):
    import requests
    from tqdm import tqdm

    file_name = os.path.basename(file_url)
    file_path = os.path.join(target_dir, file_name)
    file_size = int(requests.head(file_url).headers["content-length"])

    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024,
                ascii=True, desc=file_name, ncols=100)

    with requests.get(file_url, stream=True) as req, open(file_path, 'wb') as file:
        for chunk in req.iter_content(chunk_size=1024):
            data_size = file.write(chunk)
            pbar.update(data_size)
        pbar.close()


def extract(file_path, extract_path):
    import shutil

    print(">> Extracting", os.path.basename(file_path))
    shutil.unpack_archive(file_path, extract_path)
    pathlib.Path(file_path).unlink()
    print(">> Complete!")


def unzip(file_path, image=False):
    import gzip
    
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if image else 8)
    return data.reshape(-1, 28, 28) if image else data


def load_mnist(data_dir, download=False):
    """ http://yann.lecun.com/exdb/mnist/ """

    url = "http://yann.lecun.com/exdb/mnist/"

    train_images = mnist["train_images"]
    train_labels = mnist["train_labels"]
    test_images = mnist["test_images"]
    test_labels = mnist["test_labels"]
    filenames = [train_images, train_labels, test_images, test_labels]

    data_dir = os.path.abspath(data_dir)
    if not pathlib.Path(data_dir).exists() and download:
        for filename in filenames:
            getfile(os.path.join(url, filename), data_dir)

    x_train = unzip(os.path.join(data_dir, train_images), image=True)
    y_train = unzip(os.path.join(data_dir, train_labels), image=False)
    x_test = unzip(os.path.join(data_dir, test_images), image=True)
    y_test = unzip(os.path.join(data_dir, test_labels), image=False)
    
    class_names = [str(i) for i in range(10)]

    return (x_train, y_train), (x_test, y_test), class_names


def load_fashion_mnist(data_dir, download=False):
    """ https://github.com/zalandoresearch/fashion-mnist """

    url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

    train_images = fashion_mnist["train_images"]
    train_labels = fashion_mnist["train_labels"]
    test_images = fashion_mnist["test_images"]
    test_labels = fashion_mnist["test_labels"]
    filenames = [train_images, train_labels, test_images, test_labels]

    data_dir = os.path.abspath(data_dir)
    if not pathlib.Path(data_dir).exists() and download:
        for filename in filenames:
            getfile(os.path.join(url, filename), data_dir)

    x_train = unzip(os.path.join(data_dir, train_images), image=True)
    y_train = unzip(os.path.join(data_dir, train_labels), image=False)
    x_test = unzip(os.path.join(data_dir, test_images), image=True)
    y_test = unzip(os.path.join(data_dir, test_labels), image=False)
    
    class_names =  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return (x_train, y_train), (x_test, y_test), class_names


def load_extended_mnist(data_dir, split_name='mnist', download=False):
    """ https://www.nist.gov/itl/products-and-services/extended_mnist-dataset """

    url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/"
    filename = "gzip.zip"

    data_dir = os.path.abspath(data_dir)
    if not pathlib.Path(data_dir).exists() and download:
        getfile(os.path.join(url, filename), data_dir)
        extract(os.path.join(data_dir, filename), data_dir)

    train_images = extended_mnist[split_name]['train_images']
    train_labels = extended_mnist[split_name]['train_labels']
    test_images  = extended_mnist[split_name]['test_images']
    test_labels  = extended_mnist[split_name]['test_labels']

    x_train = unzip(os.path.join(data_dir, 'gzip', train_images), image=True)
    y_train = unzip(os.path.join(data_dir, 'gzip', train_labels), image=False)
    x_test = unzip(os.path.join(data_dir, 'gzip', test_images), image=True)
    y_test = unzip(os.path.join(data_dir, 'gzip', test_labels), image=False)

    path = os.path.join(data_dir, 'gzip', extended_mnist[split_name]['mapping'])
    class_names = [chr(int(cls)) for cls in np.genfromtxt(path)[:, 1]]

    if split_name == 'letters':
        y_train = y_train - 1
        y_test = y_test - 1

    return (x_train, y_train), (x_test, y_test), class_names


def show_images(images, labels=None, class_names=None, n_cols=5, width=12, rotation=False):
    n_rows = images.shape[0] // n_cols + (1 if images.shape[0] % n_cols else 0)
    height = width*n_rows/n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))
    for i, ax in enumerate(axes.flat):
        if i < images.shape[0]:
            img = np.flipud(np.rot90(images[i])) if rotation else images[i]
            ax.imshow(img, cmap='gray_r')            

            if labels is not None:
                if class_names is not None:
                    ax.set_title(class_names[labels[i]])
                else:
                    ax.set_title(labels[i])

        ax.set_axis_off()
    fig.tight_layout()

    if labels is None:
        fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def print_info(train_data, test_data, class_names, split_name=None):
    print(">>", split_name if split_name is not None else "classes")
    print(class_names)

    images, labels = train_data
    print("\n>> Train data")
    print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
    print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())

    images, labels = test_data
    print("\n>> Test data")
    print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
    print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())


if __name__ == "__main__":
    
    # Load mnist dataset
    data_dir = "../datasets/mnist"
    print(os.path.abspath(data_dir))
    
    train_data, test_data, class_names = load_mnist(data_dir, download=True)
    print_info(train_data, test_data, class_names)    

    # Load fashion_mnist dataset
    data_dir = "../datasets/fashion_mnist"
    print(os.path.abspath(data_dir))
    
    train_data, test_data, class_names = load_fashion_mnist(data_dir, download=True)
    print_info(train_data, test_data, class_names)    
    
    # Load mnist dataset
    data_dir = "../datasets/extended_mnist"
    print(os.path.abspath(data_dir))
    
    split_name = "bymerge"
    train_data, test_data, class_names = load_extended_mnist(data_dir, split_name, download=True)
    print_info(train_data, test_data, class_names, split_name=split_name)    