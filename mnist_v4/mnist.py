import os
import pathlib


def download_data(url, data_path):
    import requests
    from tqdm import tqdm

    files_size = int(requests.head(url).headers["content-length"])
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(data_path, os.path.basename(url))

    pbar = tqdm(total=files_size, unit='B', unit_scale=True, unit_divisor=1024,
                ascii=True, desc=os.path.basename(url), ncols=100)

    with requests.get(url, stream=True) as req, open(file_path, 'wb') as file:
        for chunk in req.iter_content(chunk_size=1024):
            data_size = file.write(chunk)
            pbar.update(data_size)
        pbar.close()
        
        
def load(file_path, image=False):
    import gzip
    import numpy as np
    
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if image else 8)
    return data.reshape(-1, 28, 28) if image else data


def load_mnist(data_path, download=False):
    """ The MNIST dataset: http://yann.lecun.com/exdb/mnist/ """

    url = "http://yann.lecun.com/exdb/mnist/"

    train_images = "train-images-idx3-ubyte.gz"
    train_labels = "train-labels-idx1-ubyte.gz"
    test_images = "t10k-images-idx3-ubyte.gz"
    test_labels = "t10k-labels-idx1-ubyte.gz"
    filenames = [train_images, train_labels, test_images, test_labels]

    data_path = os.path.abspath(data_path)
    if not pathlib.Path(data_path).exists() and download:
        for filename in filenames:
            download_data(os.path.join(url, filename), data_path)

    x_train = load(os.path.join(data_path, train_images), image=True)
    y_train = load(os.path.join(data_path, train_labels), image=False)
    x_test = load(os.path.join(data_path, test_images), image=True)
    y_test = load(os.path.join(data_path, test_labels), image=False)
    
    class_names = [str(i) for i in range(10)]

    return (x_train, y_train), (x_test, y_test), class_names


def load_fashion_mnist(data_path, download=False):
    """ https://github.com/zalandoresearch/fashion-mnist """

    url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

    train_images = "train-images-idx3-ubyte.gz"
    train_labels = "train-labels-idx1-ubyte.gz"
    test_images = "t10k-images-idx3-ubyte.gz"
    test_labels = "t10k-labels-idx1-ubyte.gz"
    filenames = [train_images, train_labels, test_images, test_labels]

    data_path = os.path.abspath(data_path)
    if not pathlib.Path(data_path).exists() and download:
        for filename in filenames:
            download_data(os.path.join(url, filename), data_path)

    x_train = load(os.path.join(data_path, train_images), image=True)
    y_train = load(os.path.join(data_path, train_labels), image=False)
    x_test = load(os.path.join(data_path, test_images), image=True)
    y_test = load(os.path.join(data_path, test_labels), image=False)
    
    class_names =  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return (x_train, y_train), (x_test, y_test), class_names


if __name__ == "__main__":
    
    data_path = '../../datasets/cifar10'
    print(os.path.abspath(data_path))

    train_data, test_data, class_names = load_mnist(data_path, download=True)

    images, labels = train_data
    print("images: ", type(images), images.shape, images.dtype, images.min(), images.max())
    print("labels: ", type(labels), labels.shape, labels.dtype, labels.min(), labels.max())
    print("classes:", class_names, '\n')