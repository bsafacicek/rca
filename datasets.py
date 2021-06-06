import numpy as np
from copy import deepcopy
from skimage.transform import downscale_local_mean
from google_drive_downloader import GoogleDriveDownloader as gdd
from scipy.io import loadmat
import cv2, os, glob, scipy

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch import FloatTensor
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import SVHN, MNIST, STL10
from torch.utils.data import DataLoader

DATA_DIR = "data/"


def resize_imgs(x, out_len):
    x_resized = np.zeros((len(x), out_len, out_len))
    for i in range(len(x)):
        x_resized[i,:,:] = scipy.misc.imresize(x[i,:,:], (out_len, out_len), interp='bilinear')
    return x_resized


def convert_svhn_images(svhn_images, small_H=28):
    big_H = svhn_images.shape[2]
    margin = (big_H-small_H)//2
    svhn_images_small = np.zeros((svhn_images.shape[0], 1, small_H, small_H))

    # convert RGB to gray
    rgb_values = [0.299, 0.587, 0.114]
    for j, svhn_image in enumerate(svhn_images):
        for i, rgb_value in enumerate(rgb_values):
            svhn_images_small[j] += svhn_image[i, margin:-margin, margin:-margin]*rgb_value

    return FloatTensor(svhn_images_small)


def convert_mnist_images(mnist_images, big_H=32): #  N, 1, 28, 28 -> N, 3, 32, 32
    small_H = mnist_images.size(2)
    mnist_images_big = np.zeros((mnist_images.size(0), 3, big_H, big_H), dtype=np.uint8)

    ### pad zeros to the mnist images:
    margin = (big_H-small_H)//2
    mnist_images_big[:,0,margin:-margin,margin:-margin] = mnist_images.squeeze()
    mnist_images_big[:,1,margin:-margin,margin:-margin] = mnist_images.squeeze()
    mnist_images_big[:,2,margin:-margin,margin:-margin] = mnist_images.squeeze()

    return mnist_images_big


def convert_mnist_images_torgb(mnist_images): #  N, 1, 28, 28 -> N, 3, 32, 32
    mnist_images_big = np.zeros((mnist_images.size(0), 3, 28, 28), dtype=np.uint8)

    ### pad zeros to the mnist images:
    mnist_images_big[:, 0, :, :] = mnist_images.squeeze()
    mnist_images_big[:, 1, :, :] = mnist_images.squeeze()
    mnist_images_big[:, 2, :, :] = mnist_images.squeeze()

    return mnist_images_big


def augment_mnist_rgb(is_crop=False, is_flip=False, brightness=0, contrast=0,
                      saturation=0, hue=0):
    transform_train = [transforms.ToTensor(),
        	       transforms.Normalize((0., 0., 0.), (1., 1., 1.))]
    if is_crop:
    	transform_train.insert(0, transforms.RandomCrop(32, padding=np.random.randint(5, size=1)[0]))
    if is_flip:
    	transform_train.insert(0, transforms.RandomHorizontalFlip())
    if brightness+contrast+saturation+hue > 0:
    	jitter = torchvision.transforms.ColorJitter(
    		brightness=brightness, contrast=contrast,
    		saturation=saturation, hue=hue)

    	transform_train.insert(0, jitter)

    transform_train = transforms.Compose(transform_train)
    return transform_train


def no_augment_mnist_rgb():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
    ])
    return transform_train


def no_augment_mnist_gray():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,)),])
    return transform_train


def get_loader_mnist_rgb(batchsize):
    transform_train = augment_mnist_rgb()

    trainset_m = MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset_m = MNIST(root='./data', train=False, download=True, transform=no_augment_mnist_rgb())
    # SVHN object accepts NUMPY:
    train_data = trainset_m.train_data.numpy()
    test_data = testset_m.test_data.numpy()
    train_labels = trainset_m.train_labels.numpy()
    test_labels = testset_m.test_labels.numpy()
    print("Original MNIST")
    print(train_data.shape, len(train_labels))
    print(test_data.shape, len(test_labels))

    ### use SVHN object to load MNIST RGB data
    trainset = SVHN(root='./data', split='train', download=True, transform=transform_train)
    testset = SVHN(root='./data', split='test', download=True, transform=no_augment_mnist_rgb())
    trainset.data = convert_mnist_images(trainset_m.train_data)
    testset.data = convert_mnist_images(testset_m.test_data)
    trainset.labels = trainset_m.train_labels
    testset.labels = testset_m.test_labels
    print("RGB MNIST")
    print(trainset.data.shape, len(trainset.labels))
    print(testset.data.shape, len(testset.labels))

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

    print("MNIST train min=%f, max=%f" % (trainset.data.min(), trainset.data.max()))
    print("MNIST test min=%f, max=%f" % (testset.data.min(), testset.data.max()))

    return trainloader, testloader


def get_loader_mnist_rgb_28(batchsize):
    transform_train = no_augment_mnist_rgb()

    trainset_m = MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset_m = MNIST(root='./data', train=False, download=True, transform=no_augment_mnist_rgb())
    # SVHN object accepts NUMPY:
    train_data = trainset_m.train_data.numpy()
    test_data = testset_m.test_data.numpy()
    train_labels = trainset_m.train_labels.numpy()
    test_labels = testset_m.test_labels.numpy()
    print("Original MNIST with size 28")
    print(train_data.shape, len(train_labels))
    print(test_data.shape, len(test_labels))

    ### use SVHN object to load MNIST RGB data
    trainset = SVHN(root='./data', split='train', download=True, transform=transform_train)
    testset = SVHN(root='./data', split='test', download=True, transform=no_augment_mnist_rgb())
    trainset.data = convert_mnist_images_torgb(trainset_m.train_data)
    testset.data = convert_mnist_images_torgb(testset_m.test_data)
    trainset.labels = trainset_m.train_labels
    testset.labels = testset_m.test_labels
    print("RGB MNIST with size 28")
    print(trainset.data.shape, len(trainset.labels))
    print(testset.data.shape, len(testset.labels))

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

    print("MNIST with size 28 train min=%f, max=%f" % (trainset.data.min(), trainset.data.max()))
    print("MNIST with size 28 test min=%f, max=%f" % (testset.data.min(), testset.data.max()))

    return trainloader, testloader


def get_loader_svhn_rgb(batchsize):
    transform_train = augment_mnist_rgb()

    trainset = SVHN(root='./data', split='train', download=True, transform=transform_train)
    testset = SVHN(root='./data', split='test', download=True, transform=no_augment_mnist_rgb())
    print (trainset.data.shape, len(trainset.labels))
    print (testset.data.shape, len(testset.labels))

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

    print("SVHN train min=%f, max=%f" % (trainset.data.min(), trainset.data.max()))
    print("SVHN test min=%f, max=%f" % (testset.data.min(), testset.data.max()))

    return trainloader, testloader


def get_loader_digit_rgb(batchsize):
    transform_train = no_augment_mnist_rgb()

    ########## download synth data from Ganin's Google Drive ####################
    gdd.download_file_from_google_drive(file_id='0B9Z4d7lAwbnTSVR1dEFSRUFxOUU', dest_path='data/SynthDigits.zip', unzip=True)

    folder_name = "data/"
    file_train = 'synth_train_32x32.mat'
    train_data = loadmat(folder_name+file_train)
    train_x = train_data["X"]
    train_x = np.rollaxis(train_x, 3, 0)
    train_x = np.rollaxis(train_x, 3, 1)
    train_y = train_data["y"]
    print(train_x.shape)
    print(train_y.shape)

    file_test = 'synth_test_32x32.mat'
    test_data = loadmat(folder_name+file_test)
    test_x = test_data["X"]
    test_x = np.rollaxis(test_x, 3, 0)
    test_x = np.rollaxis(test_x, 3, 1)
    test_y = test_data["y"]
    print(test_x.shape)
    print(test_y.shape)

    trainset = SVHN(root='./data', split='train', download=True, transform=transform_train)
    testset = SVHN(root='./data', split='test', download=True, transform=no_augment_mnist_rgb())

    trainset.data = train_x
    testset.data = test_x
    trainset.labels = train_y
    testset.labels = test_y

    print (trainset.data.shape, len(trainset.labels))
    print (testset.data.shape, len(testset.labels))

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

    print("synth train min=%f, max=%f" % (trainset.data.min(), trainset.data.max()))
    print("synth test min=%f, max=%f" % (testset.data.min(), testset.data.max()))

    return trainloader, testloader


def get_loader_mnist_gray(batchsize):
    transform_train = no_augment_mnist_rgb()
    trainset = MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = MNIST(root='./data', train=False, download=True, transform=no_augment_mnist_rgb())

    print(trainset.train_data.type(), testset.test_data.type(), trainset.train_labels.type() , testset.test_labels.type())

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

    print((trainset.train_data.size(), len(trainset.train_labels)))
    print((testset.test_data.size(), len(testset.test_labels)))
    print("MNIST train min=%f, max=%f" % (trainset.train_data.min(), trainset.train_data.max()))
    print("MNIST test min=%f, max=%f" % (testset.test_data.min(), testset.test_data.max()))

    return trainloader, testloader


class CIFAR10(torchvision.datasets.CIFAR10):
    def __len__(self):
        if self.train:
            return len(self.data)
        else:
            return len(self.data)


def noaug_cifar():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ])
    return transform_train


def get_loader_CIFAR(batchsize):
    transform_train = noaug_cifar()

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = CIFAR10(root='./data', train=False, download=True, transform=noaug_cifar())
    print (trainset.data.shape, len(trainset.targets))
    print (testset.data.shape, len(testset.targets))

    ### remove frog samples
    trainset.targets = np.array(trainset.targets)
    final_inds_train = np.where(trainset.targets != 6)[0]
    trainset.data = trainset.data[final_inds_train]
    trainset.targets = trainset.targets[final_inds_train]
    testset.targets = np.array(testset.targets)
    final_inds_test = np.where(testset.targets != 6)[0]
    testset.data = testset.data[final_inds_test]
    testset.targets = testset.targets[final_inds_test]
    print (trainset.data.shape, len(trainset.targets))
    print (testset.data.shape, len(testset.targets))

    ### shift label indexes
    labels_train = deepcopy(trainset.targets)
    trainset.targets[labels_train==7] = 6
    trainset.targets[labels_train==8] = 7
    trainset.targets[labels_train==9] = 8
    labels_test = deepcopy(testset.targets)
    testset.targets[labels_test==7] = 6
    testset.targets[labels_test==8] = 7
    testset.targets[labels_test==9] = 8

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

    print("CIFAR train min=%f, max=%f" % (trainset.data.min(), trainset.data.max()))
    print("CIFAR test min=%f, max=%f" % (testset.data.min(), testset.data.max()))

    return trainloader, testloader


def get_loader_STL(batchsize):
    transform_train = noaug_cifar()

    trainset = STL10(root='./data', split='train', download=True, transform=transform_train)
    testset = STL10(root='./data', split='test', download=True, transform=noaug_cifar())
    print (trainset.data.shape, len(trainset.labels))
    print (testset.data.shape, len(testset.labels))

    ### remove monkey samples
    trainset.labels = np.array(trainset.labels)
    final_inds_train = np.where(trainset.labels != 7)[0]
    trainset.data = trainset.data[final_inds_train]
    trainset.labels = trainset.labels[final_inds_train]
    testset.labels = np.array(testset.labels)
    final_inds_test = np.where(testset.labels != 7)[0]
    testset.data = testset.data[final_inds_test]
    testset.labels = testset.labels[final_inds_test]
    print (trainset.data.shape, len(trainset.labels))
    print (testset.data.shape, len(testset.labels))

    ### change label indexes to be the same as cifar10
    labels_train = deepcopy(trainset.labels)
    trainset.labels[labels_train==1] = 2
    trainset.labels[labels_train==2] = 1
    trainset.labels[labels_train==8] = 7
    trainset.labels[labels_train==9] = 8
    labels_test = deepcopy(testset.labels)
    testset.labels[labels_test==1] = 2
    testset.labels[labels_test==2] = 1
    testset.labels[labels_test==8] = 7
    testset.labels[labels_test==9] = 8

    ### resize images N X 9 6 X 96 X 3 -> N X 32 X 32 X 3:
    trainset.data = downscale_local_mean(trainset.data, (1, 1, 3, 3)).astype(np.uint8)
    testset.data = downscale_local_mean(testset.data, (1, 1, 3, 3)).astype(np.uint8)

    print (trainset.data.shape, len(trainset.labels))
    print (testset.data.shape, len(testset.labels))

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

    print("STL train min=%f, max=%f" % (trainset.data.min(), trainset.data.max()))
    print("STL test min=%f, max=%f" % (testset.data.min(), testset.data.max()))

    return trainloader, testloader


class BSDS500(Dataset):
    """
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/data.py
    or
    https://github.com/vingving/pytorch-domain-adaptation/blob/62c1bcffeb07e8cbceae6fbb3b3cc888041b8afa/data.py
    """
    def __init__(self):
        if not os.path.exists(DATA_DIR+"BSDS500/"):
            ### download if not downloaded already
            os.system('git clone https://github.com/BIDS/BSDS500.git %s' % (DATA_DIR+"BSDS500/"))
            print("BSDS500 is downloaded to %s."% (DATA_DIR))
        else:
            print("BSDS500 has already been downloaded to %s."% (DATA_DIR))

        image_folder = DATA_DIR + 'BSDS500/BSDS500/data/images'
        self.image_files = list(map(str, glob.glob('%s/*/*.jpg'%(image_folder)))) # list all test, train, val images

    def __getitem__(self, i):
        image = cv2.imread(self.image_files[i], cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor

    def __len__(self):
        return len(self.image_files)


class MNISTM(Dataset):

    def __init__(self, train, transform):
        super(MNISTM, self).__init__()
        self.mnist = datasets.MNIST('./data', train=train, download=True, transform=transform)
        self.bsds = BSDS500()
        self.rng = np.random.RandomState(1) # Fix RNG so the same images are used for blending

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = patch.float() / 255
        blend = torch.abs(patch - digit)
        return blend, label

    def _random_patch(self, image, size=(28, 28)):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-size[1])
        y = self.rng.randint(0, im_height-size[0])
        return image[:, y:y+size[0], x:x+size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)


def get_loader_mnistm_rgb_28(batchsize):
    trainset = MNISTM(train=False, transform=no_augment_mnist_gray())

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testloader = DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=0)

    return trainloader, testloader


def load_datasets(config):
    b_source = config['b_source']
    b_target = config['b_target']
    source_dataset = config['source_dataset']
    target_dataset = config['target_dataset']

    if source_dataset == 'MNIST':
        trainloader_source, testloader_source = get_loader_mnist_rgb(b_source)
    elif source_dataset == 'SVHN':
        trainloader_source, testloader_source = get_loader_svhn_rgb(b_source)
    elif source_dataset == 'CIFAR':
        trainloader_source, testloader_source = get_loader_CIFAR(b_source)
    elif source_dataset == 'STL':
        trainloader_source, testloader_source = get_loader_STL(b_source)
    elif source_dataset == 'MNIST28':
        trainloader_source, testloader_source = get_loader_mnist_rgb_28(b_source)
    elif source_dataset == 'MNISTM28':
        trainloader_source, testloader_source = get_loader_mnistm_rgb_28(b_source)
    elif source_dataset == 'DIGIT':
        trainloader_source, testloader_source = get_loader_digit_rgb(b_source)
    else:
        raise NotImplementedError

    if target_dataset == 'MNIST':
        trainloader_target, testloader_target = get_loader_mnist_rgb(b_target)
    elif target_dataset == 'SVHN':
        trainloader_target, testloader_target = get_loader_svhn_rgb(b_target)
    elif target_dataset == 'CIFAR':
        trainloader_target, testloader_target = get_loader_CIFAR(b_target)
    elif target_dataset == 'STL':
        trainloader_target, testloader_target = get_loader_STL(b_target)
    elif target_dataset == 'MNIST28':
        trainloader_target, testloader_target = get_loader_mnist_rgb_28(b_target)
    elif target_dataset == 'MNISTM28':
        trainloader_target, testloader_target = get_loader_mnistm_rgb_28(b_target)
    elif target_dataset == 'DIGIT':
        trainloader_target, testloader_target = get_loader_digit_rgb(b_target)

    return trainloader_source, testloader_source, trainloader_target, testloader_target
