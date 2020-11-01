import numpy as np
import torch
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms


def random_split_train_val(X, y, num_val, seed=42):
    '''
    Randomly splits dataset into training and validation
    
    Arguments:
    X - np array with samples
    y - np array with labels
    num_val - number of samples to put in validation
    seed - random seed

    Returns:
    train_X, np array (num_train, 32, 32, 3) - training images
    train_y, np array of int (num_train) - training labels
    val_X, np array (num_val, 32, 32, 3) - validation images
    val_y, np array of int (num_val) - validation labels
    '''
    np.random.seed(seed)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:-num_val]
    train_X = X[train_indices]
    train_y = y[train_indices]

    val_indices = indices[-num_val:]
    val_X = X[val_indices]
    val_y = y[val_indices]

    return train_X, train_y, val_X, val_y

def count_mean_std(data_folder):
    '''
    Функция подсчитывает усредненные значения mean и std по всему датасету и возвращает их
    '''
    data = dset.ImageFolder(
        root=data_folder,
        transform = transforms.ToTensor())
    
    dataiter = iter(data)
    
    count = 0
    for i in dataiter:
        features, _ = i
        if not count:
            mean_sum = features.mean(axis=(1,2))
            std_sum = features.std(axis=(1,2))
        else:
            mean_sum += features.mean(axis=(1,2))
            std_sum += features.std(axis=(1,2))
        count += 1

    mean_value = mean_sum / count
    std_value = std_sum / count
    
    return (mean_value, std_value)

def load_dataset(data_folder, batch_size):
    '''
    Функция импортирует данные из указанной директории и возвращает DataLoader объекты
    '''
    mean_value = [0.6007, 0.5609, 0.6516]
    std_value = [0.0821, 0.0864, 0.0670]
    
    if not (mean_value and std_value):
        mean_value, std_value = count_mean_std(data_folder)
    
    data = dset.ImageFolder(
        root=data_folder,
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_value, std=std_value)
        ])
    )
    
    test_split = 0.2
    val_split = 0.2
    test_split_ind = int(np.floor(test_split * len(data.imgs)))
    val_split_ind = test_split_ind + int(np.floor(val_split * (len(data.imgs) - test_split_ind)))
    indices = list(range(len(data.imgs)))
    np.random.shuffle(indices)
    
    test_indices, val_indices, train_indices = indices[:test_split_ind], indices[test_split_ind:val_split_ind], indices[val_split_ind:]                  
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    #change batch_size for val and test
    val_test_b_size = int(batch_size / 10)
    
    train_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(data, batch_size = val_test_b_size, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(data, batch_size = val_test_b_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader
    
