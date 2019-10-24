# Numpy
import numpy as np
import pickle
import cPickle

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import progress_bar
import torch.backends.cudnn as cudnn

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
#exec(%matplotlib inline)
import matplotlib.pyplot as plt

# OS
import os
import argparse
import random
import math

from datetime import datetime
start=datetime.now()

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_train_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model_autoencoder():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder1, autoencoder.decoder1)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder

def create_model_resnet():
    resnet = ResNet34()
    #print(resnet)
    if torch.cuda.is_available():
        resnet = resnet.cuda()
        print("Model moved to GPU in order to speed up training.")
    return resnet

def create_model_net():
    net = Net()
    print(net)
    if torch.cuda.is_available():
        net = net.cuda()
        print("Model moved to GPU in order to speed up training.")
    return net


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 12, 2, stride=1, padding=1),            # [batch, 12, 16, 16], kernel=4, stride=2
            #nn.Dropout2d(p=0.02),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 2, stride=1, padding=1),   			# [batch, 24, 8, 8]
            #nn.Dropout2d(p=0.02),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 2, stride=1, padding=1),           # [batch, 48, 4, 4]
            nn.Dropout2d(p=0.02),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            #nn.MaxPool2d(2, stride=2),
			#nn.ReLU(),
            #nn.Conv2d(48, 96, 2, stride=1, padding=1),           # [batch, 96, 2, 2]
            #nn.BatchNorm2d(96),
            #nn.ReLU(),

        )
        self.decoder1 = nn.Sequential(
            #nn.ConvTranspose2d(96, 48, 2, stride=1, padding=1),  # [batch, 48, 4, 4]
            #nn.BatchNorm2d(48),            
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 2, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 2, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 2, stride=1, padding=1),   # [batch, 3, 32, 32]
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            #nn.Softmax(),
        )

    def forward(self, x):
        encoded = self.encoder1(x)
        #encoded = self.encoder(x)
        decoded = self.decoder1(encoded)
        return encoded, decoded

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        #self.softmax = nn.Softmax()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.softmax(out)
        return out

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

# simple network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(32, 6, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


""" #Use this function for python 3
def unpickle(file):
    #load the cifar-10 data

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
"""

def unpickle(file): #Use this function for python 2
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def splitDataset(category, percentage):
    """Split each category into test and train based on percentage"""
    new_category_ratio = int((len(category) * percentage))
    print("new_category_ratio: ", int(new_category_ratio))
    train_category = category[:new_category_ratio]
    test_category = category[new_category_ratio:]
    
    return train_category, test_category

def load_cifar_10_data(data_dir, negatives=False):
    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072
    encoding = 'utf-8' #used only for python 3 for pickle
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    cifar_all_data = None
    cifar_all_filenames = []
    cifar_all_labels = []

    new_train_labels = []
    new_test_labels = []


    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_all_data = cifar_train_data_dict[b'data']
        else:
            cifar_all_data = np.vstack((cifar_all_data, cifar_train_data_dict[b'data']))
        cifar_all_filenames += cifar_train_data_dict[b'filenames']
        cifar_all_labels += cifar_train_data_dict[b'labels']

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_all_data = np.vstack((cifar_all_data, cifar_test_data_dict[b'data']))
    cifar_all_filenames += cifar_test_data_dict[b'filenames']
    cifar_all_labels += cifar_test_data_dict[b'labels']

    cifar_all_filenames = np.array(cifar_all_filenames)
    cifar_all_labels = np.array(cifar_all_labels)

    print("All Data image: ", cifar_all_data.shape)
    print("All Data label: ", cifar_all_labels.shape)

    airplane = None
    automobile = None
    bird = None
    cat = None
    deer = None
    dog = None
    frog = None
    horse = None
    ship = None
    truck = None

    airplane_c = 0
    automobile_c = 0
    bird_c = 0
    cat_c = 0
    deer_c = 0
    dog_c = 0
    frog_c = 0
    horse_c = 0
    ship_c = 0
    truck_c = 0

    category_counter = [0] * 10
	
    print("Sorting dataset...")

    for i in range(len(cifar_all_data)):
        if (cifar_all_labels[i] == 0):
            if category_counter[cifar_all_labels[i]] == 0:
                airplane = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                airplane = np.vstack((airplane, cifar_all_data[i]))
                print("sorting airplane", cifar_all_data[i])
        elif (cifar_all_labels[i] == 1):
            if category_counter[cifar_all_labels[i]] == 0:
                automobile = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                automobile = np.vstack((automobile, cifar_all_data[i]))
                print("sorting automobile", cifar_all_data[i])
        elif (cifar_all_labels[i] == 2):
            if category_counter[cifar_all_labels[i]] == 0:
                bird = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                bird = np.vstack((bird, cifar_all_data[i]))
                print("sorting bird", cifar_all_data[i])
        elif (cifar_all_labels[i] == 3):
            if category_counter[cifar_all_labels[i]] == 0:
                cat = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                cat = np.vstack((cat, cifar_all_data[i]))
                print("sorting cat", cifar_all_data[i])
        elif (cifar_all_labels[i] == 4):
            if category_counter[cifar_all_labels[i]] == 0:
                deer = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                deer = np.vstack((deer, cifar_all_data[i]))
                print("sorting deer", cifar_all_data[i])
        elif (cifar_all_labels[i] == 5):
            if category_counter[cifar_all_labels[i]] == 0:
                dog = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                dog = np.vstack((dog, cifar_all_data[i]))
                print("sorting dog", cifar_all_data[i])
        elif (cifar_all_labels[i] == 6):
            if category_counter[cifar_all_labels[i]] == 0:
                frog = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                frog = np.vstack((frog, cifar_all_data[i]))
                print("sorting frog", cifar_all_data[i])
        elif (cifar_all_labels[i] == 7):
            if category_counter[cifar_all_labels[i]] == 0:
                horse = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                horse = np.vstack((horse, cifar_all_data[i]))
                print("sorting horse", cifar_all_data[i])
        elif (cifar_all_labels[i] == 8):
            if category_counter[cifar_all_labels[i]] == 0:
                ship = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                ship = np.vstack((ship, cifar_all_data[i]))
                print("sorting ship", cifar_all_data[i])
        elif (cifar_all_labels[i] == 9):
            if category_counter[cifar_all_labels[i]] == 0:
                truck = cifar_all_data[i]
                category_counter[cifar_all_labels[i]] += 1
            else:
                truck = np.vstack((truck, cifar_all_data[i]))
                print("sorting truck", cifar_all_data[i])

    print("Sorting dataset completed!...")
    
    print("")
    print("airplane: ", airplane.shape)
    print("automobile: ", automobile.shape)
    print("bird: ", bird.shape)#50%
    print("cat: ", cat.shape)
    print("deer: ", deer.shape)#50%
    print("dog: ", dog.shape)
    print("frog: ", frog.shape)
    print("horse: ", horse.shape)
    print("ship: ", ship.shape)
    print("truck: ", truck.shape)#50%

    train_airplane, test_airplane = splitDataset(airplane, 0.9)
    train_automobile, test_automobile = splitDataset(automobile, 0.9)
    train_bird, test_bird = splitDataset(bird, 0.5)
    train_cat, test_cat = splitDataset(cat, 0.9)
    train_deer, test_deer = splitDataset(deer, 0.5)
    train_dog, test_dog = splitDataset(dog, 0.9)
    train_frog, test_frog = splitDataset(frog, 0.9)
    train_horse, test_horse = splitDataset(horse, 0.9)
    train_ship, test_ship = splitDataset(ship, 0.9)
    train_truck, test_truck = splitDataset(truck, 0.5)

    print(" ")
    print("New train dataset ratio: ")
	
    print("airplane: ", train_airplane.shape)
    print("automobile: ", train_automobile.shape)
    print("bird: ", train_bird.shape)#50%
    print("cat: ", train_cat.shape)
    print("deer: ", train_deer.shape)#50%
    print("dog: ", train_dog.shape)
    print("frog: ", train_frog.shape)
    print("horse: ", train_horse.shape)
    print("ship: ", train_ship.shape)
    print("truck: ", train_truck.shape)#50%

    total_train_data = train_airplane
    total_train_data = np.vstack((total_train_data, train_automobile))
    total_train_data = np.vstack((total_train_data, train_bird))
    total_train_data = np.vstack((total_train_data, train_cat))
    total_train_data = np.vstack((total_train_data, train_deer))
    total_train_data = np.vstack((total_train_data, train_dog))
    total_train_data = np.vstack((total_train_data, train_frog))
    total_train_data = np.vstack((total_train_data, train_horse))
    total_train_data = np.vstack((total_train_data, train_ship))
    total_train_data = np.vstack((total_train_data, train_truck))

    print(" ")
    print("New test dataset ratio: ")
	
    print("airplane: ", test_airplane.shape)
    print("automobile: ", test_automobile.shape)
    print("bird: ", test_bird.shape)#50%
    print("cat: ", test_cat.shape)
    print("deer: ", test_deer.shape)#50%
    print("dog: ", test_dog.shape)
    print("frog: ", test_frog.shape)
    print("horse: ", test_horse.shape)
    print("ship: ", test_ship.shape)
    print("truck: ", test_truck.shape)#50%
    print(" ")

    total_test_data = test_airplane
    total_test_data = np.vstack((total_test_data, test_automobile))
    total_test_data = np.vstack((total_test_data, test_bird))
    total_test_data = np.vstack((total_test_data, test_cat))
    total_test_data = np.vstack((total_test_data, test_deer))
    total_test_data = np.vstack((total_test_data, test_dog))
    total_test_data = np.vstack((total_test_data, test_frog))
    total_test_data = np.vstack((total_test_data, test_horse))
    total_test_data = np.vstack((total_test_data, test_ship))
    total_test_data = np.vstack((total_test_data, test_truck))

    train_airplane_labels = [0] * len(train_airplane)
    train_automobile_labels = [1] * len(train_automobile)
    train_bird_labels = [2] * len(train_bird)
    train_cat_labels = [3] * len(train_cat)
    train_deer_labels = [4] * len(train_deer)
    train_dog_labels = [5] * len(train_dog)
    train_frog_labels = [6] * len(train_frog)
    train_horse_labels = [7] * len(train_horse)
    train_ship_labels = [8] * len(train_ship)
    train_truck_labels = [9] * len(train_truck)

    new_train_labels = train_airplane_labels + train_automobile_labels + train_bird_labels + train_cat_labels + train_deer_labels + train_dog_labels + train_frog_labels + train_horse_labels + train_ship_labels + train_truck_labels

    test_airplane_labels = [0] * len(test_airplane)
    test_automobile_labels = [1] * len(test_automobile)
    test_bird_labels = [2] * len(test_bird)
    test_cat_labels = [3] * len(test_cat)
    test_deer_labels = [4] * len(test_deer)
    test_dog_labels = [5] * len(test_dog)
    test_frog_labels = [6] * len(test_frog)
    test_horse_labels = [7] * len(test_horse)
    test_ship_labels = [8] * len(test_ship)
    test_truck_labels = [9] * len(test_truck)

    new_test_labels = test_airplane_labels + test_automobile_labels + test_bird_labels + test_cat_labels + test_deer_labels + test_dog_labels + test_frog_labels + test_horse_labels + test_ship_labels + test_truck_labels

    total_train_data = total_train_data.reshape((len(total_train_data), 32, 3, 32))
    total_test_data = total_test_data.reshape((len(total_test_data), 32, 3, 32))
    if negatives:
        total_train_data = total_train_data.transpose(0, 2, 3, 1).astype(np.float32)
        total_test_data = total_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        total_train_data = np.rollaxis(total_train_data, 1, 4)
        total_test_data = np.rollaxis(total_test_data, 1, 4)

    new_train_labels = np.array(new_train_labels)
    new_test_labels = np.array(new_test_labels)

    perm_train = np.random.permutation(len(total_train_data))
    sorted_train_data = total_train_data[perm_train]
    sorted_train_labels = new_train_labels[perm_train]

    perm_test = np.random.permutation(len(total_test_data))
    sorted_test_data = total_test_data[perm_test]
    sorted_test_labels = new_test_labels[perm_test]

    print("")
    print("Total training dataset: ", len(total_train_data))
    print("Total training labels: ", str(len(new_train_labels)))
    print("Total test dataset: ", len(total_test_data))
    print("Total test labels: ", str(len(new_test_labels)))

    return sorted_train_data, sorted_train_labels, sorted_test_data, sorted_test_labels

def split_batch(labels, batchSize, counter):
    batch = math.floor(len(labels)/batchSize) # 182
    #labels[counter:batchSize*counter]
    if counter == 0:
        return labels[0:batchSize]
    elif counter > 0 and counter < batch:
        return labels[batchSize * counter:batchSize * (counter+1)]
    elif counter == batch:
        return labels[batchSize*counter:]

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    #Hyperparameter
    batchSize = 256
    epoch_No = 100

    # Create model
    autoencoder = create_model_autoencoder()
    resnet = create_model_resnet()
    #resnet = create_model_net()
    autoencoder = autoencoder.to(device)
    resnet = resnet.to(device)
    if device == 'cuda':
       resnet = torch.nn.DataParallel(resnet)
       autoencoder = torch.nn.DataParallel(autoencoder)
       cudnn.benchmark = True

    """
    # Load dataset -- Original/Default
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                             shuffle=False, num_workers=2)
    """

    #Load Dataset -- bird(50%), deer(50%), truck(50%), rest(90%)
    cifar_10_dir = './data/cifar-10-batches-py'
    trainset, trainLabel, testset, testLabel = load_cifar_10_data(cifar_10_dir)

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    # Define an optimizer and criterion for autoencoder
    criterion = nn.MSELoss()
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    #, lr=1e-3, weight_decay=1e-5
	
    # Define an optimizer and criterion for resnet
    criterion_resnet = nn.CrossEntropyLoss()
    #criterion_resnet = nn.CategoricalCrossEntropyLoss()
    #optimizer_resnet = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer_resnet = optim.Adam(resnet.parameters())  

    for epoch in range(epoch_No):
        global best_train_acc
        running_loss = 0.0
        train_loss = 0
        correct = 0
        total = 0
        print("Original Train input Shape: ", trainset.shape)
        print("Oriignal Train label Shape: ", trainLabel.shape)
        iterations_train =  int((math.floor(len(trainLabel)/batchSize))) + 1

        #--------------------Training section--------------------

        for i in range(0, iterations_train):#i, (inputTemp, labelTemp) in enumerate(trainloader):
            inputs = torch.tensor(split_batch(trainset, batchSize, i), dtype=torch.float)
            labels = torch.tensor(split_batch(trainLabel, batchSize, i), dtype=torch.long)
            inputs, labels = inputs.to(device), labels.to(device)

			#----------------Autoencoder----------------
            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            #----------------RESNET---------------------
            #============= Forward ============
            output_resnet = resnet(encoded)
            #print('output.shape: ', output_resnet.shape)
            loss_resnet = criterion_resnet(output_resnet, labels)
            # ============ Backward ============
            optimizer_resnet.zero_grad()
            loss_resnet.backward(retain_graph=True)
            optimizer_resnet.step()

            train_loss += loss_resnet.item()
            _, predicted = output_resnet.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            print('Epoch: %d | (%d/%d) | Loss: %.3f | Training Acc: %.3f%% (%d/%d) | Best Test Accuracy: %.3f' % (epoch+1, i+1, iterations_train, train_loss/(i+1), 100.*correct/total, correct, total, best_acc))
            #progress_bar(i, iterations_train, 'Loss: %.3f | Training Acc: %.3f%% (%d/%d) | Best Test Accuracy: %.3f' % (train_loss/(i+1), 100.*correct/total, correct, total, best_acc))

        global best_acc
        #net.eval()
        test_loss = 0
        test_loss_autoencoder = 0
        test_correct = 0
        test_total = 0
        iterations_test =  int((math.floor(len(testLabel)/batchSize))) + 1

        #--------------------Testing section--------------------

        with torch.no_grad():
            for p in range(0, iterations_test):#batch_idx, (inputs, test_labels) in enumerate(testloader):

                test_inputs = torch.tensor(split_batch(testset, batchSize, p), dtype=torch.float)
                test_labels = torch.tensor(split_batch(testLabel, batchSize, p), dtype=torch.long)
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

                #----------------Autoencoder----------------
                # ============ Forward ============
                test_encoded, test_outputs = autoencoder(test_inputs)
                test_loss_AE = criterion(test_outputs, test_inputs)
                test_loss_autoencoder += test_loss_AE.item()
                # ============ Backward ============
                #optimizer.zero_grad()
                #test_loss_AE.backward(retain_graph=True)
                #optimizer.step()

                test_outputs_resnet = resnet(test_encoded)
                print('Outputs_resnet', test_outputs_resnet.shape)
                test_loss_resnet = criterion_resnet(test_outputs_resnet, test_labels)

                test_loss += test_loss_resnet.item()
                _, predicted = test_outputs_resnet.max(1)
                test_total += test_labels.size(0)
                test_correct += predicted.eq(test_labels).sum().item()
   
                print('Epoch: %d | (%d/%d) | Loss Resnet: %.3f | Loss AE: %.3f | Test Acc: %.3f%% (%d/%d) | Best Test Acc: %.3f'
                    % (epoch+1, p+1, iterations_test, test_loss/(p+1), test_loss_autoencoder/(p+1),100.*test_correct/test_total, test_correct, test_total, best_acc))
                
                #progress_bar(p, iterations_test, 'Loss Resnet: %.3f | Loss AE: %.3f | Acc: %.3f%% (%d/%d) | Best Accuracy: %.3f'
                #    % (test_loss/(p+1), test_loss_autoencoder/(p+1),100.*test_correct/test_total, test_correct, test_total, best_acc))

        #---------------------------Save checkpoint----------------------------
        train_acc = 100.*correct/total
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            print('Best Training Epoch: ', epoch)
            print('Best Training Accuracy: ', best_train_acc)
            file1 = open("Best_train_result_datasetB.txt","w")
            file1.write("Best Training Epoch: " + str(epoch))
            file1.write("\n")
            file1.write("Best Training Accuracy: " + str(best_train_acc))
            file1.close()

        acc = 100.*test_correct/test_total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': resnet.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint_datasetB'):
                os.mkdir('checkpoint_datasetB')
            torch.save(state, './checkpoint_datasetB/ckpt.pth')
            best_acc = acc
            print('Best Test Epoch: ', epoch)
            print('Best Test Accuracy: ', best_acc)
            file1 = open("Best_test_result_datasetB.txt","w")
            file1.write("Best Testing Epoch: " + str(epoch) + "/n")
            file1.write("Best Test Accuracy: " + str(best_acc) + "/n")
            file1.close()

    print('Training completed!')
    print('Saving Model...')
    if not os.path.exists('./weights_datasetB'):
        os.mkdir('./weights_datasetB')
    torch.save(autoencoder.state_dict(), "./weights_datasetB/autoencoder.pkl")

    #----------------------Print category wise----------------------
    correct_category = list(0. for i in range(10))
    total_category = list(0. for i in range(10))
    iterations_Category =  int((math.floor(len(testLabel)/batchSize))) + 1
    with torch.no_grad():
        for x in range(0, iterations_Category):
            inputs_category = torch.tensor(split_batch(testset, batchSize, x), dtype=torch.float)
            labels_category = torch.tensor(split_batch(testLabel, batchSize, x), dtype=torch.long)
            inputs_category, labels_category = inputs_category.to(device), labels_category.to(device)

            test_encoded_category, test_outputs_category = autoencoder(inputs_category)
            test_outputs_resnet_category = resnet(test_encoded_category)
            _, predicted = torch.max(test_outputs_resnet_category, 1)
            c = (predicted == labels_category).squeeze()
            for i in range(4):
                labels_category_temp = labels_category[i]
                correct_category[labels_category_temp] += c[i].item()
                total_category[labels_category_temp] += 1

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * correct_category[i] / total_category[i]))

    print("Elasped: " + datetime.now()-start)


if __name__ == '__main__':
    main()
