# Numpy
import numpy as np

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
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            #nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
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

        self.encoder2 = nn.Sequential(
            nn.Conv2d(48, 96, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            #nn.Dropout2d(p=0.02),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),   			# [batch, 24, 8, 8]
            #nn.Dropout2d(p=0.02),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 1000, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.Dropout2d(p=0.02),
            nn.BatchNorm2d(1000),
            nn.ReLU(),
            #nn.MaxPool2d(2, stride=2),
			#nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            #nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            #nn.ReLU(),
            nn.ConvTranspose2d(1000, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
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

def test(epoch, ):
    global best_acc
    #net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Accuracy: %d'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        print('Best Epoch: ', epoch)
        print('Best Accuracy: ', best_acc)
        file1 = open("Best_result.txt","w")
        file1.write("Best Epoch: " + epoch + "\n") 
        file1.write("Best Accuracy: " + best_acc + "\n")
        file1.close() 


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
    autoencoder = autoencoder.to(device)
    resnet = resnet.to(device)
    if device == 'cuda':
       resnet = torch.nn.DataParallel(resnet)
       autoencoder = torch.nn.DataParallel(autoencoder)
       cudnn.benchmark = True
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                              shuffle=True, num_workers=2)
	#16
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                             shuffle=False, num_workers=2)

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
    criterion = nn.BCELoss()
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
        for i, (inputs, labels) in enumerate(trainloader):
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
            #net = Net()
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

            progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Accuracy: %.3f' % (train_loss/(i+1), 100.*correct/total, correct, total, best_acc))

        global best_acc
        #net.eval()
        test_loss = 0
        test_loss_autoencoder = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, test_labels) in enumerate(testloader):
                test_inputs, test_labels = inputs.to(device), test_labels.to(device)

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

                test_loss_resnet = criterion_resnet(test_outputs_resnet, test_labels)

                test_loss += test_loss_resnet.item()
                _, predicted = test_outputs_resnet.max(1)
                test_total += test_labels.size(0)
                test_correct += predicted.eq(test_labels).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss Resnet: %.3f | Loss AE: %.3f | Acc: %.3f%% (%d/%d) | Best Accuracy: %.3f'
                    % (test_loss/(batch_idx+1), test_loss_autoencoder/(batch_idx+1),100.*test_correct/test_total, test_correct, test_total, best_acc))

        # Save checkpoint.
        train_acc = 100.*correct/total
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            print('Best Training Epoch: ', epoch)
            print('Best Training Accuracy: ', best_train_acc)
            file1 = open("Best_train_result_datasetOriginal.txt","w")
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
            if not os.path.isdir('checkpoint_datasetOriginal'):
                os.mkdir('checkpoint_datasetOriginal')
            torch.save(state, './checkpoint_datasetOriginal/ckpt.pth')
            best_acc = acc
            print('Best Test Epoch: ', epoch)
            print('Best Test Accuracy: ', best_acc)
            file1 = open("Best_test_result_datasetOriginal.txt","w")
            file1.write("Best Testing Epoch: " + str(epoch) + "/n")
            file1.write("Best Test Accuracy: " + str(best_acc) + "/n")
            file1.close()

    print('Training completed!')
    print('Saving Model...')
    if not os.path.exists('./weights_datasetOriginal'):
        os.mkdir('./weights_datasetOriginal')
    torch.save(autoencoder.state_dict(), "./weights_datasetOriginal/autoencoder.pkl")

    #----------------------Print category wise----------------------
    correct_category = list(0. for i in range(10))
    total_category = list(0. for i in range(10))
    iterations_Category =  int((math.floor(len(testLabel)/batchSize))) + 1
    with torch.no_grad():
        for data in testloader:
            images_category, labels_category = data
            test_encoded_category, test_outputs_category = autoencoder(images_category)
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
