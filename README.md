# Autoencoder-Resnet_cifar10_pytorch

## Framework:
Pytorch

## Models:
A combination of Autoencoder and ResNet34

## Dataset:
### 1. Original dataset distribution (https://www.cs.toronto.edu/~kriz/cifar.html)
* Training images: 50,000
* Testing images: 10,000
* Category Distribution(Training):
   - default (some categories may have more data than other categories)

### 2. DatasetB
* Training images: 46,800
* Testing images: 13,200
* Category Distribution(Training):
   - airplane (90%)
   - automobile (90%)
   - bird (50%)
   - cat (90%)
   - deer (50%)
   - dog (90%)
   - frog (90%)
   - horse (90%)
   - ship (90%)
   - truck (50%)

## Results:
### 1. main_datasetOriginal
 - Training: 99.968%
 - Testing: 86.56%
    - Accuracy of plane : 81 %
    - Accuracy of   car : 84 %
    - Accuracy of  bird : 84 %
    - Accuracy of   cat : 68 %
    - Accuracy of  deer : 92 %
    - Accuracy of   dog : 73 %
    - Accuracy of  frog : 61 %
    - Accuracy of horse : 91 %
    - Accuracy of  ship : 100 %
    - Accuracy of truck : 94 %


### 2. main_datasetB
 - Training: 100.0%
 - Testing: 68.5%
    - Accuracy of plane : 91 %
    - Accuracy of   car : 90 %
    - Accuracy of  bird : 60 %
    - Accuracy of   cat : 80 %
    - Accuracy of  deer : 53 %
    - Accuracy of   dog : 68 %
    - Accuracy of  frog : 88 %
    - Accuracy of horse : 83 %
    - Accuracy of  ship : 100 %
    - Accuracy of truck : 60 %



