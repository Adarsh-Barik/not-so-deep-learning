import torch
import torchvision

# for data loading
from torchvision import transforms
from torchvision.datasets import ImageFolder

# for batch size
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

# for base model, base class
import torch.nn as nn
import torch.nn.functional as F

# for plots
from matplotlib import pyplot as plt

# loading the data
# directories must have structure input/class, here we do not need class but we create one nevertheless
input_dir="/Users/adarsh/LocalStorage/Research/Tutorial/not-so-deep-learning/images/inputimages/"
output_dir="/Users/adarsh/LocalStorage/Research/Tutorial/not-so-deep-learning/images/outputimages/"

# input images, loaded in two parts, X[i] = tensor, label
# only need tensor part for training
X = ImageFolder(input_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

# output images, loaded in two parts, Y[i] = tensor, label
# only need tensor part for training
Y = ImageFolder(output_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))


# training and validation set

#  batch_size = 128 # 2^k, depends on memory
generator1 = torch.Generator().manual_seed(1) # fixing seed
val_size = 100 # validation set size
train_size = len(X) - val_size # training set size

train_X_data = torch.utils.data.Subset(X, range(train_size))
val_X_data = torch.utils.data.Subset(X, range(train_size, len(X)))
train_Y_data = torch.utils.data.Subset(Y, range(train_size))
val_Y_data = torch.utils.data.Subset(Y, range(train_size, len(X)))
#  train_Y_data,val_Y_data = random_split(Y, [train_size,val_size], generator=generator1)

#  for i in range(train_size):
#      train_X_data[i] = X[i]
#      train_Y_data[i] = Y[i]
#
#  for i in range(train_size, train_size+val_size):
#      val_X_data[i] = X[i]
#      val_Y_data[i] = Y[i]


# base class
class PIVBase(nn.Module):
    def training_step(self, Xdata, Ydata):
        Ypred = self(Xdata) # generate prediction
        #  print(Ypred.shape)
        #  print(Ydata.shape)
        loss = F.mse_loss(Ypred, Ydata)
        return loss

    def validation_step(self, Xdata, Ydata):
        Ypred = self(Xdata)
        loss = F.mse_loss(Ypred, Ydata)
        return loss.detach()

# cnn model
class ImageGenModel(PIVBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                #  nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                #  nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                #  nn.BatchNorm2d(128),
                nn.ReLU(),

                #  nn.MaxPool2d(2),

                #  nn.Upsample(scale_factor=2),

                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
                #  nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
                #  nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
                #  nn.BatchNorm2d(3),
                nn.ReLU() # should this be sigmoid
                )

    def forward(self, xb):
        return self.network(xb)
        #  """ encoder """
        #  x = self.conv1(x)
        #  x = self.batchnorm1(x)
        #  x = F.relu(x)
        #
        #  x = self.conv2(x)
        #  x = self.batchnorm2(x)
        #  x = F.relu(x)
        #
        #  x = self.conv3(x)
        #  x = self.batchnorm3(x)
        #  bottlenecks = F.relu(x)
        #
        #  """ decoder """
        #  x = self.deconv1(bottlenecks)
        #  x = self.batchnorm1(x)
        #  x = F.relu(x)
        #
        #  x = self.deconv2(x)
        #  x = self.batchnorm2(x)
        #  x = F.relu(x)
        #
        #  x = self.deconv3(x)
        #  x = torch.sigmoid(x)
        #
        #  return x

@torch.no_grad()
def evaluate(model, Xdata, Ydata):
    model.eval()
    output = 0
    for i in range(len(Xdata)):
        output = output + model.validation_step(Xdata[i], Ydata[i])
    return {'val_loss': output/len(Xdata)}

def fit(maxIter, lr, model, trainX, trainY, valX, valY, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    trainXdata = [a for (a, b) in trainX]
    trainYdata = [a for (a, b) in trainY]
    valXdata = [a for (a, b) in valX]
    valYdata = [a for (a, b) in valY]

    for t in range(maxIter):
        print("Iter: ", t)
        model.train()
        train_losses = []
        for i in range(len(trainXdata)):
            loss = model.training_step(trainXdata[i], trainYdata[i])
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, valXdata, valYdata)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        print("train loss: ", result['train_loss'], "val loss: ", result['val_loss'])
        history.append(result)

    return history


# running everything
model = ImageGenModel()
maxIter = 45
opt_func = torch.optim.Adam
lr = 0.00005

history = fit(maxIter, lr, model, train_X_data, train_Y_data, val_X_data, val_Y_data, opt_func)

# store the model
torch.save(model.state_dict(), './data/storedmodel')

# to load
# model = ImageGenModel()
# model.load_state_dict(torch.load('./data/storedmodel'))

def displayimg(i, model=model, X = val_X_data, Y = val_Y_data):
    x, a = X[i]
    y, a = Y[i]
    ypred = model(x).detach()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('input, output, prediction')
    ax1.imshow(x.permute(1,2,0))
    ax2.imshow(y.permute(1,2,0))
    ax3.imshow(ypred.permute(1,2,0))
    plt.show()
