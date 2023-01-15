# %% [markdown]
# # Homework 2 - Convolutional Neural Nets
#
# In this homework, we will be working with google [colab](https://colab.research.google.com/).
# Google colab allows you to run a jupyter notebook on google servers using a GPU or TPU.
# To enable GPU support, make sure to press Runtime -> Change Runtime Type -> GPU.

# %% [markdown]
# ## Cats vs dogs classification
#
# To learn about and experiment with convolutional neural nets we will be working on a problem
# of great importance in computer vision - classifying images of cats and dogs.
#
# The problem is so important that there's even an easter egg in colab: go to
# Tools -> Settings -> Miscellaneous and enable 'Corgi mode' and 'Kitty mode'
# to get more cats and dogs to classify when you're tired of coding.
#
#

# %% [markdown]
# ### Getting the data
#
# To get started with the classification, we first need to download and unpack the dataset
# (note that in jupyter notebooks commands starting with `!` are executed in bash, not in python):

# %%

# %%

# %% [markdown]
# This dataset contains two directories, `train` and `validation`. Both in turn contain two
# directories with images: `cats` and `dogs`. In `train` we have 1000 images of cats,
# and another 1000 images of dogs. For `validation`, we have 500 images of each class.
# Our goal is to implement and train a convolutional neural net to classify these images, i.e.
# given an image from this dataset, tell if it contains a cat or a dog.
#
#

# %%

# %%
# ### Loading the data
#  Now that we have the data on our disk, we need to load it so that we can use it to train our
# model. In Pytorch ecosystem, we use `Dataset` class, documentation for which can be found
# [here](https://pytorch.org/docs/stable/data.html).
#
#  In the case of computer vision, the datasets with the folder structure 'label_name/image_file'
# are very common, and to process those there's already a class `torchvision.datasets.ImageFolder`
# (documented [here](https://pytorch.org/vision/0.8/datasets.html)). Torchvision is a Pytorch
# library with many commonly used tools in computer vision.
#
#  Another thing we need from Torchvision library is transforms
# ([documentation](https://pytorch.org/docs/stable/torchvision/transforms.html)).
# In computer vision, we very often want to transform the images in certain ways.
# The most common is normalization. Others include flipping, changing saturation, hue,
# contrast, rotation, and blurring.
#
#  Below, we create a training, validation and test sets. We use a few transforms for augmentation
# on the training set, but we don't use anything but resize and normalization for validation
# and test.

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import nn
from torchvision import transforms
from tqdm.notebook import tqdm

# These numbers are mean and std values for channels of natural images.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Inverse transformation: needed for plotting.
unnormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

train_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(hue=0.1, saturation=0.1, contrast=0.1),  # type: ignore
        transforms.RandomRotation(20),
        transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
        transforms.ToTensor(),  # convert PIL to Pytorch Tensor
        normalize,
    ]
)

validation_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ]
)

train_dataset = torchvision.datasets.ImageFolder(
    "./homeworks/HW2/cats_and_dogs_filtered/train", transform=train_transforms
)
validation_dataset, test_dataset = torch.utils.data.random_split(  # type: ignore
    torchvision.datasets.ImageFolder(
        "./homeworks/HW2/cats_and_dogs_filtered/validation", transform=validation_transforms
    ),
    [500, 500],
    generator=torch.Generator().manual_seed(42),
)

# %% [markdown]
# Let's see what one of the images in the dataset looks like (you can run this cell multiple
# times to see the effects of different augmentations):

# %%
plt.rcParams["figure.dpi"] = 200  # change dpi to make plots bigger


def show_normalized_image(img, title: str = ""):
    plt.imshow(unnormalize(img).detach().cpu().permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")


show_normalized_image(train_dataset[10][0])

# %% [markdown]
# ### Creating the model

# %% [markdown]
# Now is the time to create a model. All models in Pytorch are subclassing
# `torch.nn.Module`, and have to implement `__init__` and `forward` methods.
#
# Below we provide a simple model skeleton, which you need to expand.
# The places to put your code are marked with `TODO`. Here, we ask you to implement a
# convolutional neural network containing the following elements:
#
# * Convolutional layers (at least two)
# * Batch Norm
# * Non-linearity
# * Pooling layers
# * A residual connection similar to that of Res-Net
# * A fully connected layer
#
# For some examples of how to implement Pytorch models, please refer to our lab notebooks,
# such as [this one](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/06-convnet.ipynb).

# %%


class CNN(torch.nn.Module):
    def __init__(self, n_feature=3, output_size=2):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature * 61 * 61, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


# %% [markdown]
# ### Training the model
#
# Now we train the model on the dataset. Again, we're providing you with the skeleton with
# some parts marked as `TODO` to be filled by you.

# %%


def get_loss_and_correct(model, batch, criterion, device):
    # Implement forward pass and loss calculation for one batch.
    # Remember to move the batch to device.
    #
    # Return a tuple:
    # - loss for the batch (Tensor)
    # - number of correctly classified examples in the batch (Tensor)
    criterion.to(device)
    output = batch[0]
    target = batch[1]
    prediction = model(output)
    loss = criterion(prediction, target)
    correct = torch.sum(target == torch.argmax(prediction, dim=1))

    return loss, correct


def step(loss, optimizer):
    # Implement backward pass and update.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


N_EPOCHS = 5  # TODO
BATCH_SIZE = 64  # TODO

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, num_workers=0
)
model = CNN()

criterion = torch.nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

model.train()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

# Issue with tqdm: https://github.com/microsoft/vscode-jupyter/issues/8552
pbar = tqdm(range(N_EPOCHS))

for i in pbar:
    total_train_loss = 0.0
    total_train_correct = 0.0
    total_validation_loss = 0.0
    total_validation_correct = 0.0

    model.train()

    for batch in tqdm(train_dataloader, leave=False):
        loss, correct = get_loss_and_correct(model, batch, criterion, device)
        step(loss, optimizer)
        total_train_loss += loss.item()
        total_train_correct += correct.item()

    with torch.no_grad():
        for batch in validation_dataloader:
            loss, correct = get_loss_and_correct(model, batch, criterion, device)
            total_validation_loss += loss.item()
            total_validation_correct += correct.item()

    mean_train_loss = total_train_loss / len(train_dataset)
    train_accuracy = total_train_correct / len(train_dataset)

    mean_validation_loss = total_validation_loss / len(validation_dataset)
    validation_accuracy = total_validation_correct / len(validation_dataset)

    train_losses.append(mean_train_loss)
    validation_losses.append(mean_validation_loss)

    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)

    pbar.set_postfix(
        {
            "train_loss": mean_train_loss,
            "validation_loss": mean_validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        }
    )

# %% [markdown]
# Now that the model is trained, we want to visualize the training and validation losses
# and accuracies:

# %%
plt.figure(dpi=200)

plt.subplot(121)
plt.plot(train_losses, label="train")
plt.plot(validation_losses, label="validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Losses")
plt.legend()

plt.subplot(122)
plt.plot(train_accuracies, label="train")
plt.plot(validation_accuracies, label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.title("Accuracies")

plt.tight_layout()

# %% [markdown]
# Now, change your model to achieve at least 75% accuracy on validation set. You can change the
# model you've implemented, the optimizer, and the augmentations.
#
# Looking at the loss and accuracy plots, can you see if your model overfits the trainig set? Why?
#
# Answer:
# `put your answer here`

# %% [markdown]
# ### Testing the model
#
# Now, use the `test_dataset` to get the final accuracy of your model. Visualize some correctly
# and incorrectly classified examples.

# %%
# TODO
# 1. Calculate and show the test_dataset accuracy of your model.
# 2. Visualize some correctly and incorrectly classified examples.

# %% [markdown]
# ### Visualizing filters
#
# In this part, we are going to visualize the output of one of the convolutional layers to see
# what features they focus on.
#
# First, let's get some image.

# %%
image = validation_dataset[10][0]
show_normalized_image(image)

# %% [markdown]
# Now, we are going to 'clip' our model at different points to get different intermediate
# representation.
# Clip your model at two or three different points and plot the filters output.
#
# In order to clip the model, you can use `model.children()` method. For example, to get output
# only after the first 4 layers, you can do:
#
# ```
# clipped = nn.Sequential(
#     *list(model.children()[:4])
# )
# intermediate_output = clipped(input)
# ```
#
#

# %%


def plot_intermediate_output(result, title):
    """Plots the intermediate output of shape
    N_FILTERS x H x W
    """
    n_filters = result.shape[1]
    N = int(math.sqrt(n_filters))
    M = (n_filters + N - 1) // N
    assert N * M >= n_filters

    fig, axs = plt.subplots(N, M)
    fig.suptitle(title)

    for i in range(N):
        for j in range(M):
            if i * N + j < n_filters:
                axs[i][j].imshow(result[0, i * N + j].cpu().detach())
                axs[i][j].axis("off")


# TODO:
# pick a few intermediate representations from your network and plot them using
# the provided function.

# %% [markdown]
# What can you say about those filters? What features are they focusing on?
#
# Anwer: `Your answer here`
