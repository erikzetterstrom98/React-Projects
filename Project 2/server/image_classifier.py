import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.nn.functional import linear
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import sys
import os
from tqdm import tqdm
import datetime

class Squash(nn.Module):
    def __init__(self, eps = 10e-21, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps 

    def forward(self, s):
        n = torch.linalg.norm(s, dim = -1, keepdim = True)
        return (1 - 1/(torch.exp(n)+self.eps))*(s/(n+self.eps))
    
class FullyConnectedCapsule(nn.Module):
    def __init__(self, N, D, **kwargs) -> None:
        super().__init__(**kwargs)

        self.N = N 
        self.D = D
        self.W = nn.init.kaiming_normal_(torch.empty((self.N, 16, 8, self.D)))
        self.b = torch.zeros((self.N, 16, 1))

    def forward(self, inputs):
        # self.N = 10
        # self.D = 16
        # input_N = 16 
        # input_D = 8
        u = torch.einsum( 
            '...ji,kjiz->...kjz', # inputs[:, j = 16, i = 8], self.W[k = 10, j = 16, i = 8, z = 16] -> u[:, k = 10, j = 16, z = 16]
            [inputs, self.W]
        )

        c = torch.einsum(
            '...ij,...kj->...i', # u[: = 32, : = 10, i = 16, j = 16], u[: = 32, : = 10, k = 16, j = 16] -> c[: = 32, : = 10, i = 16]
            [u, u]
        )[..., None]
        
        
        c =  c/torch.sqrt(torch.Tensor([self.D]))
        
        c = nn.functional.softmax(
            input = c,
            dim = 1
        )

        c = c + self.b # This does not do shit?
        
        s = torch.sum(torch.multiply(u, c), dim = -2)
        
        return s

class MNIST_Classifier(nn.Module):
    def __init__(self, image_height: int, image_width: int, name: str = None) -> None:
        super().__init__()
        
        if not name:
            time = datetime.datetime.now()
            name = f'{time.day}-{time.month}-{time.hour}:{time.minute}:{time.second}'

        self.model_name = f'model-{name}.pth'
        # Efficient-CapsNet (https://github.com/EscVM/Efficient-CapsNet)
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels = 1, # Monochrome images
                out_channels = 32,
                kernel_size = 5
            ),
            nn.BatchNorm2d(
                num_features = 32
            ),
            nn.ReLU(
                inplace = True
            ),
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 64,
                kernel_size = 3
            ),
            nn.BatchNorm2d(
                num_features = 64
            ),
            nn.ReLU(
                inplace = True
            ),
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 64,
                kernel_size = 3
            ),
            nn.BatchNorm2d(
                num_features = 64
            ),
            nn.ReLU(
                inplace = True
            ),
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 128,
                kernel_size = 3,
                stride = 2
            ),
            nn.BatchNorm2d(
                num_features = 128
            ),
            nn.ReLU(
                inplace = True # Oklart om den ska vara här?
            ),
            nn.Conv2d( # PrimaryCaps (https://github.com/EscVM/Efficient-CapsNet/blob/main/utils/layers.py)
                in_channels = 128,
                out_channels = 128,
                kernel_size = 9,
                groups = 128
            ),
            nn.Flatten(
                start_dim = 1,
                end_dim = 3
            ),
            nn.Unflatten(
                dim = 1,
                unflattened_size = (16, 8)
            ),
            Squash(),
            FullyConnectedCapsule(
                N = 10,
                D = 16
            ),
            Squash()
        )
        
    def capsule_length(self, inputs, eps = 1e-12):
        return torch.sqrt(
            torch.sum(
                torch.square(
                    inputs
                ),
                dim = -1
            )
            +
            eps
        )
    
    def forward(self, inputs):
        digit_campsules = self.network(inputs)
        capsule_lengths = self.capsule_length(
            inputs = digit_campsules
        )
        probabilities = nn.functional.softmax(capsule_lengths, dim = 1)
        return capsule_lengths, probabilities
    
    def run_training(self, training_data_loader, test_data_loader, optimiser, loss_function, epochs = 100, verbose = True):
        if verbose:
            print(f'Starting training. Will train for {epochs} epochs.')

        best_evaluation_accuracy = -1

        for epoch in range(epochs):
            training_loss, training_accuracy = self.train_epoch(
                data_loader = training_data_loader,
                optimiser = optimiser,
                loss_function = loss_function
            )

            evaluation_loss, evaluation_accuracy = self.evaluate_epoch(
                data_loader = test_data_loader,
                loss_function = loss_function
            )

            if verbose:
                print(f'  Epoch {epoch+1}:\n    Training loss: {training_loss}\n    Training accuracy: {str(training_accuracy*100)[:5]}%\n    Evaluation loss: {evaluation_loss}\n    Evaluation accuracy: {str(evaluation_accuracy*100)[:5]}%')

            if evaluation_accuracy > best_evaluation_accuracy:
                torch.save(self, self.model_name)
                best_evaluation_accuracy = evaluation_accuracy 

                if verbose:
                    print('  New best evaluation accuracy! Saving model...')

        if verbose:
            print(f'Done!')

    def train_epoch(self, data_loader, optimiser, loss_function):
        self.train()
        total_epoch_loss = 0
        total_epoch_accuracy = 0

        for images, labels in tqdm(data_loader):
            outputs, probabilities = self.forward(images)
            loss = loss_function(
                input = outputs,
                target = labels
            )

            accuracy = sum([1 for index in range(len(labels)) if torch.argmax(outputs, dim = 1)[index] == labels[index]])/len(labels)

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

            total_epoch_loss += loss 
            total_epoch_accuracy += accuracy
        
        return total_epoch_loss/len(data_loader), total_epoch_accuracy/len(data_loader)
    
    def evaluate_epoch(self, data_loader, loss_function):
        self.eval()
        total_epoch_loss = 0
        total_epoch_accuracy = 0

        with torch.no_grad():
            for images, labels in tqdm(data_loader):
                outputs, probabilities = self.forward(images)

                loss = loss_function(
                    input = outputs,
                    target = labels
                )

                accuracy = sum([1 for index in range(len(labels)) if torch.argmax(outputs, dim = 1)[index] == labels[index]])/len(labels)

                total_epoch_loss += loss 
                total_epoch_accuracy += accuracy

        return total_epoch_loss/len(data_loader), total_epoch_accuracy/len(data_loader)
    
    def fit(self, images):
        self.eval()

        with torch.no_grad():
            outputs, probabilities = self.forward(images)

        return outputs, probabilities

# training_dataset = MNIST(
#     root = 'data',
#     train = True,
#     transform = ToTensor()
# )

test_dataset = MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

# training_data_loader = DataLoader(
#     dataset = training_dataset,
#     batch_size = 32,
#     shuffle = True
# )

test_data_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 32,
    shuffle = False
)

loss_function = nn.CrossEntropyLoss()

# model = MNIST_Classifier(
#     image_height = 28,
#     image_width = 28,
#     name = 'efficient_capsules'
# )

# optimiser = Adam(
#     model.parameters()
# )

# model.run_training(
#     training_data_loader = training_data_loader,
#     test_data_loader = test_data_loader,
#     optimiser = optimiser,
#     loss_function = loss_function
# )

# def load_model():
#     model = torch.load('model-efficient_capsules.pth')
#     model.eval()
#     return model

# model = load_model()


# loss, accuracy = model.evaluate_epoch(
#     data_loader = test_data_loader,
#     loss_function = loss_function
# )

# print('Loss:', loss)
# print('Accuracy:', accuracy)

# images = [torch.load(f'{number}.pth') for number in range(10)]
# print(images)