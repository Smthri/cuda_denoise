import torch
from torch import nn
from torchinfo import summary
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imsave


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.act1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.act2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)

        # decoder
        self.pool3 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        self.act3 = nn.ReLU(inplace=False)
        self.pool4 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        self.act4 = nn.ReLU(inplace=False)
        self.pool5 = nn.Conv2d(32, 1, 1)
        self.act5 = nn.Sigmoid()

    def forward(self, x):
        # encode
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        # decode
        x = self.pool3(x)
        x = self.act3(x)
        x = self.pool4(x)
        x = self.act4(x)
        x = self.pool5(x)
        x = self.act5(x)
        return x


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=15)
    ])


if __name__ == '__main__':
    model = MyModel()
    summary(model, input_size=(1, 1, 28, 28))
    dataset = MNIST('./MNIST', transform=get_transforms(), download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    test_batch = next(iter(dataloader))
    data, _ = test_batch
    noisy_data = torch.clip(data + 0.4 * torch.randn(*data.shape), 0, 1)
    for idx, img in enumerate(noisy_data):
        img = img.squeeze().numpy()
        imsave(f'test_images/{idx}.png', img)

    n_epochs = 50
    for epoch in range(n_epochs):
        losses = []
        for batch in tqdm(dataloader):
            data, _ = batch
            data = data.to(device)
            optimizer.zero_grad()

            noisy_data = torch.clip(data + 0.4 * torch.randn(*data.shape), 0, 1)
            outputs = model(noisy_data)
            inputs = torch.zeros_like(outputs)
            h, w = data.shape[2:]
            inputs[..., :h, :w] = data
            noisy_inputs = torch.zeros_like(outputs)
            noisy_inputs[..., :h, :w] = noisy_data

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        H = 5
        W = 2
        fig, axs = plt.subplots(H, W, figsize=(W, H))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for i in range(H):
            idx = np.random.randint(0, noisy_inputs.shape[0])
            noisy = noisy_inputs[idx].detach().cpu().numpy()
            noisy = np.transpose(noisy, [1, 2, 0])
            axs[i, 0].imshow(noisy)
            denoised = outputs[idx].detach().cpu().numpy()
            denoised = np.transpose(denoised, [1, 2, 0])
            axs[i, 1].imshow(denoised)
        plt.savefig('example.png')


        print(f'Done epoch {epoch}. Average MSE loss is {sum(losses) / len(losses):.2f}.')
    torch.save(model.state_dict(), 'ckpt.pth')
