import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cifar_pytorch import Net, CIFAR3
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 256

    criterion = torch.nn.CrossEntropyLoss()

    test_transform = transforms.Compose([
        transforms.Normalize(mean=[127.5, 127.5, 127.5],
                             std=[127.5, 127.5, 127.5])
    ])
    test_data = CIFAR3("test", transform=test_transform)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Prevent non-model file to avoid crashes
    model_files = sorted(filter(lambda file: os.path.splitext(file)[1] == '.pt',
                                os.listdir('saved_models')))

    test_accs = []

    for i, model_file in enumerate(model_files):
        model = Net()
        model.to(device)

        model.load_state_dict(torch.load(
            os.path.join('saved_models', model_file),
            map_location=device))

        test_acc = 0
        model.eval()

        for j, input in enumerate(testloader, 0):
            x = input[0].to(device)
            y = input[1].type(torch.LongTensor).to(device)

            out = model(x)

            loss = criterion(out, y)
            _, predicted = torch.max(out.data, 1)
            correct = (predicted == y).sum()

            test_acc += correct.item()

        test_acc /= len(test_data)

        print(f'Model({i}):{model_file}, Test accuracy:{test_acc}')

        test_accs.append(test_acc)

    color = 'tab:blue'
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax1.plot(range(len(test_accs)), test_accs, c=color, label="Test Acc.", alpha=0.25)
    ax1.set_ylabel(" Accuracy", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.01, 1.01)
    plt.xticks(range(len(test_accs)), model_files,
               rotation=45, ha='right', rotation_mode='anchor')
    fig.tight_layout()
    ax1.legend(loc="center")
    plt.grid(True)
    plt.show()
