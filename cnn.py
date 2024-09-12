import torch
from torch.utils.data import DataLoader
from torch import nn
import torchmetrics
from torchvision.transforms import transforms
import torchvision
import json

# Transforms
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Create the model
class OurCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(OurCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(12, 20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 75 * 75, num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = torch.flatten(output, 1)

        output = self.fc(output)

        return output

def train_cnn():
    # Model initialization
    model = OurCNN()

    # Train the model
    epochs = 250
    learning_rate = 0.001
    train_path = 'myDataset/train/'
    test_path = 'myDataset/test/'

    # Create the dataloader
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transformer)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=transformer)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    # Create the accuracy metric
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)

    # Store accuracy and loss data
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    }

    # Define the training loop
    def train_loop(dataloader, model, loss_fn, optimizer):
        model.train()
        total_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()

            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            acc = metric(pred, y)
            total_loss += loss.item()

        acc = metric.compute()
        metric.reset()
        return total_loss / len(dataloader), acc

    # Define the testing loop
    def test_loop(dataloader, model, loss_fn):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                loss = loss_fn(pred, y)
                acc = metric(pred, y)
                total_loss += loss.item()

        acc = metric.compute()
        metric.reset()
        return total_loss / len(dataloader), acc

    best_accuracy = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_accuracy'].append(train_acc.item())
        history['test_accuracy'].append(test_acc.item())

        if test_acc > best_accuracy:
            torch.save(model.state_dict(), 'cnn4.model')
            best_accuracy = test_acc

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save the training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    train_cnn()
