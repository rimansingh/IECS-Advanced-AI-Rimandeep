# train.py
import gin
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

@gin.configurable
def train(model, learning_rate=0.001, epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    mlflow.start_run()
    mlflow.log_params({'learning_rate': learning_rate, 'epochs': epochs, 'batch_size': batch_size, 'model': model.__class__.__name__})

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        mlflow.log_metric('loss', avg_loss, step=epoch)
        mlflow.log_metric('accuracy', accuracy, step=epoch)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Log model with an input example converted to numpy
    example_input = torch.randn(1, 1, 28, 28, device=device).detach().cpu().numpy()
    mlflow.pytorch.log_model(model, "model", input_example=example_input)
    mlflow.end_run()
