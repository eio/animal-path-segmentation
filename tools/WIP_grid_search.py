import torch

# Define your hyperparameter options
hyperparams = {
    "model_type": ["RNN", "LSTM"],
    "optimizer": ["Adam", "SGD"],
    "learning_rate": [0.001, 0.01],
    "dropout": [0.2, 0.3, 0.4],
    "hidden_size": [32, 64, 128],
    "num_layers": [2, 3, 4],
}

# Define a function to train and evaluate the model
def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, device):
    model.to(device)
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation loop
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Test Accuracy = {accuracy:.3f}%")


# Perform grid search
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50

best_accuracy = 0.0
best_hyperparams = {}

for model_type in hyperparams["model_type"]:
    for optimizer_name in hyperparams["optimizer"]:
        for learning_rate in hyperparams["learning_rate"]:
            for dropout in hyperparams["dropout"]:
                for hidden_size in hyperparams["hidden_size"]:
                    for num_layers in hyperparams["num_layers"]:
                        # Create the model instance
                        if model_type == "RNN":
                            model = MyModel(
                                input_size=28,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                            )
                        else:
                            model = MyModel(
                                input_size=28,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                            )

                        # Define the optimizer and criterion
                        if optimizer_name == "Adam":
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        else:
                            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                        criterion = nn.CrossEntropyLoss()

                        # Train and evaluate the model
                        train_and_evaluate(
                            model,
                            optimizer,
                            criterion,
                            train_loader,
                            test_loader,
                            device,
                        )

                        # Check if the current model's accuracy is better than the best
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_hyperparams = {
                                "model_type": model_type,
                                "optimizer": optimizer_name,
                                "learning_rate": learning_rate,
                                "dropout": dropout,
                                "hidden_size": hidden_size,
                                "num_layers": num_layers,
                            }

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:")
print(best_hyperparams)
print("Best Accuracy:")
print(best_accuracy)
