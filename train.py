import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import json

# Define a function to parse command-line arguments
def parse_args():
    print('parse_args')
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset")
    parser.add_argument('data_dir', type=str, help='Directory containing training, validation, and test data')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (e.g., vgg16, densenet121)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    
    return parser.parse_args()

# Function to load data with transformations
def load_data(data_dir):
    print('load_data')
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    # TODO: Define your transforms for the training, validation, and testing sets

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # TODO: Load the datasets with ImageFolder
    image_datasets = {}
    for x in ['train', 'valid', 'test']:
        print(data_dir + '/' + x)
        image_datasets[x] = datasets.ImageFolder(
            root = data_dir + '/' + x, 
            
            transform = data_transforms[x]
        )

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {}
    batch_size = 100

    for x in ['train', 'valid', 'test']:
        dataloaders[x] = torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True
        )


    trainloader=dataloaders['train']
    validloader=dataloaders['valid']
    testloader=dataloaders['test']



    return trainloader, validloader, testloader

# Function to create and load model
def create_model(arch, hidden_units):
    print('create_model')
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)

        #freeze parameters of the pre-trained network 
        for param in model.parameters():
            param.requires_grad = False
            
        # the new classifier
        classifier = nn.Sequential(
            nn.Linear(25088, hidden_units), # this is the shape of the output VGG model (to match the previous network)
            nn.ReLU(),# this to present the unlinearity
            nn.Dropout(0.5), # to prevent overfiting and for regalization
            nn.Linear(hidden_units, 102),  # 102 is the number of flower categories
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier #appending the new arch to the model
    else:
        print(f"Architecture {arch} is not supported")
        exit()


    return model

# Function to train the model
def train_model(model, trainloader, validloader, criterion, optimizer, epochs, device):
    print('train_model')
    model = model.to(device)  
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    epochs = 4  
    steps = 0
    
    for e in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)  # Get the index of the max log-probability
            total += labels.size(0)              # Increment the total by batch size
            correct += (predicted == labels).sum().item()  # Increment the correct count
            
            

        # Calculate accuracy after each epoch
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch {e+1}/{epochs}.. "
                f"Training loss: {running_loss / len(trainloader):.3f}.. "
                f"Accuracy: {accuracy:.2f}%")
        
        valid_loss = 0
        total_valid = 0  # Track the total number of samples in validation
        correct_valid = 0  # Track the correct predictions in validation
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                valid_loss += criterion(output, labels).item()
                
                # Calculate validation accuracy
                _, predicted = torch.max(output, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        

        valid_accuracy = 100 * correct_valid / total_valid

        print(f"Epoch {e+1}/{epochs}.. "
                f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                f"Validation accuracy: {valid_accuracy:.2f}%")
        

    
    return model


# Function to test the model on the test dataset
def test_model(model, testloader,  device):
    print('test_model')

    model.eval()

    test_loss = 0
    accuracy = 0
    correct_images = []  # List to store correctly classified images
    correct_labels = []  # List to store corresponding labels (flower names)

    with torch.no_grad():  # Turn off gradients for validation
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            output = model(images) # Shape: [batch_size, 102] (log probabilities for 102 classes)
            test_loss += criterion(output, labels).item()

            # Calculate accuracy
            ps = torch.exp(output)  # Convert log probabilities to probabilities
            top_p, top_class = ps.topk(1, dim=1)  # Get the class with the highest probability
            equals = top_class == labels.view(*top_class.shape)  # Compare predicted classes with actual labels
            
            accuracy += torch.mean(equals.type(torch.FloatTensor).to(device)).item()  # Calculate batch accuracy
            
            # Store the correctly classified images and their corresponding labels
            for i in range(len(images)):
                if equals[i]:  # If the image is correctly classified
                    correct_images.append(images[i].cpu())  # Move to CPU before storing
                    
            
    # Print the test loss and accuracy
    print(f"Test Loss: {test_loss / len(testloader):.3f}.. "
        f"Test Accuracy: {accuracy / len(testloader) * 100:.2f}%")





# Function to save the checkpoint
def save_checkpoint(model, save_dir, train_data,epochs):
    print('save_checkpoint')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'epochs': epochs
    }

    save_dir = 'flower_classifier_checkpoint.pth'
    torch.save(checkpoint, save_dir)

    print(f"Checkpoint saved to {save_dir}")


################################################### MAIN ############################################################


if __name__ == '__main__':
    print('__main__')
    
    args = parse_args()
    trainloader, validloader, train_data = load_data(args.data_dir)
    model = create_model(args.arch, args.hidden_units)
    
    # Use GPU if specified
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Train the model
    trained_model = train_model(model, trainloader, validloader, criterion, optimizer, args.epochs, device)
    
    # Save the checkpoint
    save_checkpoint(trained_model, args.save_dir, train_data, args.epochs)