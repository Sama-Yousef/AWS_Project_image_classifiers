import torch
from torchvision import models
import argparse
import json
from PIL import Image
import numpy as np
from torch import nn, optim


# Function to parse arguments
def parse_args():
    print('parse_args')

    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    
    return parser.parse_args()

# Load the model from checkpoint
def load_checkpoint(filepath, return_only_model=False):
    print('load_checkpoint')

    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)

    # Recreate the classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if return_only_model:
        return model
    return model, optimizer, checkpoint['class_to_idx'], checkpoint['epochs']

    #return model
# Preprocess the image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    print('process_image')
    
    # Load the image
    img = Image.open(image)
    
    # Resize the image where the shortest side is 256 pixels
    img.thumbnail((256, 256))
    
    # Crop the center 224x224
    width, height = img.size
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize
    np_image = np.array(img) / 255.0  
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    # Transpose to fit the model input
    np_image = np_image.transpose((2, 0, 1))  
    
    tensor_image = torch.tensor(np_image).float()
    
    return tensor_image

# Predict function
def predict(image_path, model,topk=5):
    print('predict')

    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Set the model to evaluation mode
    model.eval()
    
    # Process the image
    img_tensor = process_image(image_path).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)  # Move input tensor to the same device as the model
    
    with torch.no_grad():
        output = model(img_tensor)  # Forward pass
    
    # Calculate probabilities
    ps = torch.exp(output)
    
    # Get the top K probabilities and their corresponding class indices
    top_p, top_class = ps.topk(topk, dim=1)
    
    # Convert to numpy and adjust class indices if needed
    top_p = top_p[0].cpu().numpy()
    top_class = top_class[0].cpu().numpy().astype(int) + 1    # Adjust class indices to start from 1
     
    return top_p, top_class
    
# Map category to name





################################### MAIN ######################################################




if __name__ == '__main__':
    print('main')
    args = parse_args()
    
    # Load the checkpoint and model
    model, optimizer, class_to_idx, epochs = load_checkpoint(args.checkpoint)

    # Use class_to_idx directly
    

    
    # Use GPU if specified
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available()) 
    model.to(device)
    
    # Predict the top K classes
    probs, classes = predict(args.image_path, model,args.top_k)
    
    # Load category names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(cls)] for cls in classes]
    else:
        class_names = classes
    
    # Display results
    print("Top K Probabilities:", probs)
    print("Top K Classes:", class_names)
