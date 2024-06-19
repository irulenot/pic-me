from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights

img = read_image("meowster.jpeg")

# Step 1: Initialize model with the best available weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")