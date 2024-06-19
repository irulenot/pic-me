class Classifier:
    def __init__(self):
        from torchvision.io import read_image
        from torchvision.models import resnet18, ResNet18_Weights
        self.read_image = read_image
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)
        self.model.eval()

    def forward(self, img_path):
        from torchvision.io import read_image
        img = read_image(img_path)
        preprocess = self.weights.transforms()
        batch = preprocess(img).unsqueeze(0)
        prediction = self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.weights.meta["categories"][class_id]
        return category_name, score

classifier = Classifier()
img_path = "meowster.jpeg"
category_name, score = classifier.forward(img_path)
print(f"{category_name}: {100 * score:.1f}%")