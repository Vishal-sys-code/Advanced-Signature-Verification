# -------------------------------
# Vision Transformer (ViT) Definition
# -------------------------------
class ViTSignature(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTSignature, self).__init__()
        # Load a pre-trained ViT model from timm. Here we use "vit_base_patch16_224".
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Modify the classifier head to output the desired number of classes.
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
    
    def forward(self, x):
        # If input images are grayscale (1 channel), replicate to 3 channels.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)

# -------------------------------
# Training Function for Vision Transformer
# -------------------------------
def train_one_epoch_vit(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# -------------------------------
# 1. Train Vision Transformer on CEDAR Dataset
# -------------------------------
print("Training ViT on CEDAR Dataset")
model_vit_cedar = ViTSignature(num_classes=2).to(device)
optimizer_vit_cedar = optim.Adam(model_vit_cedar.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    loss_vit_cedar, acc_vit_cedar = train_one_epoch_vit(model_vit_cedar, cedar_train_loader, optimizer_vit_cedar, criterion, device)
    print(f"Epoch {epoch+1:2d} - CEDAR ViT Loss: {loss_vit_cedar:.4f}, Accuracy: {acc_vit_cedar*100:.2f}%")