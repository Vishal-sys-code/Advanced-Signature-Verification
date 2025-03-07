# Advanced Model: DenseNetSignature using DenseNet121
class DenseNetSignature(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNetSignature, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        # Modify first convolution: DenseNet expects 3 channels; change to 1.
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.densenet(x)

# Example training loop for DenseNetSignature on CEDAR dataset:
print("\nTraining DenseNetSignature on CEDAR Dataset")
model_densenet_cedar = DenseNetSignature(num_classes=2).to(device)
optimizer_densenet_cedar = optim.Adam(model_densenet_cedar.parameters(), lr=0.001)

for epoch in range(num_epochs):
    loss_dense, acc_dense = train_one_epoch(model_densenet_cedar, cedar_train_loader, optimizer_densenet_cedar, criterion, device)
    print(f"Epoch {epoch+1:2d} - CEDAR DenseNet Loss: {loss_dense:.4f}, Accuracy: {acc_dense*100:.2f}%")