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

# Repeat similarly for GPDS, SIGCOMP, and MCYT:
print("\nTraining DenseNetSignature on GPDS Dataset")
model_densenet_gpds = DenseNetSignature(num_classes=2).to(device)
optimizer_densenet_gpds = optim.Adam(model_densenet_gpds.parameters(), lr=0.001)
for epoch in range(num_epochs):
    loss_dense_gpds, acc_dense_gpds = train_one_epoch(model_densenet_gpds, gpds_train_loader, optimizer_densenet_gpds, criterion, device)
    print(f"Epoch {epoch+1:2d} - GPDS DenseNet Loss: {loss_dense_gpds:.4f}, Accuracy: {acc_dense_gpds*100:.2f}%")

print("\nTraining DenseNetSignature on SIGCOMP Dataset")
model_densenet_sigcomp = DenseNetSignature(num_classes=2).to(device)
optimizer_densenet_sigcomp = optim.Adam(model_densenet_sigcomp.parameters(), lr=0.001)
for epoch in range(num_epochs):
    loss_dense_sigcomp, acc_dense_sigcomp = train_one_epoch(model_densenet_sigcomp, sigcomp_train_loader, optimizer_densenet_sigcomp, criterion, device)
    print(f"Epoch {epoch+1:2d} - SIGCOMP DenseNet Loss: {loss_dense_sigcomp:.4f}, Accuracy: {acc_dense_sigcomp*100:.2f}%")

print("\nTraining DenseNetSignature on MCYT Dataset")
model_densenet_mcyt = DenseNetSignature(num_classes=2).to(device)
optimizer_densenet_mcyt = optim.Adam(model_densenet_mcyt.parameters(), lr=0.001)
for epoch in range(num_epochs):
    loss_dense_mcyt, acc_dense_mcyt = train_one_epoch(model_densenet_mcyt, mcyt_train_loader, optimizer_densenet_mcyt, criterion, device)
    print(f"Epoch {epoch+1:2d} - MCYT DenseNet Loss: {loss_dense_mcyt:.4f}, Accuracy: {acc_dense_mcyt*100:.2f}%")