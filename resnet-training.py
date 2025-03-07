# Advanced Model: ResNetSignature using ResNet18
class ResNetSignature(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetSignature, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Modify first conv layer to accept 1 channel (grayscale) instead of 3 channels.
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace final FC layer for our two-class problem.
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# Training loop for ResNetSignature on CEDAR dataset:
print("Training ResNetSignature on CEDAR Dataset")
model_resnet_cedar = ResNetSignature(num_classes=2).to(device)
optimizer_resnet_cedar = optim.Adam(model_resnet_cedar.parameters(), lr=0.001)

for epoch in range(num_epochs):
    loss_resnet, acc_resnet = train_one_epoch(model_resnet_cedar, cedar_train_loader, optimizer_resnet_cedar, criterion, device)
    print(f"Epoch {epoch+1:2d} - CEDAR ResNet Loss: {loss_resnet:.4f}, Accuracy: {acc_resnet*100:.2f}%")


# For GPDS:
print("\nTraining ResNetSignature on GPDS Dataset")
model_resnet_gpds = ResNetSignature(num_classes=2).to(device)
optimizer_resnet_gpds = optim.Adam(model_resnet_gpds.parameters(), lr=0.001)
for epoch in range(num_epochs):
    loss_resnet_gpds, acc_resnet_gpds = train_one_epoch(model_resnet_gpds, gpds_train_loader, optimizer_resnet_gpds, criterion, device)
    print(f"Epoch {epoch+1:2d} - GPDS ResNet Loss: {loss_resnet_gpds:.4f}, Accuracy: {acc_resnet_gpds*100:.2f}%")

    
# For SIGCOMP:
print("\nTraining ResNetSignature on SIGCOMP Dataset")
model_resnet_sigcomp = ResNetSignature(num_classes=2).to(device)
optimizer_resnet_sigcomp = optim.Adam(model_resnet_sigcomp.parameters(), lr=0.001)
for epoch in range(num_epochs):
    loss_resnet_sigcomp, acc_resnet_sigcomp = train_one_epoch(model_resnet_sigcomp, sigcomp_train_loader, optimizer_resnet_sigcomp, criterion, device)
    print(f"Epoch {epoch+1:2d} - SIGCOMP ResNet Loss: {loss_resnet_sigcomp:.4f}, Accuracy: {acc_resnet_sigcomp*100:.2f}%")

# For MCYT:
print("\nTraining ResNetSignature on MCYT Dataset")
model_resnet_mcyt = ResNetSignature(num_classes=2).to(device)
optimizer_resnet_mcyt = optim.Adam(model_resnet_mcyt.parameters(), lr=0.001)
for epoch in range(num_epochs):
    loss_resnet_mcyt, acc_resnet_mcyt = train_one_epoch(model_resnet_mcyt, mcyt_train_loader, optimizer_resnet_mcyt, criterion, device)
    print(f"Epoch {epoch+1:2d} - MCYT ResNet Loss: {loss_resnet_mcyt:.4f}, Accuracy: {acc_resnet_mcyt*100:.2f}%")