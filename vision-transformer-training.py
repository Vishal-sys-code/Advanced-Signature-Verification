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