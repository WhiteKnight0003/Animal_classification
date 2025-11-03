import torch
import torch.nn as nn
from torchvision import models # <--- Import thư viện models

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, freeze_features=True):
        """
        Khởi tạo mô hình EfficientNet pre-trained.
        
        Args:
            num_classes (int): Số lượng lớp đầu ra (ví dụ: 10 cho Animals-10).
            freeze_features (bool): Nếu True, đóng băng các lớp pre-trained.
        """
        super().__init__()
        
        # 1. Tải mô hình EfficientNet-B0 với trọng số pre-trained "DEFAULT"
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = models.efficientnet_b0(weights=weights)

        # 2. Đóng băng (Freeze) các lớp đặc trưng (features)
        if freeze_features:
            for param in self.efficientnet.features.parameters():
                param.requires_grad = False

        # 3. Lấy số lượng đặc trưng đầu vào từ lớp classifier gốc
        in_features = self.efficientnet.classifier[1].in_features
    
        # Chúng ta giữ lại nn.Dropout(p=0.2) và thay thế nn.Linear
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

if __name__ =='__main__':
    # 1. Tải các phép biến đổi (transforms) CHUẨN cho model này
    weights = models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()
    
    print("--- Transforms chuẩn cho model ---")
    print(auto_transforms)
    print("---------------------------------")

    # 2. Khởi tạo mô hình
    model = EfficientNetClassifier(num_classes=10)
    
    # 3. Tạo input giả lập
    # Kích thước ảnh (ví dụ: 8 ảnh, 3 kênh, 224x224)
    # Kích thước 224x224 là kích thước ảnh chuẩn của B0
    input_tensor = torch.randn(8, 3, 224, 224) 

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # 4. Chạy thử
    output = model(input_tensor)   
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}") # Sẽ là [8, 10]