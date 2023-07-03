import torch
import torchvision.models as models
import torch.nn as nn


PATH_TO_MODEL = "models/Res50_lr=0.0005_batchSize=10"   #omit the .pth extension

def load(path):
    print("Loading from ", path)
    net = models.resnet50(weights='DEFAULT')
    net.fc = nn.Linear(net.fc.in_features, 4)
    net.load_state_dict(torch.load(path))    
    print("Loaded")
    return net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = load(PATH_TO_MODEL + ".pth")
model.eval()
dummy_input = torch.randn(1, 3, 128, 128)
output_names = [ "output" ]

torch.onnx.export(model,
                 dummy_input,
                 PATH_TO_MODEL + ".onnx",
                 verbose=False,
                 keep_initializers_as_inputs=True,
                 export_params=True,
                 )