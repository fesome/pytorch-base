import torch
from torch import nn
import skimage.io
import numpy as np
from torchvision.transforms import ToTensor, Lambda, Compose

#1. build a NN
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(640*640*3,512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
            
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

 
#2. build function for inference
def inference(img, model, loss_fn, device):
    num_batches = 100
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i in range(num_batches):
            x = np.float32(img/255.0)
            x = np.expand_dims(x, axis=-1) if len(x.shape) == 2 else x
            x = np.transpose(x,(2,0,1)) #CHW-format
            X = torch.from_numpy(np.expand_dims(x, axis=0)).to(device) 
            y = torch.from_numpy(np.array([0]).astype(np.int64)).to(device)
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        correct /= num_batches
        print(f"Test Error: \n Accucay: {(100*correct):>0.1f}%, avg. Loss: {test_loss:>8f}\n")           

        
if __name__ == '__main__':

    #1. create a NN model 
    device = 'cuda' if torch.cuda.is_available else "cpu"  
	print(device)
    loss_fn = nn.CrossEntropyLoss()       
    model = NeuralNetwork().to(device)
    print(model)
    
    print('\nstart demo inference ...')
    img = skimage.io.imread('test.jpg')
    inference(img, model, loss_fn, device)