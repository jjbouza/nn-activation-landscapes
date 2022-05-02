import torch
import torch.nn as nn

_EPS_ = 1e-6
class MSELossClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = torch.nn.MSELoss()
    def forward(self, input, target):
        input_ = input
        target_ = torch.nn.functional.one_hot(target, input.shape[1]).float()
        return self.MSE(input_, target_)

class SphereLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sm = nn.Softmax(dim=1)

    def to_sphere(self, x):
        """
        Map x to hypersphere.
        """
        y = torch.sqrt(torch.clamp(x, min=0))
        return x 
    
    def sphere_geodesic_distance(self, x, y):
        """
        Compute acos(<x,y>), the arc length on the sphere.
        """
        z = torch.clamp(torch.sum(x*y, dim=-1)[..., None], min=_EPS_-1, max=1-_EPS_)
        return torch.acos(z)

    def forward(self, input, target):
        input_ = self.sm(input)
        target_ = torch.nn.functional.one_hot(target, num_classes=input.shape[1]).float()
        output = self.sphere_geodesic_distance(self.to_sphere(input_), self.to_sphere(target_))[0,0]
        return output


if __name__ == '__main__':
    x = torch.tensor([[-1,1,-1.0]])
    y = torch.tensor([0])

    sl = SphereLoss()
    print(sl(x,y))
