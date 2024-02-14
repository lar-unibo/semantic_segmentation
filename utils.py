import torch, random
import model as network
import numpy as np


# Set up model
MODEL_MAP = {
    "resnet50": network.deeplabv3plus_resnet50,
    "resnet101": network.deeplabv3plus_resnet101,
    "swinT": network.deeplabv3plus_swinT,
    "swinS": network.deeplabv3plus_swinS,
    "swinB": network.deeplabv3plus_swinB,
    "swinL": network.deeplabv3plus_swinL,
    "convnextT": network.deeplabv3plus_convnextT,
    "convnextS": network.deeplabv3plus_convnextS,
    "convnextB": network.deeplabv3plus_convnextB,
    "convnextL": network.deeplabv3plus_convnextL,
    "convnextXL": network.deeplabv3plus_convnextXL,
}


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
            for base_lr in self.base_lrs
        ]


class DiceCoeff(torch.autograd.Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
