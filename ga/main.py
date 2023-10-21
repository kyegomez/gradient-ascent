import torch


class GradientAscent(torch.nn.optim.Optimizer):
    """
    Gradient Ascent Optimizer

    Optimizer that performs gradient ascent on the parameters of the model.

    Args:
        parameters (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 0.01)
        momentum (float, optional): momentum factor (default: 0.9)
        beta (float, optional): beta factor (default: 0.999)
        eps (float, optional): epsilon (default: 1e-8)

    Attributes:
        defaults (dict): default optimization options
        parameters (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        momentum (float): momentum factor
        beta (float): beta factor
        eps (float): epsilon
        v (dict): momentum
        m (dict): adaptive learning rate

    Example:
        >>> optimizer = GradientAscent(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()




    """

    def __init__(
        self,
        parameters,
        lr=0.01,
        momentum=0.9,
        beta=0.999,
        eps=1e-8,
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.eps = eps

        # Initalize momentum and adaptive learning rate
        self.v = {p: torch.zeros_like(p.data) for p in self.parameters}
        self.m = {p: torch.zeros_like(p.data) for p in self.parameters}

    def step(self, closure=None):
        """Step function for gradient ascent optimizer"""
        for param in self.parameters:
            if param.grad is not None:
                # Momentum
                self.v[param] = (
                    self.momentum * self.v[param]
                    + (1.0 - self.monentum) * param.grad.data
                )

                # Adaptive learning rate
                self.m[param] = (
                    self.beta * self.m[param] + (1.0 - self.beta) * param.grad.data**2
                )
                adapted_lr = self.lr / (torch.sqrt(self.m[param]) + self.eps)

                # Gradient Ascent
                param.data.add_(adapted_lr * self.v[param])

    def zero_grad(self):
        """Zero the gradient of the parameters"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
