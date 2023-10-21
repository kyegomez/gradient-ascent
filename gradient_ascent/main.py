import torch


class GradientAscent:
    """
    Gradient Ascent Optimizer

    Optimizer that performs gradient ascent on the parameters of the model.

    Args:
        parameters (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 0.01)
        momentum (float, optional): momentum factor (default: 0.9)
        beta (float, optional): beta factor (default: 0.999)
        eps (float, optional): epsilon (default: 1e-8)
        nesterov (bool, optional): enables Nesterov accelerated gradient (default: False)
        clip_value (float, optional): gradient clipping value (default: None)
        lr_decay (float, optional): learning rate decay (default: None)

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
        nesterov=False,
        clip_value=None,
        lr_decay=None,
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.eps = eps
        # Nesterov accelerated gradient NAG => Provides a lookahead in the direction of the parameter updates => optimizer converge faster
        self.nesterov = nesterov
        # Gradient Clipping => Prevents exploding gradients
        self.clip_value = clip_value
        # Learning Rate Decay => Prevents oscillations
        self.lr_decay = lr_decay

        # Initalize momentum and adaptive learning rate
        self.v = {p: torch.zeros_like(p.data) for p in self.parameters}
        self.m = {p: torch.zeros_like(p.data) for p in self.parameters}

    def step(self, closure=None):
        """Step function for gradient ascent optimizer"""
        for param in self.parameters:
            if param.grad is not None:
                if self.clip_value:
                    torch.nn.utils.clip_grad_value_(param.grad, self.clip_value)

                # Nesterov Accelerated Gradient
                if self.nesterov:
                    grad = param.grad + self.momentum * self.v[param]
                else:
                    grad = param.grad

                # Momentum
                self.v[param] = self.momentum * self.v[param] + grad

                # Adaptive learning rate
                self.m[param] = self.beta * self.m[param] + (1 - self.beta) * grad**2
                adapted_lr = self.lr / (torch.sqrt(self.m[param]) + self.eps)

                # Gradient Ascent
                param.data.add_(adapted_lr * self.v[param])

                # Learning Rate Decay
                if self.lr_decay:
                    self.lr *= self.lr_decay

    def zero_grad(self):
        """Zero the gradient of the parameters"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
