import torch
from torch import nn
from gradient_ascent.main import GradientAscent


class Quadratic(nn.Module):
    def __init__(self):
        super(Quadratic, self).__init__()
        self.x = nn.Parameter(torch.tensor([1.0]))

    def forward(self):
        return -self.x**2


model = Quadratic()


# Define the optimizer
optimizer = GradientAscent(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    beta=0.99,
    eps=1e-8,
    nesterov=True,
    clip_value=1.0,
    lr_decay=0.98,
    warmup_steps=5,
    logging_interval=2,
)


# Optimization loop
num_epochs = 30
for epoch in range(num_epochs):
    # Forward pass
    output = model()

    # Zero gradients
    optimizer.zero_grad()

    # Negative of backward since we're doing ascent, not descent
    (-output).backward()

    # Update parameters
    optimizer.step()

    # Print updates
    print(
        f"Epoch {epoch+1}/{num_epochs}, x: {model.x.item():.4f}, f(x): {output.item():.4f}"
    )

# You should observe 'x' moving towards 0 (the maximum of our function)
# and f(x) moving towards its maximum value, which is 0.
# Additionally, the logging interval will print the learning rate and gradient norm every 2 steps.
