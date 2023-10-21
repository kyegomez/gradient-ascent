import torch
from gradient_ascent.main import GradientAscent


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


# Test the optimizer
model = SimpleModel()
# Define the optimizer
optimizer = GradientAscent(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    beta=0.99,
    eps=1e-8,
    # nesterov=True, # Nesterov makes the gradient go to nan with warmup steps for some reason so turn of
    # clip_value=1.0,
    # lr_decay=0.98,
    # warmup_steps=5,
    # logging_interval=2,
)


# General some sample data
data = torch.tensor([[2.0]])
target = torch.tensor([[3.0]])

for _ in range(1000):
    optimizer.zero_grad()
    output = model(data)

    # Negative loss as we are maximizing]
    loss = -torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

print("Final output after training: ", model(data))
