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
optimizer = GradientAscent(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    beta=0.99,
    # nesterov=True,
    # clip_value=2.0,
    # lr_decay=0.98,
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
