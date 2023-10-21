import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from gradient_ascent import GradientAscent
from gradient_ascent.main import GradientAscent


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


# Set up real-time plotting
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1000)  # Assuming 1000 epochs
ax.set_ylim(0, 10)  # Arbitrary y-axis limits for visualization purposes
ax.set_title("Model Output vs. Target during Training")
ax.set_xlabel("Epoch")
ax.set_ylabel("Value")
(line_output,) = ax.plot([], [], "r-", label="Model Output")
(line_target,) = ax.plot([], [], "g-", label="Target Value")
ax.legend()


# Initialization function for the animation
def init():
    line_output.set_data([], [])
    line_target.set_data([], [])
    return line_output, line_target


def update(epoch, model_output_value):
    x_data_output, y_data_output = line_output.get_data()
    x_data_target, y_data_target = line_target.get_data()

    # Convert numpy arrays to lists only if they aren't already lists
    if not isinstance(x_data_output, list):
        x_data_output = x_data_output.tolist()
    if not isinstance(y_data_output, list):
        y_data_output = y_data_output.tolist()
    if not isinstance(x_data_target, list):
        x_data_target = x_data_target.tolist()
    if not isinstance(y_data_target, list):
        y_data_target = y_data_target.tolist()

    # Append new data
    x_data_output.append(epoch)
    y_data_output.append(model_output_value)
    x_data_target.append(epoch)
    y_data_target.append(target.item())

    line_output.set_data(x_data_output, y_data_output)
    line_target.set_data(x_data_target, y_data_target)
    fig.canvas.flush_events()
    return line_output, line_target


# Test the optimizer
model = SimpleModel()
# Define the optimizer
optimizer = GradientAscent(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    beta=0.99,
    eps=1e-8,
    # Add or remove optional parameters as needed
)

# Generate some sample data
data = torch.tensor([[2.0]])
target = torch.tensor([[9.0]])

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(data)
    # Negative loss as we are maximizing
    loss = -torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    # Update the visualization
    update(epoch, output.item())

print("Final output after training: ", model(data))
plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
