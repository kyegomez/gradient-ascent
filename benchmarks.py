import numpy as np
import torch
import torch.nn as nn
from gradient_ascent.main import GradientAscent


# Benchmark 1: Maximize output for simple linear regression
def benchmark_1():
    model = nn.Linear(1, 1)
    optimizer = GradientAscent(model.parameters(), lr=0.01)
    data = torch.tensor([[2.0]])
    target = torch.tensor([[10.0]])
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(data)
        loss = -torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    return model(data).item()

# Benchmark 2: Maximize quadratic function
def benchmark_2():
    x = torch.tensor([1.0], requires_grad=True)
    optimizer = GradientAscent([x])
    for _ in range(1000):
        optimizer.zero_grad()
        y = -x**2
        y.backward()
        optimizer.step()
    return x.item()

# Benchmark 3: Maximize a neural network's output
def benchmark_3():
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
    optimizer = GradientAscent(model.parameters(), lr=0.01)
    data = torch.randn(1, 10)
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(data)
        (-output).backward()
        optimizer.step()
    return output.item()

# Benchmark 4-10: Variations of the above with different complexities and data
# For simplicity, I'll demonstrate just one more, but similar variations can be created.
def benchmark_4():
    model = nn.Sequential(nn.Linear(20, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 1))
    optimizer = GradientAscent(model.parameters(), lr=0.01)
    data = torch.randn(1, 20)
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(data)
        (-output).backward()
        optimizer.step()
    return output.item()


# ...

# Benchmark 5: Maximize a deeper neural network's output
def benchmark_5():
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    optimizer = GradientAscent(model.parameters(), lr=0.01)
    data = torch.randn(1, 10)
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(data)
        (-output).backward()
        optimizer.step()
    return output.item()

# Benchmark 6: Maximize a function with multiple peaks
def benchmark_6():
    x = torch.tensor([0.5], requires_grad=True)
    optimizer = GradientAscent([x], lr=0.01)
    for _ in range(1000):
        optimizer.zero_grad()
        y = -torch.sin(5*np.pi*x) * x
        y.backward()
        optimizer.step()
    return x.item()

# Benchmark 7: Maximize a 2D function
def benchmark_7():
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    optimizer = GradientAscent([x], lr=0.01)
    for _ in range(1000):
        optimizer.zero_grad()
        y = -(x[0]**2 + x[1]**2)
        y.backward()
        optimizer.step()
    return x.tolist()

# Benchmark 8: Maximize output for a convolutional layer
def benchmark_8():
    model = nn.Conv2d(3, 16, 3, 1)
    optimizer = GradientAscent(model.parameters(), lr=0.01)
    data = torch.randn(1, 3, 32, 32)
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(data).mean()
        (-output).backward()
        optimizer.step()
    return output.item()

# Benchmark 9: Maximize output for recurrent neural network layer
def benchmark_9():
    model = nn.RNN(10, 20, 2)
    optimizer = GradientAscent(model.parameters(), lr=0.01)
    data = torch.randn(5, 1, 10) # sequence length of 5
    h0 = torch.randn(2, 1, 20) # initial hidden state
    for _ in range(1000):
        optimizer.zero_grad()
        output, _ = model(data, h0)
        loss = -output.mean()
        loss.backward()
        optimizer.step()
    return loss.item()

# Benchmark 10: Maximize output for an LSTM layer
def benchmark_10():
    model = nn.LSTM(10, 20, 2)
    optimizer = GradientAscent(model.parameters(), lr=0.01)
    data = torch.randn(5, 1, 10)
    h0 = torch.randn(2, 1, 20)
    c0 = torch.randn(2, 1, 20)
    for _ in range(1000):
        optimizer.zero_grad()
        output, _ = model(data, (h0, c0))
        loss = -output[0].mean()
        loss.backward()
        optimizer.step()
    return loss.item()

# Running the benchmarks
if __name__ == '__main__':
    results = {
        "Benchmark 1": benchmark_1(),
        "Benchmark 2": benchmark_2(),
        "Benchmark 3": benchmark_3(),
        "Benchmark 4": benchmark_4(),
        "Benchmark 5": benchmark_5(),
        "Benchmark 6": benchmark_6(),
        "Benchmark 7": benchmark_7(),
        "Benchmark 8": benchmark_8(),
        "Benchmark 9": benchmark_9(),
        "Benchmark 10": benchmark_10(),
    }

    for key, value in results.items():
        print(f"{key}: {value}")
