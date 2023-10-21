[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Gradient Ascent
Gradient Ascent is just the opposite of Gradient Descent. While Gradient Descent adjusts the parameters in the opposite direction of the gradient to minimize a loss function, Gradient Ascent adjusts the parameters in the direction of the gradient to maximize some objective function.


I got the idea for this while playing basketball, I don't know why or how but this is my attempt to implement it.



# Appreciation
* Lucidrains
* Agorians

# Install
`pip install gradient-ascent`

# Usage
```python
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
optimizer = GradientAscent(model.parameters(), lr=0.01)

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

```

# Architecture

### Theoretical Overview
For a function \( f(\theta) \), the update step in gradient ascent is given by:

\[ \theta_{new} = \theta_{old} + \alpha \nabla f(\theta_{old}) \]

Where:
- \( \theta \) are the parameters.
- \( \alpha \) is the learning rate.
- \( \nabla f(\theta_{old}) \) is the gradient of the function with respect to the parameters.

### Gradient Ascent Pseudocode

```
Algorithm: GradientAscentOptimizer

1. Input: 
   - Objective function f(θ)
   - Initial parameters θ₀
   - Learning rate α
   - Maximum iterations max_iter

2. For iteration = 1 to max_iter:
   a. Compute gradient: ∇θ = gradient of f(θ) w.r.t θ
   b. Update parameters: θ = θ + α * ∇θ

3. Return final parameters θ
```

### 3 New Features
1. Non-Convexity:
Many problems in deep learning involve non-convex optimization landscapes. Gradient ascent, like gradient descent, can get stuck in local maxima when dealing with such landscapes. Adding mechanisms to escape from these local optima can be necessary.

2. Momentum:
Momentum can be integrated to accelerate gradient vectors in any consistent direction, which can help in faster convergence and also in avoiding getting stuck in shallow local maxima.

3. Adaptive Learning Rates:
The learning rate might need to adapt based on the recent history of gradients, allowing the optimization to move faster during the early stages and slow down during fine-tuning. This is seen in optimizers like AdaGrad, RMSProp, and Adam.



# Applications:
The Gradient Ascent with features like momentum and adaptive learning rates, as discussed, is tailored to handle challenges in non-convex optimization landscapes. Here are some tasks and scenarios where this optimizer would be particularly beneficial:

1. **Maximizing Likelihoods**: 
   - Many models in machine learning are framed as maximum likelihood estimation (MLE) problems, where the goal is to adjust parameters to maximize the likelihood of the observed data. Examples include Gaussian Mixture Models, Hidden Markov Models, and some Neural Network configurations. Gradient ascent is directly applicable in such cases.

2. **Generative Adversarial Networks (GANs)**:
   - The generator in a GAN tries to maximize an objective function where it fools the discriminator. While traditional GANs use gradient descent with a flipped objective, using gradient ascent can be a more direct way to express this maximization problem.

3. **Game Theoretic Frameworks**:
   - In scenarios where multiple agents are in competition or cooperation, and their objectives are to maximize certain rewards, gradient ascent can be used. This applies to multi-agent reinforcement learning and certain types of equilibrium-seeking networks.

4. **Policy Gradient Methods in Reinforcement Learning**:
   - Policy gradient methods aim to maximize the expected reward by adjusting a policy in the direction that increases expected returns. Gradient ascent is a natural fit for this optimization problem.

5. **Eigenproblems**:
   - In tasks where the goal is to find the maximum eigenvalue of a matrix or the corresponding eigenvector, gradient ascent techniques can be applied.

6. **Feature Extraction and Representation Learning**:
   - When the goal is to learn features that maximize the variance or mutual information (e.g., Principal Component Analysis or some Information Maximization approaches), gradient ascent can be used to optimize the objective directly.

7. **Sparse Coding**:
   - The aim here is to find a representation that maximizes the sparsity under certain constraints. The problem can be reframed as a maximization problem solvable with gradient ascent.

For scenarios with non-convex landscapes, the features like **momentum** help escape shallow local maxima, and **adaptive learning rates** ensure efficient traversal of the optimization landscape, adapting the step sizes based on the gradient's recent history.

However, while this optimizer can be effective in the above scenarios, one should always consider the specific nuances of the problem. It's essential to remember that no optimizer is universally the best, and empirical testing is often necessary to determine the most effective optimizer for a particular task.

# Benchmarks
`python benchmarks.py`

```
Benchmark 1: 9.999994277954102
Benchmark 2: 1.375625112855263e-23
Benchmark 3: -131395.9375
Benchmark 4: -333186848.0
Benchmark 5: -166376013824.0
Benchmark 6: 0.31278279423713684
Benchmark 7: [1.375625112855263e-23, 1.375625112855263e-23]
Benchmark 8: -28.793724060058594
Benchmark 9: 1.0
Benchmark 10: 0.8203693628311157
```

# Documentation
The `GradientAscent` class is a custom optimizer designed to perform gradient ascent operations. This is especially useful in scenarios where the goal is to maximize an objective function, rather than the more typical gradient descent optimizers which are used to minimize an objective function. Common use cases for gradient ascent include maximizing the likelihood in certain statistical models.

## Features

* Momentum for accelerating the convergence.
* Nesterov accelerated gradient for a lookahead in the direction of parameter updates.
* Gradient Clipping to handle exploding gradients.
* Adaptive learning rates to adjust learning rates based on the magnitude of parameter updates.
* Learning Rate Decay for reducing oscillations.
* Warmup steps for slowly ramping up the learning rate at the beginning of the optimization.

## Class Definition

### GradientAscent

Optimizer that performs gradient ascent on the parameters of a given model.

| Parameter         | Type      | Description                                                                                       | Default Value |
|-------------------|-----------|---------------------------------------------------------------------------------------------------|---------------|
| parameters        | iterable  | Iterable of parameters to optimize or dicts defining parameter groups.                            | None          |
| lr                | float     | Learning rate.                                                                                     | 0.01          |
| momentum          | float     | Momentum factor.                                                                                  | 0.9           |
| beta              | float     | Beta factor used in adaptive learning rate.                                                       | 0.999         |
| eps               | float     | Epsilon to prevent division by zero in the adaptive learning rate computation.                    | 1e-8          |
| nesterov          | bool      | Enables Nesterov accelerated gradient.                                                            | False         |
| clip_value        | float     | Gradient clipping value. If None, no clipping is done.                                            | None          |
| lr_decay          | float     | Learning rate decay factor. If None, no learning rate decay is done.                              | None          |
| warmup_steps      | int       | Number of initial steps during which the learning rate is linearly increased up to its initial value. | 0             |
| logging_interval  | int       | Logging interval in terms of steps to print learning rate and gradient norm.                      | 10            |

### Methods

#### step()
Update function for gradient ascent optimizer. This method should be called after computing the gradients using the `backward()` function.

#### zero_grad()
Resets the gradients of all the parameters to zero.

## Example Usage

```python
import torch.nn as nn
import torch
from gradient_ascent import GradientAscent

# Sample model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc(x)

# Create the model and optimizer
model = SimpleModel()
optimizer = GradientAscent(model.parameters(), lr=0.01)

# Dummy data
data = torch.randn(10, 5)
target = torch.randn(10, 1)

# Loss function
loss_fn = nn.MSELoss()

for epoch in range(100):
    # Forward pass
    output = model(data)
    
    # Calculate loss
    loss = loss_fn(output, target)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward pass
    (-loss).backward() # Negative because we want to maximize
    
    # Update parameters
    optimizer.step()
```

In this example, a simple linear regression model `SimpleModel` is defined with a single fully connected layer. We use the `GradientAscent` optimizer to maximize the negative mean squared error, effectively performing gradient ascent to minimize the loss.

## Tips and Recommendations

* Adjust the learning rate `lr` based on the problem at hand. If the model is not converging or if the updates are too aggressive, reduce the learning rate. Conversely, if the learning is too slow, consider increasing the learning rate.
* The warmup feature can be particularly useful in cases where aggressive updates in the initial stages can destabilize the learning process. Setting an appropriate value for `warmup_steps` can alleviate this.
* Regularly monitor the gradient norms (provided in the logs every `logging_interval` steps) to diagnose potential issues such as exploding or vanishing gradients.

## References and Further Reading
* [Deep Learning Book by Goodfellow et. al. - Chapter 8](http://www.deeplearningbook.org/contents/optimization.html)
* [Why Momentum Really Works](https://distill.pub/2017/momentum/)

---




# License
MIT

# Todo
- Provide metric logging + make more dynamic
- Add more benchmarks
- Validate by training a small Hidden Markov Model or another model