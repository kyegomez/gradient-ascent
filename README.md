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
Alright, let's build this step-by-step, focusing on the gradient ascent optimizer. 

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



# UseCases:
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

# License
MIT

