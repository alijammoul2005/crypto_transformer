"""
Learning rate scheduler with warmup.

Implements the learning rate schedule from "Attention is All You Need":
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch.optim as optim


class WarmupScheduler:
    """
    Learning rate scheduler with linear warmup and inverse square root decay.

    The learning rate increases linearly during the warmup phase, then
    decreases proportional to the inverse square root of the step number.

    Formula:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """
        Update learning rate and step counter.
        """
        self.current_step += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """
        Calculate learning rate for current step.

        Returns:
            Current learning rate
        """
        step = self.current_step
        if step == 0:
            step = 1  # Avoid division by zero

        # Formula from "Attention is All You Need"
        lr = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

        return lr

    def get_last_lr(self):
        """
        Get current learning rate (for compatibility with PyTorch schedulers).

        Returns:
            List containing current learning rate
        """
        return [self.get_lr()]


def test_scheduler():
    """
    Test function for warmup scheduler.
    """
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0)

    # Create scheduler
    d_model = 256
    warmup_steps = 4000
    scheduler = WarmupScheduler(optimizer, d_model, warmup_steps)

    # Track learning rates
    learning_rates = []
    steps = []

    # Simulate training steps
    for step in range(20000):
        scheduler.step()
        lr = scheduler.get_lr()
        learning_rates.append(lr)
        steps.append(step + 1)

    # Plot learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(steps, learning_rates)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title(f'Warmup Learning Rate Schedule (d_model={d_model}, warmup_steps={warmup_steps})')
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    print("Learning rate schedule saved to lr_schedule.png")

    # Print some sample learning rates
    print(f"\nSample learning rates:")
    for step in [1, 100, 1000, 4000, 8000, 16000]:
        idx = step - 1
        print(f"Step {step:5d}: lr = {learning_rates[idx]:.6f}")


if __name__ == '__main__':
    test_scheduler()
