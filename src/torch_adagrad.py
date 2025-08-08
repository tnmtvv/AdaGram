import torch
from torch.optim import Optimizer
import math

class CustomAdaGrad(Optimizer):
    """
    Implements the standard element-wise AdaGrad algorithm.

    At the end of the step, it replaces p.grad with the preconditioned gradient
    for analysis purposes, as requested.
    """
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0, initial_accumulator_value=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, 
            eps=eps, 
            weight_decay=weight_decay, 
            initial_accumulator_value=initial_accumulator_value
        )
        super(CustomAdaGrad, self).__init__(params, defaults)

        # Initialize the state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                # Initialize the accumulator for squared gradients
                state['sum'] = torch.full_like(
                    p, 
                    group['initial_accumulator_value'], 
                    memory_format=torch.preserve_format
                )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad
                    state = self.state[p]
    
                    state['step'] += 1
    
                    # Optional: Apply weight decay
                    if group['weight_decay'] != 0:
                        grad = grad.add(p, alpha=group['weight_decay'])
    
                    # 1. Update the accumulator with the square of the current gradient
                    # state['sum'] += grad * grad
                    state['sum'].addcmul_(grad, grad, value=1.0)
                    
                    # 2. Calculate the preconditioned gradient
                    # precond_grad = grad / (sqrt(state['sum']) + eps)
                    std = state['sum'].sqrt().add_(group['eps'])
                    preconditioned_grad = grad / std
    
                    # 3. Apply the update to the parameter
                    p.add_(preconditioned_grad, alpha=-group['lr'])
    
                    # 4. As requested: Overwrite the raw gradient with the preconditioned gradient
                    # This is done for analysis and does not affect the next training iteration
                    # because zero_grad() will be called.
                    p.grad.data = preconditioned_grad.reshape(-1)

        return loss