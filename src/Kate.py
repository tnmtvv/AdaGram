import torch

class KATE(torch.optim.Optimizer):  # delta 0 or 1e-8
    """
    Original implementation from the https://github.com/nazya/KATE/tree/main
    """


    def __init__(
        self,
        params,
        lr,
        eta=0.9,
        eps=1e-8,
        weight_decay=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):  # lr=1e-3, eta=0.9, eps=1e-8, delta=0, weight_decay=0):
        defaults = dict(
            device=device,
            lr=lr,
            eta=eta,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # All tensors will be created on the same device as the parameter 'p'
                device = p.device
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('This optimizer does not support sparse gradients.')
                    
                state = self.state[p]
        
                # State initialization: Use p.device to ensure consistency
                if len(state) == 0:
                    state["step"] = 0
                    # --- CRITICAL FIX ---
                    # Create state tensors on the same device as the parameter
                    state["m"] = torch.zeros_like(p.data, device=device)
                    state["b"] = torch.zeros_like(p.data, device=device)
        
                m, b = state["m"], state["b"]
                eta = group["eta"]
        
                # All subsequent operations will be on the correct device because
                # 'p', 'grad', 'm', and 'b' are already on the same device.
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])
        
                g = grad.pow(2)
        
                b.add_(g)
                denom = b.add(group["eps"])
                
                # Create a temporary tensor for the division to avoid modifying 'g' in-place
                g_div_denom = g.div(denom)
                m.add_(torch.mul(eta, g)).add_(g_div_denom)
        
                lr = group["lr"]
                
                # Create a temporary sqrt_m tensor
                sqrt_m = torch.sqrt(m)
                update_val = sqrt_m.mul(grad).div(denom)
                
                p.data.add_(update_val, alpha=-lr)
                
                state["step"] += 1
        
        return loss