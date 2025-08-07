import torch


class KATE(torch.optim.Optimizer):  # delta 0 or 1e-8
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
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data, device=group["device"])
                    state["b"] = torch.zeros_like(p.data, device=group["device"])

                m, b = state["m"], state["b"]
                eta = group["eta"]

                if group["weight_decay"] != 0:
                    grad = grad + group["weight_decay"] * p.data

                g = grad * grad

                b = b + g
                denom = b + group["eps"]
                m = m + torch.mul(eta, g) + g / denom

                lr = group["lr"]

                p.data = p.data - lr * torch.sqrt(m) * grad / denom

                # Save state
                state["m"], state["b"] = m, b
                state["step"] += 1

        return loss
