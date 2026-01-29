import torch

class OptParamGenerator:
    def __init__(self, d, device, eps=0.0):
        self.d = d
        self.device = device
        self.eps = eps

    def VQ(self, method, *, theta0=None, cnt=None, m=None):
        V = self.eps * torch.randn(self.d + 2, self.d + 2, device=self.device)
        Q = self.eps * torch.randn(self.d + 2, self.d + 2, device=self.device)

        if method == "pretrain":
            assert m is not None
            V[-1, -1] = 1.0
            Q[1:-1, 1:-1] = torch.eye(self.d, device=self.device) * m / (m + 1 + self.d)
        elif method == "zs_finetune":
            assert theta0 is not None and cnt is not None
            V[-1, 1:-1] = cnt * theta0
            V[-1, -1] = 1.0
            Q[0, 0] = 1.0 / cnt
        elif method == "zs_finetune_v_only":
            assert theta0 is not None and cnt is not None
            V[-1, 1:-1] = theta0 / (self.d + 4.0)
            V[-1, -1] = cnt
            Q[1:-1, 1:-1] = torch.eye(self.d, device=self.device)
        else:
            assert False

        return V, Q

    @torch.no_grad()
    def opt_params(self, model_eval, method, **kw):
        V, Q = self.VQ(method, **kw)
        model_eval.v_matrix.copy_(V)
        model_eval.q_matrix.copy_(Q)
        return model_eval.allparam.detach().clone()
