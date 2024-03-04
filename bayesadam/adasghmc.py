
import torch
from torch.optim.optimizer import Optimizer


class AdaSGHMC(Optimizer):
    def __init__(self, 
        params, learning_rate=1e-3, gradient_ema=0.99, momentum=0.9, eps=1e-8):
    
        self._validate_inputs(learning_rate, gradient_ema, eps, momentum)

        defaults = dict(learning_rate=learning_rate, momentum=momentum, gradient_ema=gradient_ema, eps=eps)

        super().__init__(params, defaults)
        for group in self.param_groups:
            self._init_group(group)
    
    def _validate_inputs(self, learning_rate, momentum, gradient_ema, eps):
        if not 0.0 <= learning_rate:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if not 1.0 > momentum > 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= gradient_ema <= 1.0:
            raise ValueError(f"Invalid ema_fraction: {gradient_ema}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")

    def _init_group(self, group):
        for p in group["params"]:
            state =  self.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros((), dtype=torch.float32)
                #Average gradient value squared
                state["exp_g_avg_sq"] = torch.zeros_like(p)
                #Normalized Momentum
                state["normalized_momentum"] = torch.zeros_like(p)
                #Average gradient difference value squared
                state["exp_ngd_avg_sq"] = torch.zeros_like(p)
                #Previous Grad
                state["previous_grad"] = torch.zeros_like(p)

    def _update_state_and_parameters(self, group):
        values = []
        for p in group["params"]:
            if p.grad is None:
                continue
            param = p
            grad = p.grad
            step = self.state[p]["step"]
            exp_g_avg_sq = self.state[p]["exp_g_avg_sq"]
            normalized_momentum = self.state[p]["normalized_momentum"]
            exp_ngd_avg_sq = self.state[p]["exp_ngd_avg_sq"]
            previous_grad = self.state[p]["previous_grad"]

            learning_rate = group["learning_rate"]
            momentum = group["momentum"]
            gradient_ema = group["gradient_ema"]
            eps = group["eps"]

            self._update(param, grad, step, 
                        exp_g_avg_sq, normalized_momentum, 
                        exp_ngd_avg_sq, previous_grad,
                        learning_rate, momentum, gradient_ema,
                        eps)
        
    
    def _update(self, param, grad, step, 
                exp_g_avg_sq, normalized_momentum, 
                exp_ngd_avg_sq, previous_grad,
                learning_rate, momentum, gradient_ema,
                eps):
        param_value = param.detach()
        
        step += 1
        
        exp_g_avg_sq *= gradient_ema
        exp_g_avg_sq += (1-gradient_ema)*grad**2

        second_moment_estimate = exp_g_avg_sq/(1-gradient_ema**step)

        scale = torch.sqrt(second_moment_estimate) + eps
        friction_factor = ((1-momentum)/(learning_rate*scale))**0.5
        step_size = (1-momentum)/friction_factor

        normalized_grad = grad/scale
        normalized_prev_grad = previous_grad/scale
        norm_grad_diff = normalized_grad-normalized_prev_grad
        
        exp_ngd_avg_sq *= (gradient_ema)
        exp_ngd_avg_sq += (1-gradient_ema)*norm_grad_diff**2

        g_variance = exp_ngd_avg_sq/2
        noise_estimate = (1/2)*step_size*g_variance
        
        noise_scale = torch.sqrt(
            2*torch.max(torch.tensor(0.), friction_factor-noise_estimate)*step_size)
        
        noise = torch.randn_like(param_value)*noise_scale

        normalized_momentum -= \
            step_size*normalized_grad + \
            step_size*friction_factor*normalized_momentum + \
            noise

        normalized_delta = step_size*normalized_momentum

        previous_grad *= 0
        previous_grad += grad
        
        param.data += normalized_delta/scale

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            self._update_state_and_parameters(group)
        
        return loss


def construct_prior(model):
    """construct a prior based on the distibution of weights in a model"""
    prior = [] 
    for p in model.parameters():
        if torch.numel(p) >= 1:
            mean = torch.mean(p)
            std = torch.std(p)
        else:
            mean = torch.squeeze(p)
            std = torch.max(torch.squeeze(p),1)
        prior.append(mean,std)
    
    return prior
        

def evaluate_prior(model, prior):
    neg_log_p = 0
    for i,p in enumerate(model.parameters()):
        mean,std = prior[i] 
        z = (p-mean)/std
        neg_log_p += torch.sum(z**2)/2
    return neg_log_p


