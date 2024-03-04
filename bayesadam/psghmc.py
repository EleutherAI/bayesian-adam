
import torch
from torch.optim.optimizer import Optimizer, required


class PSGHMC(Optimizer):
    def __init__(self, 
        params, step_size=0.01, ema_fraction_a=0.1, eps_a=1e-8, 
        ema_fraction_b=0.1, eps_b=1e-8, friction_factor=0.1):
    
        self._validate_inputs(step_size, ema_fraction_a, eps_a, 
                              ema_fraction_b, eps_b, friction_factor)

        defaults = dict(step_size=step_size, 
                        ema_fraction_a=ema_fraction_a, eps_a=eps_a,
                        ema_fraction_b=ema_fraction_b, eps_b=eps_b,
                        friction_factor=friction_factor)

        super().__init__(params, defaults)
        for group in self.param_groups:
            self._init_group(group)
    
    def _validate_inputs(self, step_size, ema_fraction_a, eps_a,
                         ema_fraction_b, eps_b, friction_factor):
        if not 0.0 <= step_size:
            raise ValueError(f"Invalid step_size: {step_size}")
        if not 0.0 <= ema_fraction_a <= 1.0:
            raise ValueError(f"Invalid ema_fraction: {step_size}")
        if not 0.0 <= ema_fraction_b <= 1.0:
            raise ValueError(f"Invalid ema_fraction: {step_size}")
        if not 0.0 <= eps_a:
            raise ValueError(f"Invalid eps: {eps_a}")
        if not 0.0 <= eps_b:
            raise ValueError(f"Invalid eps: {eps_b}")
        if not 1.0 > friction_factor*step_size:
            raise ValueError(f"Invalid friction_factor: {friction_factor}")

    def _init_group(self, group):
        for p in group["params"]:
            state =  self.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros((), dtype=torch.float32)
                #Average parameter value
                state["exp_p_avg"] = torch.zeros_like(p)
                #Average parameter value squared
                state["exp_p_avg_sq"] = torch.zeros_like(p)
                #Average gradient difference value squared
                state["exp_ngd_avg_sq"] = torch.zeros_like(p)
                #Normalized Momentum
                state["normalized_momentum"] = torch.zeros_like(p)
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
            exp_p_avg = self.state[p]["exp_p_avg"]
            exp_p_avg_sq = self.state[p]["exp_p_avg_sq"]
            normalized_momentum = self.state[p]["normalized_momentum"]
            exp_ngd_avg_sq = self.state[p]["exp_ngd_avg_sq"]
            previous_grad = self.state[p]["previous_grad"]

            step_size = group["step_size"]
            ema_fraction_a = group["ema_fraction_a"]
            eps_a = group["eps_a"]
            ema_fraction_b = group["ema_fraction_b"]
            eps_b = group["eps_b"]
            friction_factor = group["friction_factor"]

            self._update(param, grad, step, 
                        exp_p_avg, exp_p_avg_sq, 
                        normalized_momentum, exp_ngd_avg_sq, 
                        previous_grad,
                        step_size, ema_fraction_a, eps_a,
                        ema_fraction_b, eps_b, friction_factor)
        
    
    def _update(self, param, grad, step, 
                exp_p_avg, exp_p_avg_sq, 
                normalized_momentum, exp_ngd_avg_sq,
                previous_grad, 
                step_size, ema_fraction_a, eps_a,
                ema_fraction_b, eps_b, friction_factor):
        param_value = param.detach()
        
        ema_timescale_a = max((1 + step)*ema_fraction_a, 1)
        ema_factor_a = 1/ema_timescale_a

        exp_p_avg *= (1-ema_factor_a)
        exp_p_avg += (ema_factor_a)*param_value

        exp_p_avg_sq *= (1-ema_factor_a)
        exp_p_avg_sq += (ema_factor_a)*param_value**2

        #TODO: Improve variance estimate numerical stabilitiy
        p_variance = torch.maximum(exp_p_avg_sq-exp_p_avg**2,torch.tensor(0))
        
        scale = torch.sqrt(p_variance+eps_a)

        normalized_grad = grad*scale
        normalized_prev_grad = previous_grad*scale
        norm_grad_diff = normalized_grad-normalized_prev_grad
        
        ema_timescale_b = max((1 + step)*ema_fraction_b, 1)
        ema_factor_b = 1/ema_timescale_b

        exp_ngd_avg_sq *= (1-ema_factor_b)
        exp_ngd_avg_sq += (ema_factor_b)*norm_grad_diff**2

        g_variance = exp_ngd_avg_sq/2 + eps_b
        noise_estimate = (1/2)*step_size*g_variance
        
        noise_scale = torch.sqrt(
            2*torch.max(torch.tensor(0.), friction_factor-noise_estimate)*step_size)
        
        noise = torch.randn_like(param_value)*noise_scale

        normalized_momentum -= \
            step_size*normalized_grad + \
            step_size*friction_factor*normalized_momentum + \
            noise

        delta = step_size*normalized_momentum

        
        param.data += delta*scale

        previous_grad *= 0
        previous_grad += grad

        step += 1

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


