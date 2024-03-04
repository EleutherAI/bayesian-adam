# bayesian-adam
Exactly what it says on the tin.

This repo provides an implementation of AdaSGHMC which combines [SGHMC](https://arxiv.org/abs/1402.4102) with [Adam](https://arxiv.org/abs/1412.6980). The algorithm samples correctly from the posterior distribution in the limit of alpha -> 0, beta2 -> 1, contains correction factors for uniform diagonal noise, and behaves exactly like adam when the magnitude of the loss -> infinity.

TODO: 
- [ ] Add utilities for deriving sensible priors from transformers
- [ ] Usage instructions
- [ ] Use cases
- [ ] Better test cases
- [ ] Parallel tempering
