---
title: Sample Efficient Accuracy Estimation
tags: [Model Evaluation]
style: 
color: 
description: In high-stakes ML applications where the cost of labelling is expensive, it is imperative to perform model monitoring in sample efficient way. 
---

Consider a scenario where we are only given a small labelling budget $M$  ($< N$), we employ the technique presented in [Yilmaz et.al., 2021](https://arxiv.org/pdf/2109.12043.pdf) to sample data points which contribute more to any given test statistic. There are two sampling approaches presented in the above paper, currently we focus on using the importance sampling approach. In short, the importance sampling approach samples those points which contribute more to the squared error between the finite $M$ estimator $\hat{F}$ and infinite $M$ limit $F^\prime.$ Clearly, we do not have access to the true underlying posterior distribution, to mitigate this the paper suggests approximations. 

## Theory
In this specific paper [[Yilmaz et.al., 2021]](https://arxiv.org/pdf/2109.12043.pdf), the authors consider metrics of the form: 
$$F = \frac{\sum_{n=1}^{N} f(c_p, c_t)}{\sum_{n=1}^{N} g(c_p, c_t)}$$

For example, for the accuracy metric we have $f(c_p,c_t)=\mathbb{I}[c_p =c_t], g(c_p,c_t)=1$. Here $\mathbb{I}[x=y]$ is the indicator function. The paper forms the estimator: 
$$\hat{F} = \frac{\hat{x}}{\hat{y}}$$

Here, for a chosen $M$ samples, we define 
$$\hat{x} = \frac{1}{MN}\sum_{i = 1}^{M}\frac{f_{m_i}}{q_{m_i}} \hspace{2em} \hat{y} = \frac{1}{MN}\sum_{i = 1}^{M}\frac{g_{m_i}}{q_{m_i}}$$

The importance distribution is given by $q_n, n \in \{ 1, . . . , N \}$. The optimal sampling distribution is then proportional to The squared error between the finite   $M$ estimator $\hat{F}$ and infinite $M$ limit $F^\prime.$ 

$$\mathbb{E}\left[(\hat{F} - F^\prime)^2\right] \propto \frac{h_n^2}{q_n}\,,$$

where, $h_n² \propto \left \langle \left (\frac{f_n}{g_n} − F^′\right)^2\right \rangle = (c^t_n =1|x_n)(f(c^p_n,c^t_n =1)−F^′g(c^p_n,c^t_n =1))2
+p(c^t_n=0|x_n)(f(c^p_n,c^t_n=0)−F^′g(c^p_n,c^t_n=0)^2$. To calculate the optimal sampler, we therefore need to know the true class distribution $p(c^t_n = 1|x_n)$. In other works the assumption $p(c^t_n = 1|x_n) = p(c^p_n = 1|x_n)$ is made. However, this places great faith in the model and can lead to overconfidence, particularly in models with very high or very low probabilities. To address this the authors replace the unknown $p(c^t_n|x_n)$ with an estimate, 

$$p^t_a(c_n = 1|x_n) = \lambda p(c^p_n = 1|x_n) + (1 − \lambda)0.5 \,,$$

for some user chosen $0 \leq \lambda \leq 1$.

We use this to form an approximation to $F^′$ by computing the expectation with respect to $p_a(c^t_n|x_n)$. We denote this by:
```math
F_a^′ =  \frac{\sum_{n=1}^{N} \mathbb{E}_{p_a} [f_n]}{\sum_{n=1}^{N} \mathbb{E}_{p_a} [g_n]}
```

This approximation can be used in place of the true $F^\prime$. This enables us to fully define $h_n$ and the optimal Importance Sampler $q_n \propto |h_n|$.
