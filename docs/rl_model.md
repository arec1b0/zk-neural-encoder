# PPO Agent Mathematical Model

## Environment Formalization
The problem of selecting optimal ZK encodings is modeled as a single-step Reinforcement Learning task (Contextual Bandit).
* **State ($s_t$)**: Deterministic features of the ABI variable (e.g., size in bytes, mutability).
* **Action ($a_t$)**: The selected encoding strategy (e.g., `SINGLE_FIELD`, `LIMB_DECOMPOSITION`).
* **Reward ($R_t$)**: The negative estimated constraint cost. $R_t = - \text{cost}(s_t, a_t)$.

## Proximal Policy Optimization (PPO)
To prevent catastrophic policy updates, we use the PPO-Clip objective.

### 1. Advantage Estimation
Since this is a single-step environment without future discounted rewards, the advantage $\hat{A}_t$ is simply the difference between the actual reward and the baseline value predicted by the Critic network:
$$ \hat{A}_t = R_t - V_\theta(s_t) $$

### 2. Surrogate Objective (Actor Loss)
The probability ratio between the new policy and the old policy is defined as:
$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} $$

The clipped surrogate objective function restricts the policy update step:
$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right] $$
Where $\epsilon$ is the clipping hyperparameter (typically $0.2$).

### 3. Value Function (Critic Loss)
The Critic minimizes the mean squared error against the actual reward:
$$ L^{VF}(\theta) = \left( V_\theta(s_t) - R_t \right)^2 $$

### 4. Entropy Bonus
To encourage exploration and prevent premature convergence to sub-optimal encodings, we add an entropy bonus $S$:
$$ S[\pi_\theta](s_t) = -\sum_{a} \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) $$

### 5. Total Objective
The final objective function maximized by the agent combines the components:
$$ L^{PPO}(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) $$
Where $c_1$ and $c_2$ are scaling coefficients for the value loss and entropy bonus, respectively.