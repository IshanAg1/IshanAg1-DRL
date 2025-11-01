# 🚀 **Autonomous Rocket Lander using Deep Reinforcement Learning (PPO)**  

### 🧠 *Summer Internship Project – VNIT Nagpur (2025)*  

Trained a reinforcement learning agent to achieve **stable, fuel-efficient vertical landings** using the **Proximal Policy Optimization (PPO)** algorithm in a **custom simulation environment** built with **OpenAI Gym** and **Box2D** physics.  

---

## 🧩 **Table of Contents**
- [Overview](#overview)
- [Environment Description](#environment-description)
- [Action & State Spaces](#action--state-spaces)
- [Reward Function](#reward-function)
- [Neural Network Architecture](#neural-network-architecture)
- [Training & Hyperparameters](#training--hyperparameters)
- [Results](#results)
- [Challenges & Future Work](#challenges--future-work)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## 🌍 **Overview**

This project focuses on developing a **Deep Reinforcement Learning (DRL)** agent capable of performing **soft landings** on both **stationary and moving barges**.  
The PPO algorithm was selected for its **stability, ease of implementation, and performance** in **continuous control environments**.  

The objective is to **minimize crash probability and fuel consumption** while maintaining a **smooth, upright, and precise touchdown** under dynamic conditions.  

---

## 🌌 **Environment Description**

The custom environment simulates a 2D rocket controlled via thrust and torque.  
It provides continuous feedback about position, velocity, and contact status.

| Observation Index | Description |
|--------------------|-------------|
| 0 | Horizontal Position |
| 1 | Vertical Position |
| 2 | Horizontal Velocity |
| 3 | Vertical Velocity |
| 4 | Angle (radians) |
| 5 | Angular Velocity |
| 6 | Left Leg Contact (1/0) |
| 7 | Right Leg Contact (1/0) |

**Termination Conditions:**
- ✅ Landed successfully  
- 💥 Crashed or out of bounds  
- ⛽ Fuel exhausted  

---

## 🎮 **Action & State Spaces**

| Action | Range | Description |
|---------|--------|-------------|
| Main Engine Throttle | [0.0, 1.0] | Controls vertical lift |
| Side Engine Thrust | [-1.0, 1.0] | Controls lateral movement |
| Nozzle Angle | [-1.0, 1.0] | Adjusts rocket orientation |

The **state space** is continuous (8-dimensional), allowing the policy to observe smooth dynamics of flight.

---

## 🧮 **Reward Function**

The reward function encourages **safe, stable, and fuel-efficient landings**.  

\[
R_t = \frac{(S_t - S_{t-1})}{10} - 0.3 \times (\text{main\_power} + \text{side\_power})
\]

**Terminal Rewards:**
- **+10** → Landed safely  
- **−10** → Crashed / Out of bounds  

**Penalties:**
- Upward velocity → −1  
- Fuel usage → proportional penalty  

This formulation uses a **potential-based shaping function** to stabilize learning and ensure smooth policy convergence.

---

## 🧠 **Neural Network Architecture**

| Network | Layers | Activation |
|----------|---------|-------------|
| Actor (Policy) | [256, 256] | ReLU |
| Critic (Value) | [256, 256] | ReLU |

**Input:** 8-dimensional state vector  
**Outputs:**  
- Actor → 3 continuous actions (thrust, side, rotation)  
- Critic → scalar value \(V(s)\)  

<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20250224190459513673/Actor-Critic-Method.webp" width="550">
</p>

The **Actor–Critic** structure ensures that the policy learns **continuous control commands** while maintaining **value-based stability**.

---

## ⚙️ **Training & Hyperparameters**

| Parameter | Value | Description |
|------------|--------|-------------|
| `learning_rate` | 0.0001 | Step size for updating network weights |
| `n_steps` | 1024 | Steps collected before each policy update |
| `n_epochs` | 10 | Gradient passes per update |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | Bias–variance trade-off for advantage estimation |
| `clip_range` | 0.2 | PPO clipping threshold |
| `ent_coef` | 0.01 | Encourages exploration |
| `batch_size` | 64 | Samples per gradient step |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping to ensure stability |

---

## 📈 **Results**

<!-- <p align="center">
  <img src="https://pylessons.com/media/Tutorials/Reinforcement-learning-tutorial/LunarLander-v2-PPO/1.png" width="480"><br>
  <em>Training Curve: Average Episode Reward and Length (TensorBoard Visualization)</em>
</p> -->

<p align="center">
  <img src="l4 - Made with Clipchamp_1761941731693.gif" width="550">
  <br>
  <em>Final Trained Rocket Landing Simulation using PPO</em>
</p>


**Performance Summary:**
- Mean episodic reward improved from **−8.5 → +8.9**  
- Average episode length increased → **rocket survives longer**  
- Final landing success rate: **≈ 91 %**  
- Average fuel consumption reduced by **~50 %**  
- Policy achieves **smooth & stable descent** with minimal oscillation  

---

## ⚠️ **Challenges & Future Work**

**Challenges**
- High reward variance during early training → unstable convergence  
- Balancing fuel penalty vs. landing reward  
- Random initial states causing inconsistent learning  
- Long training times due to small learning rate  
- Occasional policy collapse when critic became over-confident  

**Future Work**
- Extend environment to **3D landing dynamics**  
- Add **wind and random turbulence** for robustness  
- Implement **domain randomization** for real-world transfer  
- Explore **multi-agent coordination** for multiple landers  
- Use **reward normalization** and **adaptive clipping** for stability  

---

## 💻 **Usage**

To train and evaluate the PPO agent:  

```bash
git clone https://github.com/<your-username>/rocket-lander-drl.git
cd rocket-lander-drl
pip install -r requirements.txt
python train_and_eval.py
