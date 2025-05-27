# 🎯 Algorithms Implemented

This project implements and compares three reinforcement learning approaches on the **PandaReach-v3** robotic manipulation environment:

---

### 1. Soft Actor-Critic (SAC)
- Off-policy actor-critic algorithm
- Entropy regularization for improved exploration and stability

### 2. Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Deterministic policy gradient method
- Twin critics and delayed updates to reduce overestimation bias

### 3. Hindsight Experience Replay (HER)
- Goal relabeling technique for sparse reward environments
- Treats failed episodes as successful by redefining goals

---

## 🔧 Implementation Variants

The project evaluates the following configurations:

- **SAC (baseline)**: Standard SAC implementation  
- **TD3 (baseline)**: Standard TD3 implementation  
- **SAC + HER**: SAC enhanced with Hindsight Experience Replay  
- **TD3 + HER**: TD3 enhanced with Hindsight Experience Replay  

---

## ⚙️ HER Configuration Options

### Goal Selection Strategies:
- **Final Strategy**: Uses the final achieved state as the relabeled goal  
- **Future Strategy**: Samples goals from future states within the same episode  

### Future Strategy Parameters:
- **`k` parameter**: Controls the number of future goals sampled per transition  
- **Tested values**: `k = 1, 4, 8, 16`  
- Higher `k` values provide more diverse learning experiences but increase computational cost

