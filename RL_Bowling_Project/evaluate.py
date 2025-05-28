
import torch
from agents.rainbow_dqn_agent import RainbowDQN
from environment.bowling_env import make_bowling_env
from train import preprocess

def evaluate(model_path, num_episodes=5):
    env = make_bowling_env(render_mode="human")
    obs, _ = env.reset()
    input_shape = preprocess(obs).shape[1:]
    n_actions = env.action_space.n

    policy_net = RainbowDQN(input_shape, n_actions)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                q_values = policy_net(state)
                action = q_values.argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = preprocess(next_obs)
            total_reward += reward

        print(f"Episode {episode + 1} — Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    evaluate("rainbow_model.pth")
