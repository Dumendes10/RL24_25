
from train import train
from utils.plotting import plot_rewards


def main():
    episode_rewards = train(num_episodes=300)
    plot_rewards(episode_rewards, title="Rainbow DQN – Bowling-v5 Reward Curve", window=10)

if __name__ == "__main__":
    main()
