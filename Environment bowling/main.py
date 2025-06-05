from train_q_learning import train_q_learning
from train_dqn import train_dqn
from train_ppo import train_ppo
from train_a2c import train_a2c
from compare_algorithms import comparar_algoritmos
import sys

def main():
    print("\n|*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*|")
    print("|               Choose the algoritm you want to run                 |")
    print("|               1. Q-Learning                                       |")
    print("|               2. DQN (Deep Q Network)                             |")
    print("|               3. PPO (Proximal Policy Optimization)               |")
    print("|               4. A2C (Advantage Actor-Critic)                     |")
    print("|               5. Compare all algoritms                            |")
    print("|               0. exit                                             |")
    print("|*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*|")
    print("\n")

    choice = input("Choose one option: ")
    return choice


if __name__ == "__main__":
    while True:
        option = main()

        if option == "1":
            train_q_learning(episodes=150)
        elif option == "2":
            train_dqn(episodes=150)
        elif option == "3":
            train_ppo(episodes=100)     
        elif option == "4":
            train_a2c(episodes=20)
        elif option == "5":
            comparar_algoritmos()
        
        elif option == "0":
            print("|*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*|")
            print("|                                                                   |")
            print("|                              Good bye!                            |")
            print("|                                                                   |")
            print("|*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*|")
            sys.exit()
        else:
            print("invalid option, try again.")