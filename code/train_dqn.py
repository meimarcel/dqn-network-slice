import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

from model_dqn import DQN
from env import Env
from config import *
from utils import *
from log import init_logs


def get_action(model, state, epsilon, device, use_softmax=True):
    with torch.no_grad():
        X = torch.tensor(state).unsqueeze(0).to(torch.float).to(device)
        action_q_values = model.Q_model(X).cpu().detach().numpy().squeeze()

    if use_softmax:
        p = sp.softmax(action_q_values / epsilon).squeeze()
        p /= np.sum(p)

        return np.random.choice(model.num_actions, p=p)
    else:
        if np.random.random() >= epsilon:

            return int(np.argmax(action_q_values, axis=0))
        else:
            return np.random.randint(0, model.num_actions)


def validate(model: DQN, env: Env, action_list: list, device: torch.device):
    env.reset()
    state = env.get_state()

    cumulative_rewards = 0
    actions = []

    for time in range(8):
        action = get_action(model, state, 0, device, False)
        next_state, reward = env.processing(time, action_list[action])

        state = next_state
        if reward < 0:
            break

        actions.append(action_list[action])
        cumulative_rewards += reward

    return cumulative_rewards, actions


def train(output_path, checkpoint_path, start_episode):
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    action_list = generate_actions()

    env = Env()
    model = DQN(num_input=6, num_actions=len(action_list), device=device)

    if checkpoint_path is not None:
        model.load(checkpoint_path, start_episode)

    logs = {}
    states = []
    actions = []
    rewards = []
    energys = []

    plt.ion()

    print("Env RUs:", env.total_ru_services)

    frame = 0

    for episode in range(start_episode, EPISODES + 1):

        epsilon = get_epsilon(episode)

        time = 0
        prev_frame = frame

        env.reset()
        state = env.get_state()

        for _ in range(STEPS):
            action = get_action(model, state, epsilon, device)
            next_state, reward = env.processing(time, action_list[action])

            model.append_to_replay(state, action, reward, next_state)
            model.update()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            energys.append(env.energy_consumption)

            state = next_state

            frame += 1
            time += 1

            if reward < 0 or time == 8:
                break

        # model.plot_loss()

        total_reward, total_actions = validate(model, env, action_list, device)
        cur_state = env.get_state(len(total_actions) - 1)

        print(
            "[VALIDATION] Episode:",
            episode,
            "epsilon:",
            round(epsilon, 2),
            "frames:",
            (frame - prev_frame),
            "time:",
            len(total_actions),
            "reward:",
            total_reward,
            "state:",
            [round(x, 2) if x is not None else None for x in cur_state],
            "action:",
            total_actions,
        )

        if episode % SAVE_WINDOW == 0:
            model.save(output_path, episode)

            logs["state"] = states
            logs["action"] = actions
            logs["reward"] = rewards
            logs["energy"] = energys

            with open(os.path.join(output_path, "metrics.json"), "w") as f:
                f.write(json.dumps(logs, indent=4))

            with open(os.path.join(output_path, "ru.txt"), "w") as f:
                f.write(str(env.total_ru_services))

    model.save(output_path, "FINAL")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-o", "--output_paht", help="Output path", type=str, dest="output_path", required=True)
    parser.add_argument(
        "-c", "--checkpoint_path", help="Checkpoint path", type=str, dest="checkpoint_path", default=None
    )
    parser.add_argument("-e", "--start_episode", help="Start episode", type=int, dest="start_episode", default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    init_logs(os.path.join(args.output_path, "log.txt"))

    train(args.output_path, args.checkpoint_path, args.start_episode)
