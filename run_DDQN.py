# --------- Double DQN Algorithm --------- ##
# libraries
import gymnasium as gym
import numpy as np
# pytorch library is used for deep learning
import torch
import os
from scipy import stats
from DDQN import DQNAgent
import imageio
import matplotlib.pyplot as plt

# 결과 및 이미지 저장을 위한 폴더 생성
if not os.path.exists('results'):
    os.makedirs('results')

# 평가
def test(model_path, eval_episodes):
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()

    total_rewards = []
    for e in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        frames = []  # 각 스텝에서의 이미지를 저장할 리스트


        while True:
            # 튜플인 데이터로 나오는 경우, 첫 번째 요소를 NumPy 배열로 변환
            if isinstance(state, tuple):
                state = np.array(state[0], dtype=np.float32)

            state = torch.from_numpy(state).float().unsqueeze(0)
            action = agent.model(state).argmax().item()
            next_state, reward,terminated, truncated, info = env.step(action)
            done = (terminated or truncated)
            episode_reward += reward
            state = next_state

            # 이미지 캡처 및 저장
            # frame = env.render(mode='rgb_array')
            # frames.append(frame)

            if done:
                break
        
        total_rewards.append(episode_reward)

        # 프레임 저장 (GIF 생성을 위해)
        if frames:
            gif_filename = f'results/mountain_car_{e}.gif'
            imageio.mimsave(gif_filename, frames, fps=30)

    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {eval_episodes} episodes: {average_reward}")

    # 평가 결과 시각화 및 저장
    rewards = torch.tensor(total_rewards, dtype=torch.float)
    fig, ax = plt.subplots(1, figsize=[10,5])
    plt.title("Rewards")
    plt.plot(rewards)
    if len(rewards) >= 100 :
        bin = 100
    else :
        bin = len(rewards)
    means = rewards.unfold(0, bin, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(bin - 1) + means[0], means))
    plt.plot(means.numpy())
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('./results/test_rewards.png')
    plt.show()

    # 신뢰 구간 계산
    mean_reward = np.mean(total_rewards)     # 평균 보상
    std_reward = np.std(total_rewards)     # 보상의 표준편차
    n = len(total_rewards)
    se = std_reward / np.sqrt(n)     # 표준 오류

    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)     # 신뢰 수준(이 경우 95%)에 해당하는 Z-점수
    margin_error = z_score * se     # 오차 한계
    ci_lower = mean_reward - margin_error     # 신뢰 구간의 하한
    ci_upper = mean_reward + margin_error     # 신뢰 구간의 상한

    print('mean of scores', np.mean(total_rewards))
    print(f"95% Confidence interval for the rewards: [{ci_lower}, {ci_upper}]")
    
    env.close()

if __name__ == '__main__':
    test('./results/policy_net_final_DDQN.pth', 100)