# --------- Double DQN Algorithm --------- ##
# libraries
import gymnasium as gym
from collections import deque
import random
import numpy as np
# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import seaborn as sns


# 1. Hyperparameters 설정
learning_rate = 0.001
gamma = 0.98
epsilon_start = 0.08
epsilon_end = 0.01
buffer_limit = 50000     # size of replay buffer
batch_size = 32
episodes = 3000
print_interval = 20

# 2. 결과 및 이미지 저장을 위한 폴더 생성
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
if not os.path.exists('results'):
    os.makedirs('results')

# 3. DQN Network Class
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 4. Replay Buffer (경험 재생 버퍼)
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            state_lst.append(s)
            action_lst.append([a])
            reward_lst.append([r])
            next_state_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(state_lst, dtype=torch.float), torch.tensor(action_lst), torch.tensor(reward_lst), torch.tensor(next_state_lst, dtype=torch.float), torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

# 5. DQN Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    # 타겟 모델 업데이트
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 에이전트의 행동 결정 (epsilon-greedy 적용)
    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return q_value.argmax().item()

    def replay(self):
        if self.memory.size() < batch_size:
            return

        states, actions, rewards, next_states, done_mask = self.memory.sample(batch_size)

        # Double DQN Update
        argmax_Q = self.model(next_states).max(1)[1].unsqueeze(1)
        
        curr_Q = self.model(states).gather(1, actions)
        max_q_prime = self.target_model(next_states).gather(1, argmax_Q)
        target = rewards + gamma * max_q_prime * done_mask

        loss = F.mse_loss(curr_Q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def plot_rewards(scores, steps_per_episode, ) :
    fig = plt.figure()
    rewards = torch.tensor(scores, dtype=torch.float)
    steps = torch.tensor(steps_per_episode, dtype=torch.int)

    plt.suptitle('Training Result')
    ax1 = fig.add_subplot(2, 1, 1)
    #ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(linestyle='--')
    ax1.tick_params('x', length=0)
    ax1.plot(rewards.numpy())
    if len(rewards) >= 100 :
        bin = 100
    else :
        bin = len(rewards)
    means = rewards.unfold(0, bin, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(bin - 1) + means[0], means))
    ax1.plot(means.numpy())
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(linestyle='--')
    ax2.plot(steps.numpy())
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.10)
    plt.pause(0.001)
    plt.savefig('./results/training_result_DDQN.png')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].set_title('Score')
    ax[1].set_title('Steps')
    sns.distplot(scores, rug=True, ax=ax[0])
    sns.distplot(steps_per_episode, rug=True, ax=ax[1])
    plt.savefig('./results/training_score_DDQN.png')
    plt.show()


# 6. 환경 설정 및 에이전트 초기화
envName = 'MountainCar-v0'
env = gym.make(envName)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 7. Training Loop (학습)
def main():
    agent = DQNAgent(state_size, action_size)
    scores = []  # 에피소드별 점수를 기록할 리스트
    steps_per_episode = []  # 에피소드별 스텝 수
    goal_reached = []  # 에피소드별 목표 달성 여부
    score = 0
    success = 0

    for e in range(episodes):
        state = env.reset()
        step_count = 0  # 스텝 수 카운트
        reached_goal = False  # 목표 달성 여부
        epsilon = max(epsilon_end, epsilon_start - epsilon_end*(e/200)) #Linear annealing from 8% to 1%

        while True:
            # 튜플인 데이터로 나오는 경우, 첫 번째 요소를 NumPy 배열로 변환
            if isinstance(state, tuple):
                state = np.array(state[0], dtype=np.float32)

            action = agent.act(torch.from_numpy(state).float(), epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            
            reward_result = reward
            # 성공하면 리워드를 더 크게 지정.
            if next_state[0] >= 0.5:      # flag 위치 0.5
                reward_result += 100     # 성공하면 리워드 100.
                success += 1    # flag에 닿으면 성공

            # # Small penalty for using energy
            if action == 0 or action == 2:  
                reward_result -= 0.1 

            agent.memory.put((state, action, reward_result/100.0, next_state, done_mask))
            state = next_state
            score += reward_result  
            step_count += 1
            
            # 목표 달성 여부 체크 (MountainCar의 경우, 위치가 0.5 이상이면 목표 달성)
            if next_state[0] >= 0.5:
                reached_goal = True

            if done:
                break

            if agent.memory.size() > 2000:
                agent.replay()

        scores.append(score)
        steps_per_episode.append(step_count)
        goal_reached.append(reached_goal)

        if e % print_interval == 0 and e != 0:
            agent.update_target()    # target 모델을 현재 모델과 동기화
            print("n_episode :{}, score : {:.1f}, steps : {}, goal_reached : {}, n_buffer : {}, eps : {:.2f}%, probability of success : {:.2f}%".format(
                                                            e, score/print_interval, step_count, reached_goal, agent.memory.size(), epsilon*100, ((success/e)*100)))
            score = 0

    # 모델 저장
    if (e % episodes == 0) :
        torch.save(agent.model.state_dict(), './checkpoint/policy_net_final_DDQN.pth')
        torch.save(agent.target_model.state_dict(), './checkpoint/target_net_final_DDQN.pth')
    if (e == episodes - 1) :
        torch.save(agent.model.state_dict(), './results/policy_net_final_DDQN.pth')
        torch.save(agent.target_model.state_dict(), './results/target_net_final_DDQN.pth')

    # 학습 결과 시각화 및 저장
    plot_rewards(scores, steps_per_episode)

    env.close()

if __name__ == '__main__':
    main()