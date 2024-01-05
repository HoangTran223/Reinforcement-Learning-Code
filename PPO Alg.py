import torch as T
class Agent:
    def __init__(self, n_actions, input_dimension, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs            # Số lần lặp để train PPO
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dimension, alpha)
        self.critic = CriticNetwork(input_dimension, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()              # Lấy mẫu 1 hành động

        # Chuyển đổi giá trị từ Tensor sang Python:
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            # Tính toán advantage A^
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0

                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]- values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)       # Chuyển sang tensor PyTorch


            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)                  # Fix cho đủ chiều

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # Tính hàm L: 
                weighted_probs = advantage[batch] * prob_ratio                          # Biểu thức 1 của công thức

                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,        # Biếu thức thứ 2 của CT
                        1+self.policy_clip)*advantage[batch]

                # Dấu - có nghĩa ta đi theo hướng giảm của loss function:
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                
                # Tính L^{VF}:
                returns = advantage[batch] + values[batch]      # Giá trị thực tế V0
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                # Tính toán tổng hàm mất mát, từ cả 2 mô hình Actor và Critic. Chọn c1 = 0.5
                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                # Tính lại gradient:
                total_loss.backward()

                # Cập nhật lại tham số:
                # self.optimizer = optim.Adam(self.parameters(), lr=alpha)
                # Sử dụng thuật toán Adam (Adaptive Moment Estimation).
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()                              # Xóa bộ nhớ đệm sau mỗi vòng