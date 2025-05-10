import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import multiprocessing as mp
from stable_baselines3.common.vec_env import SubprocVecEnv

class NoDiffusionEnv(gym.Env):
    def __init__(self, env_id, target_steps=10, max_steps=100, threshold=0.8, t_queue=None, x0_t_queue=None, seed=232):
        super(NoDiffusionEnv, self).__init__()
        
        self.env_id = env_id
        self.target_steps = target_steps
        self.max_steps = max_steps
        self.threshold = threshold
        self.t_queue = t_queue  # 用於通信的隊列
        self.x0_t_queue = x0_t_queue
        self.seed = seed

        # 假設觀察空間是 10 維的浮點數向量
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # 假設動作空間是 5 個離散動作
        self.action_space = spaces.Discrete(5)

    def reset(self):
        # 重置環境的狀態
        return np.zeros(10), {}

    def step(self, action):
        # 這裡假設每一步獲得隨機的獎勳和是否結束
        reward = np.random.random()
        done = np.random.random() > 0.95  # 隨機結束
        return np.zeros(10), reward, done, {}


# 用來創建環境的函數
def make_env(config, env_id, t_queue=None, x0_t_queue=None, seed=None):
    def _init():
        return NoDiffusionEnv(env_id, target_steps=config['target_steps'], max_steps=config['max_steps'], 
                              threshold=config['threshold'], t_queue=t_queue, x0_t_queue=x0_t_queue, seed=seed)
    return _init

def main():
    # 配置
    num_train_envs = 4
    config = {
        "target_steps": 10,
        "max_steps": 100,
        "threshold": 0.8,
        "seed": 42
    }
    
    # 創建隊列
    t_queues = {i: mp.Queue() for i in range(num_train_envs)}
    x0_t_queues = {i: mp.Queue() for i in range(num_train_envs)}

    # 使用 SubprocVecEnv 並行運行環境
    train_env = SubprocVecEnv([make_env(config, i, t_queues[i], x0_t_queues[i], config["seed"] + i) for i in range(num_train_envs)])
    print("Training environments created.")
    # 假設你有一個簡單的主訓練循環
    for _ in range(10):
        # 從每個環境的 t_queue 和 x0_t_queue 獲取數據（如果有數據）
        for i in range(num_train_envs):
            if not t_queues[i].empty():
                t_data = t_queues[i].get()  # 獲取數據
                print(f"Env {i} t_data: {t_data}")
            
            if not x0_t_queues[i].empty():
                x0_t_data = x0_t_queues[i].get()  # 獲取數據
                print(f"Env {i} x0_t_data: {x0_t_data}")
        
        # 這裡可以進行訓練模型等操作

if __name__ == "__main__":
    main()
