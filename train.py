from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3 import A2C, DQN, PPO, SAC
from gymnasium import spaces
import torch as th
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import torch.nn.functional as F
from func import MD_SAC

# TODO: remove recorder
# from perfRecord import PerformanceRecord, recorder 

warnings.filterwarnings("ignore")
register(
    id='final-v0',
    entry_point='envs:DiffusionEnv',
    # kwargs={'model_name': 'default_model_name', 'target_steps': 10, 'max_steps': 100}
)


# def make_env(my_config):
#     env = gym.make('final-v0')#, model_name=my_config["DM_model"], target_steps=my_config["target_steps"], max_steps=my_config["max_steps"])
#     return env

def make_env(my_config):
    def _init():
        config = {
            "model_name": my_config["DM_model"],
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "mode": my_config["mode"]
        }
        return gym.make('final-v0', **config)
    return _init

# TODO: cnn expected 4 dim., got 5.
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['image'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space['image'].sample()[None]).float()
            ).shape[1]

        self.fc = nn.Linear(1, 32)
        self.linear = nn.Sequential(nn.Linear(n_flatten + 32, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # images = observations['image'].squeeze(1)
        images = observations['image']
        img_features = self.cnn(images.float())
        value = observations['value'].float()   # Shape: [batch_size]
        value = value.view(-1, 1)             # Add extra dimension: [batch_size, 1]
        value_features = F.relu(self.fc(value))
        combined = th.cat([img_features, value_features], dim=1)
        return self.linear(combined)
    
def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_reward = 0
    avg_ssim = 0
    avg_ddim_ssim = 0
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        avg_reward += info[0]['reward']
        avg_ssim   += info[0]['ssim']
        avg_ddim_ssim += info[0]['ddim_ssim']

    avg_reward /= eval_episode_num
    avg_ssim /= eval_episode_num
    avg_ddim_ssim /= eval_episode_num
        
    return avg_reward, avg_ssim, avg_ddim_ssim, info[0]['time_step_sequence']

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    # with th.profiler.profile(
    #     activities=[th.profiler.ProfilerActivity.CPU, th.profiler.ProfilerActivity.CUDA],
    #     # schedule=th.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=th.profiler.tensorboard_trace_handler(
    #         '/home/B10505058/RL_final/log/SAC_2epoch'
    #     ),
    #     record_shapes=False,
    #     profile_memory=False,
    #     with_stack=True
    # ) as prof:
    for epoch in range(config["epoch_num"]):

        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_reward, avg_ssim, avg_ddim_ssim, time_step_sequence = eval(eval_env, model, config["eval_episode_num"])

        # prof.step()

        print("Avg_reward:  ", avg_reward)
        print("Avg_ssim:    ", avg_ssim)
        print("Avg_ddim_ssim:", avg_ddim_ssim)
        print("Time_step_sequence:", time_step_sequence)
        print()
        wandb.log(
            {"avg_reward": avg_reward,
             "avg_ssim": avg_ssim,
             "avg_ddim_ssim": avg_ddim_ssim}
        )
        

        ### Save best model
        if current_best < avg_ssim:
            print("Saving Model")
            current_best = avg_ssim
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")

        print("---------------")
        # TODO: remove recorder
        # recorder.epochTock()
            
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))



def main():
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32),
    )
    # TODO: extract config
    my_config = {
        "run_id": "PPO_test_multi",

        "algorithm": PPO,
        "policy_network": "MultiInputPolicy",
        "save_path": "model/PPO_test_multi_cifar10",

        "epoch_num": 500, # default is 500
        "timesteps_per_epoch": 100,
        "eval_episode_num": 10,
        "learning_rate": 1e-4,
        "policy_kwargs": policy_kwargs,

        "DM_model": "model/ddpm_ema_cifar10",
        # "DM_model": "model/ddpm-ema-church-256",
        "target_steps": 10, # T
        "max_steps": 100,
        "mode": "diffusers", # onnx, diffusers
        # "mode": "onnx", # onnx, diffusers

        "num_train_envs": 8, # default is 16, 8 is better.
        "n_steps": 128 # default is 2048
    }
    run = wandb.init(
        project="final",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"],
    )
    
    # Create training environment 
    num_train_envs = my_config['num_train_envs']
    # train_env = DummyVecEnv([make_env(my_config) for _ in range(num_train_envs)])
    train_env = SubprocVecEnv([make_env(my_config) for _ in range(num_train_envs)])
    
    # env = DiffusionEnv('google/ddpm-cifar10-32')
    # model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    # model.learn(total_timesteps=20000)

    # Create evaluation environment 
    eval_env = DummyVecEnv([make_env(my_config)])  

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        train_env, 
        verbose=2,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
        policy_kwargs=my_config["policy_kwargs"],
        n_steps=my_config["n_steps"]
    )

    train(eval_env, model, my_config)
    # recorder.printResults()

    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     print('Train info:', info)
    #     env.render()
    #     if done:
    #         obs = env.reset()

if __name__ == '__main__':
    main()
