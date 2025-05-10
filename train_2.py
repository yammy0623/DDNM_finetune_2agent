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
import os
import gc
import numpy as np
from main import parse_args_and_config
from guided_diffusion.my_diffusion import Diffusion as my_diffusion
from stable_baselines3.common.callbacks import BaseCallback
import multiprocessing as mp
import random
import torch.utils.data as data
WANDB_CALLBACK = True

th.set_printoptions(sci_mode=False)

warnings.filterwarnings("ignore")
register(
    id='final-ours',
    entry_point='envs:NoDiffusionEnv',
)
register(
    id='final-baseline',
    entry_point='envs:BaselineDiffusionEnv',
)



class RewardAccumulatorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardAccumulatorCallback, self).__init__(verbose)
        self.episode_rewards = []  # List to store sum of rewards for each episode
        self.current_episode_reward = 0  # Accumulator for the current episode's rewards
        self.all_epoch_averages = [0]  # Store the average rewards per epoch

    def _on_step(self) -> bool:
        # Accumulate rewards from each step
        self.current_episode_reward += self.locals['rewards'][0]

        # If the episode is done, append the sum to the episode_rewards list and reset
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # Reset for the next episode

        return True

    def _on_rollout_end(self):
        # At the end of each rollout (which can contain multiple episodes), calculate average
        if len(self.episode_rewards) > 0:
            epoch_average = np.mean(self.episode_rewards)
            self.all_epoch_averages.append(epoch_average)
            print(f"dh: {epoch_average}")
            wandb.log({"average_reward": epoch_average})
            # Reset for the next epoch
            self.episode_rewards = []

def make_env(my_config, env_id, t_queue, x0_t_queue, seed=None):
    # seed = my_config["seed"]
    print("Seed: ", seed)
    
    def _init():
        try:
            config = {
                "env_id": env_id,
                "target_steps": my_config["target_steps"],
                "max_steps": my_config["max_steps"],
                "threshold": my_config["threshold"],
                "seed": seed,
                "t_queue": t_queue,
                "x0_t_queue": x0_t_queue,
                "agent1": my_config["agent1"],
                "agent2": None if my_config["agent2"] is not None else my_config["agent2"],
            }
            
            if my_config["model_mode"] == "baseline":
                print('Baseline training mode ...')
                return gym.make('final-baseline', **config)
            else:
                print('2-agent training mode ...')
                return gym.make('final-ours', **config)
        except Exception as e:
            print(f"Error initializing environment {env_id}: {e}")
            raise e  # Reraise exception to notify the caller
    return _init


def log_custom_histogram(data, name, bin_width=None, num_bins=None):
    data = np.array(data)
    if bin_width is not None:
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
    elif num_bins is not None:
        bins = num_bins
    else:
        bins = 'auto'

    hist, bin_edges = np.histogram(data, bins=bins)

    if WANDB_CALLBACK:
        wandb.log({name: wandb.Histogram(np_histogram=(hist, bin_edges))})

    
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
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Normalize features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space['image'].sample()[None]).float()
            ).shape[1]

        self.fc = nn.Linear(1, 32)
        self.fc2 = nn.Linear(1, 32)
        self.fc_merge = nn.Linear(64, 32)
        self.embedding_output = nn.Linear(32, features_dim * 2)
        self.out_norm = nn.Linear(n_flatten, features_dim)  # Normalizing layer
        self.out_rest = nn.Sequential(
            nn.Linear(features_dim, features_dim),  # Further processing layer
            nn.ReLU()
        )
        # For subtask 1
        self.l_scale = nn.Parameter(th.ones(features_dim))  # Learnable scale parameter
        self.l_shift = nn.Parameter(th.zeros(features_dim))  # Learnable shift parameter
        

    def forward(self, observations: th.Tensor) -> th.Tensor:
        img_features = self.cnn(observations['image'].float())
        if 'value' in observations:
            value_features = F.relu(self.fc(observations['value'].float()))
        
        if 'remain' in observations:
            remain_features = F.relu(self.fc2(observations['remain'].float()))
            value_features = self.fc_merge(th.cat([value_features, remain_features], dim=1))

        if 'value' in observations:
            emb_out = self.embedding_output(value_features)
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.out_norm(img_features) * (1 + scale) + shift
            h = self.out_rest(h)
        else:
            h = self.out_norm(img_features) * (1 + self.l_scale) + self.l_shift
            h = self.out_rest(h)
            # h = self.out_rest(self.out_norm(img_features))

        return h


class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.start_T = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None:
            # print("get infos")
            for info in infos:
                if "t" in info:
                    self.start_T.append(info["t"].item())
        return True
    def _on_rollout_end(self):
        print("rollout_end")
        # print("start T", self.start_T)
        log_custom_histogram(self.start_T, "train_start_t", num_bins=500)
        self.start_T = []
    

def eval(env, model, eval_episode_num, num_steps):
    print("============ eval START ===============")
    """Evaluate the model and return avg_score and avg_highest"""
    avg_reward = 0
    avg_reward_t = [0 for _ in range(num_steps)]
    avg_t = [0 for _ in range(num_steps)]
    avg_ssim = 0
    avg_psnr = 0
    pivot_ssim = 0
    pivot_psnr = 0
    ddim_ssim = 0
    ddim_psnr = 0
    ddnm_ssim = 0
    ddnm_psnr = 0
    avg_start_t = 0
    start_t = []

    with th.no_grad():
        for seed in range(eval_episode_num):
            done = False
            # Set seed using SB3 API
            # env.seed(seed)
            obs, info = env.reset(isValid=True)

            now_t = 0
            # Interact with env using SB3 API
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                avg_reward_t[now_t] += info['reward']
                now_t += 1
                start_t.append(info['time_step_sequence'][0])
 
            # wandb.log({"val_histogram": wandb.Histogram(start_t)})
            
            avg_reward += info['reward']
            avg_ssim   += info['ssim']
            avg_psnr += info['psnr']
            pivot_ssim += info['pivot_ssim']
            pivot_psnr += info['pivot_psnr']
            ddim_ssim += info['ddim_ssim']
            ddim_psnr += info['ddim_psnr']
            ddnm_ssim += info['ddnm_ssim']
            ddnm_psnr += info['ddnm_psnr']
            # avg_start_t += info['time_step_sequence'][0]
            for i in range(num_steps):
                avg_t[i] += info['time_step_sequence'][i]

    avg_reward /= eval_episode_num # 跑到最後一步獲得的reward
    avg_ssim /= eval_episode_num
    avg_psnr /= eval_episode_num
    pivot_ssim /= eval_episode_num
    pivot_psnr /= eval_episode_num
    ddim_ssim /= eval_episode_num
    ddim_psnr /= eval_episode_num
    ddnm_ssim /= eval_episode_num
    ddnm_psnr /= eval_episode_num
    avg_start_t /= eval_episode_num
    for i in range(num_steps):
        avg_reward_t[i] = (avg_reward_t[i] / eval_episode_num) # 每一個步數獲得的reward
        avg_t[i] = avg_t[i] / eval_episode_num
    
    print("val start t: ", start_t)
    log_custom_histogram(start_t, "val_histogram", num_bins=500)
            
    print("============ eval DONE ===============")
    return avg_reward, avg_ssim, avg_psnr, pivot_ssim, pivot_psnr, ddim_ssim, ddim_psnr, ddnm_ssim, ddnm_psnr, info['time_step_sequence'], info['action_sequence'], info['threshold'], avg_reward_t, avg_t

# done_val, env_eval_config, args, my_config
def train(done_val, t_queues, x0_t_queues, env_config, env_eval_config, args, my_config, second_stage=False):
    num_steps = args.target_steps
    num_train_envs = my_config['num_train_envs']
    ### First stage training
    if args.finetune:
        epoch_num = epoch_num * args.finetune_ratio
        print("finetune: training with epoch_num = ", epoch_num)
    else:
        epoch_num = my_config['epoch_num'] if args.baseline else my_config["first_stage_epoch_num"]
        print("training with epoch_num = ", epoch_num)

    if not args.second_stage:
        ### First stage finetuning stage (need to load agent 2)
        if args.finetune:
            agent2 = my_config["algorithm"].load(f"{my_config['save_path']}/best_2")
            env_config['agent2'] = agent2
            # train_env = DummyVecEnv([make_env(config) for _ in range(num_train_envs)])
            train_env = SubprocVecEnv([make_env(env_config, i, t_queues[i], x0_t_queues[i], env_config["seed"]+i) for i in range(num_train_envs)])
        else:
            env_config['agent2'] = None
            # train_env = DummyVecEnv([make_env(config, seed) for seed in range(num_train_envs)])
            train_env = SubprocVecEnv([make_env(env_config, i, t_queues[i], x0_t_queues[i], env_config["seed"]+i) for i in range(num_train_envs)])
        

        model = my_config["algorithm"](
                my_config["policy_network"], 
                train_env,
                n_steps=my_config["n_steps"],
                n_epochs=my_config["n_epochs"],
                verbose=2,
                tensorboard_log=os.path.join("tensorboard_log", my_config["run_id"]),
                learning_rate=my_config["learning_rate"],
                policy_kwargs=my_config["policy_kwargs"],
            )
    eval_env = gym.make('final-baseline', **env_eval_config) if args.baseline else gym.make('final-ours', **env_eval_config)
        
    """Train agent using SB3 algorithm and my_config"""
    current_best_psnr = 0
    current_best_ssim = 0
    print("total training epochs: ", int(epoch_num))

    # if WANDB_CALLBACK:
    #     run = wandb.init(
    #         project="exp_"+args.path_y+"_"+args.deg+str(args.target_steps),
    #         config=my_config,
    #         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #         id=my_config["run_id"],
    #         resume=False
    #     )

        # wandb_callback = WandbCallback(
        #     gradient_save_freq=100,
        #     verbose=2,
        # )
    
    # start_t_callback = CustomWandbCallback()
    reward_callback = RewardAccumulatorCallback()
    
    callback_list = [reward_callback]

    for epoch in range(int(epoch_num)):

        model.learn(
            total_timesteps=my_config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=callback_list,
            progress_bar=True,
        )

        th.cuda.empty_cache()  # Clear GPU cache
        ### Evaluation
        print(my_config["run_id"])
        print("Epoch: ", epoch)
        avg_reward, avg_ssim, avg_psnr, pivot_ssim, pivot_psnr, ddim_ssim, ddim_psnr, ddnm_ssim, ddnm_psnr, time_step_sequence, action_sequence, threshold, avg_reward_t, avg_t = eval(eval_env, model, config["eval_episode_num"], num_steps)

        print("---------------")

        ### Save best model
        if (current_best_psnr + current_best_ssim) < (avg_psnr + avg_ssim) and (current_best_psnr < avg_psnr):# and epoch > 10:
        # if current_best_psnr < avg_psnr and current_best_ssim < avg_ssim:# and epoch > 10:
            print("Saving Model !!!")
            current_best_psnr = avg_psnr
            current_best_ssim = avg_ssim
            save_path = my_config["save_path"]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if second_stage:
                model.save(f"{save_path}/best_2")
            else:
                model.save(f"{save_path}/best")


        print("Threshold:", threshold)
        print("Avg_reward:  ", avg_reward)
        print("Avg_reward_t:  ", avg_reward_t)
        print("Avg_t:  ", avg_t)
        print("Avg_ssim:    ", avg_ssim)
        print("Avg_psnr:    ", avg_psnr)
        print("Current_best_ssim:", current_best_ssim)
        print("Current_best_psnr:", current_best_psnr)
        print("Pivot_ssim:  ", pivot_ssim)
        print("Pivot_psnr:  ", pivot_psnr)
        print("DDIM_ssim:   ", ddim_ssim)
        print("DDIM_psnr:   ", ddim_psnr)
        print("DDNM_ssim:   ", ddnm_ssim)
        print("DDNM_psnr:   ", ddnm_psnr)
        print("Time_step_sequence:", time_step_sequence)
        print("Action_sequence:", action_sequence)
        print("training reward", [] if not reward_callback.all_epoch_averages[-1] else reward_callback.all_epoch_averages[-1])
        print()
        gpu_index=0
        total = th.cuda.get_device_properties(gpu_index).total_memory
        used = th.cuda.memory_allocated(gpu_index)
        used_MB = used / 1024**2
        total_MB = total / 1024**2
        percent = used / total * 100
        print(f"GPU Memory Usage: {used_MB:.2f} MB / {total_MB:.2f} MB ({percent:.2f}%)")
        # wandb.log(
        #     {
        #         "avg_reward": avg_reward,
        #         "avg_ssim": avg_ssim,
        #         "avg_psnr": avg_psnr,
        #         # "pivot_ssim": pivot_ssim,
        #         # "pivot_psnr": pivot_psnr,
        #         # "ddim_ssim": ddim_ssim,
        #         # "ddim_psnr": ddim_psnr,
        #         # "ddnm_ssim": ddnm_ssim,
        #         # "ddnm_psnr": ddnm_psnr,

        #         "start_t": avg_t[0],
        #         "train_reward": [] if not reward_callback.all_epoch_averages[-1] else reward_callback.all_epoch_averages[-1],
        #         "gpu_process_memory_MB": used_MB,
        #         "gpu_process_memory_percent": percent,
        #     }
        # )
    
    with done_val.get_lock():
        done_val.value = True

def log_gpu_memory(step=None, gpu_index=0):
    total = th.cuda.get_device_properties(gpu_index).total_memory
    used = th.cuda.memory_allocated(gpu_index)

    used_MB = used / 1024**2
    total_MB = total / 1024**2
    percent = used / total * 100

    # wandb.log({
    #     "system/gpu_process_memory_MB": used_MB,
    #     "system/gpu_process_memory_percent": percent,
    # })

    


    

import torch
def diffusion_worker(INIT, done_val, t_queue, x0_t_queue, isValid, num_env, args, config):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # def seed_worker(worker_id):
    #     worker_seed = args.seed % 2 ** 32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    diffusion_model = my_diffusion(args, config)
    print("diffusion_model initialized")


    if isValid:
        dataset = diffusion_model.val_dataset
    else:
        dataset = diffusion_model.train_dataset

    
    print("prepare data_loader")
    data_loader = data.DataLoader(
        dataset,
        batch_size=num_env,
        shuffle=True,
        num_workers=config.data.num_workers,
        # worker_init_fn=seed_worker,
        generator=g,
    )
    
    print("get data_loader")
        
    for x_orig, classes in data_loader:
        # print("x_orig", x_orig)
        # print("classes", classes)
        print("load data")
        # print("x_orig shape", x_orig.shape)
        # print("classes shape", classes.shape)

        if done_val.value:
            print("done_val.value = True")
            break

        x, y, Apy, x_orig, A_inv_y = diffusion_model.preprocess(x_orig, None)
        x0_t = A_inv_y.clone()
        print("x0_t shape", x0_t.shape)

        # initialize the queue
        for env_id in range(num_env):
            x0_t_queue[env_id].put((x0_t[env_id].cpu(), None))
        

        
        # while not rl_done_val.value:
        x0_t_all = []
        t_all = []
        et_all = []

        if not t_queue[0].empty():
            print("t_queue not empty")
    
        for env_id in range(num_env):
            t = t_queue[env_id].get()
            t_all.append(torch.tensor(t).to(device))

        if INIT:
            print("first step")
            for env_id in range(num_env):
                x0_t, _ = x0_t_queue[env_id].get()
                x0_t_all.append(x0_t)
            x = diffusion_model.get_noisy_x(t_all, x0_t_all, initial=True)
            et_all = [None] * num_env  # no et for the first step
        else:
            for env_id in range(num_env):
                x0_t, et = x0_t_queue[env_id].get()
                x0_t_all.append(x0_t)
                et_all.append(et)
            x = diffusion_model.get_noisy_x(t_all, x0_t_all, et_all)

        x0_t_all, _, et_all = diffusion_model.single_step_ddnm(x, y, t_all, classes)
        print("diffusion step done")

        for env_id in range(num_env):
            x0_t_queue[env_id].put((x0_t_all[env_id], et_all[env_id]))
        print("put x0_t to queue")

            


manager = None
t_queues = None
x0_t_queues = None
done_val = None

def main():
    global manager, t_queues, x0_t_queues, done_val
    
    manager = mp.Manager()
    # Initialze DDNM
    args, config = parse_args_and_config()
    # training_data = 256
    # runner = my_diffusion(args, config)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Initialize config
    my_config = {
        "alg": "PPO",
        "target_steps": args.target_steps,
        "threshold": 0.9,
        "num_train_envs": 2,
        "n_steps": 5120,
        "n_epochs": 1,
        "policy_network": "MultiInputPolicy",
        "policy_kwargs": policy_kwargs,
        "iteration": 1,
        "eval_episode_num": 8,
        "learning_rate": 3e-4,
        "first_stage_epoch_num": 50,
        "epoch_num": 200,
        "diffusion_max_steps": 100,
        "task": args.deg,
        "model_mode": "baseline" if args.baseline else "2_agents",
        "finetune": args.finetune,
        "seed": args.seed,
        'save_path': args.save_path,
        "run_id": f"{args.id}{'_S2' if not args.baseline and args.second_stage else '_S1' if not args.baseline else ''}"
    }

    # Derived configurations
    my_config['timesteps_per_epoch'] = my_config['num_train_envs'] * my_config['n_steps'] * my_config['iteration']
    my_config['algorithm'] = PPO if my_config["alg"] == "PPO" else A2C

    # Print and create directories
    print("Run ID: ", my_config['run_id'])
    os.makedirs(my_config['save_path'], exist_ok=True)

    # Save config to file
    with open(os.path.join(my_config['save_path'], 'config.yml'), 'w') as f:
        for key, value in my_config.items():
            f.write(f"{key}: {value}\n")

    print("Config", my_config)


    
    # if WANDB_CALLBACK:
    #     run = wandb.init(
    #         project="exp_"+args.path_y+"_"+args.deg+str(args.target_steps),
    #         config=my_config,
    #         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #         id=my_config["run_id"],
    #         resume=False
    #     )
    

    # Create training environment 
    num_train_envs = my_config['num_train_envs']
    reward_accumulator = RewardAccumulatorCallback()
    

    # config for environment
    env_config = {
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["diffusion_max_steps"],
            "threshold": my_config["threshold"],
            "model_mode": my_config["model_mode"],
            "seed": my_config["seed"],
        }
    if args.baseline == False:
        env_config["agent1"] = None


    print("env_config: ", env_config)
    # create queue for each env
    
    t_queues = {env_id: manager.Queue() for env_id in range(num_train_envs)}
    x0_t_queues = {env_id: manager.Queue() for env_id in range(num_train_envs)}
    done_val = manager.Value('b', False)
    print("t_queues created")
    print("x0_t_queues created")
    
    # Multiple Process
    processes = []
    p_diffusion = mp.Process(target=diffusion_worker, args=(True, done_val, t_queues, x0_t_queues, False, num_train_envs, args, config))
    
    
    env_config["agent2"] = None
    env_eval_config = {
        "env_id": 0,
        "t_queue": t_queues[0],
        "x0_t_queue": x0_t_queues[0],
        "target_steps": my_config["target_steps"],
        "max_steps": my_config["diffusion_max_steps"],
        "threshold": my_config["threshold"],
        "seed": my_config["seed"],
        "agent1": env_config["agent1"],
        "agent2": env_config["agent2"],
        
    }
    p_rl = mp.Process(target=train, args=(done_val, t_queues, x0_t_queues, env_config, env_eval_config, args, my_config))
    p_diffusion.start()
    # processes.append(p_diffusion)
    p_rl.start()
    # processes.append(p_rl)

    p_diffusion.join()
    p_rl.join()
if __name__ == '__main__':
    print("Before setting:", th.cuda.memory_reserved())
    # th.cuda.set_per_process_memory_fraction(0., device=0)
    # cuda 不支援fork 模式
    mp.set_start_method("spawn", force=True)
    main()




