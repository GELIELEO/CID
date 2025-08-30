import gym
from gym import spaces
import numpy as np

import torch
import torch.nn as nn


from encoder.montero_small import Encoder
from decoder.montero_small import Decoder



class VAEEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, img_size, latent_dim):
        super(VAEEnv, self).__init__()
        
        # self.encoder = Encoder(img_size=img_size, latent_dim=latent_dim)
        self.decoder = Decoder(img_size=img_size, latent_dim=latent_dim)
        
        self.action_space = gym.spaces.Box(shape=(latent_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(shape=img_size, dtype=np.float32)  # 假设是28x28的灰度图像

    def step(self, action):
        
        decoded_image = self.decoder(action)
        
        # 计算奖励，这里我们使用简单的均方误差作为例子
        original_image = self._get_original_image()  # 你需要实现这个函数来获取原始图像
        reward = -np.mean((original_image - decoded_image) ** 2)
        
        # 假设我们没有终止条件，所以done总是False
        done = False

        # 可以添加一些调试信息
        info = {}

        return decoded_image, reward, done, info

    def reset(self):
        # 随机选择一个图像并编码为潜在向量
        self.original_image = self._get_random_image()  # 你需要实现这个函数来获取随机图像
        self.state = self.vae.encode(self.original_image)
        return self.state

    def render(self, mode='human', close=False):
        # 渲染当前的解码图像
        import matplotlib.pyplot as plt
        img = self._step(np.zeros(self.vae.latent_dim))[0]  # 用零向量解码，只是为了渲染
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title('Decoded Image')
        plt.show()

    def _get_original_image(self):
        # 这里应该实现获取原始图像的逻辑
        raise NotImplementedError

    def _get_random_image(self):
        # 这里应该实现获取随机图像的逻辑
        raise NotImplementedError

# 假设VAE类已经实现
vae_model = VAE()

# 创建环境
env = VAEEnv(vae_model)

# 测试环境
observation = env.reset()
for _ in range(10):
    action = env.action_space.sample()  # 随机采样一个动作
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        break
env.close()