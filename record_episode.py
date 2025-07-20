import cv2
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from agents.dqn_agent import DQNAgent

env = simple_spread_v3.env(N=3, max_cycles=100, render_mode="rgb_array")
env.reset(seed=123)

obs_shape = env.observation_space(env.agents[0]).shape[0]
action_n = env.action_space(env.agents[0]).n
agents = {name: DQNAgent(obs_shape, action_n) for name in env.agents}

frames = []
epsilon = 0.05  # use low exploration so they act greedily

for agent in env.agent_iter():
    obs, _, term, trunc, _ = env.last()
    done = term or trunc
    if done:
        action = None
    else:
        action = agents[agent].act(obs, epsilon)
    env.step(action)
    frame = env.render()  # returns RGB array
    frames.append(frame)

# === SAVE VIDEO USING OpenCV ===
height, width, _ = frames[0].shape
out = cv2.VideoWriter("agent_behavior.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

for f in frames:
    f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
    out.write(f_bgr)

out.release()
print("✅ Saved: agent_behavior.mp4")
