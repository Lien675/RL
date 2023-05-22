import numpy as np

import minihack_env as me
import matplotlib.pyplot as plt
import matplotlib as mpl
import commons


# mpl.use('MacOSX')   #uncomment this in some MacOSX machines for matplotlib

# How to get a minihack environment from the minihack_env utility.
# id = me.EMPTY_ROOM
# env = me.get_minihack_envirnment(id)
# state = env.reset()
# print("Initial state", state)
# next_state = env.step(1) # observation, reward, terminated, info
# print("Next State", next_state)
#


# How to get a minihack environment with also pixels states
# id = me.EMPTY_ROOM
# env = me.get_minihack_envirnment(id, add_pixel=True)
# state = env.reset()
# print("Initial state", state)
# plt.imshow(state["pixel"])
# plt.show()


# Crop representations to non-empty part
# id = me.EMPTY_ROOM
# env = me.get_minihack_envirnment(id, add_pixel=True)
# state = env.reset()
# print("Initial state", commons.get_crop_chars_from_observation(state))
# plt.imshow(commons.get_crop_pixel_from_observation(state))
# plt.show()


def run_OnPolicy(room_id=me.EMPTY_ROOM, num_episodes=300, epsilon=0.2, alpha=0.5, gamma=0.9, run_steps=False,
                 plot_steps=False, max_steps=30, save_step_im=False):
    room = me.get_minihack_envirnment(room_id, add_pixel=False)
    room.reset()
    agent = commons.eps_greedy_agent('mca', room.action_space, epsilon)
    task = commons.Custom_RLTask_Learning_TD_OnPolicy(room, agent, alpha=alpha, discount_factor=gamma, roomID=room_id)
    av_returns = task.interact(num_episodes)

    if run_steps:
        room2 = me.get_minihack_envirnment(room_id, add_pixel=True)
        room2.reset()
        task2 = commons.Custom_RLTask_Learning_TD_OnPolicy(room2, agent, alpha=alpha, discount_factor=gamma,
                                                           roomID=room_id, Qvalues=task.Qmatrix)
        rewards = task2.visualize_episode(max_number_steps=max_steps, save_im=save_step_im, plot_steps=plot_steps)
        return av_returns, rewards
    return av_returns


def run_OffPolicy(room_id=me.EMPTY_ROOM, num_episodes=300, epsilon=0.2, alpha=0.5, gamma=0.9, run_steps=False,
                  plot_steps=False, max_steps=30, save_step_im=False):
    room = me.get_minihack_envirnment(room_id, add_pixel=False)
    room.reset()
    agent = commons.eps_greedy_agent('mca', room.action_space, epsilon)
    task = commons.Custom_RLTask_Learning_TD_OffPolicy(room, agent, alpha=alpha, discount_factor=gamma, roomID=room_id)
    av_returns = task.interact(num_episodes)

    if run_steps:
        room2 = me.get_minihack_envirnment(room_id, add_pixel=True)
        room2.reset()
        task2 = commons.Custom_RLTask_Learning_TD_OffPolicy(room2, agent, alpha=alpha, discount_factor=gamma,
                                                            roomID=room_id, Qvalues=task.Qmatrix)
        rewards = task2.visualize_episode(max_number_steps=max_steps, save_im=save_step_im, plot_steps=plot_steps)
        return av_returns, rewards
    return av_returns


def run_dyna_OffPolicy(room_id=me.EMPTY_ROOM, num_episodes=300, epsilon=0.2, alpha=0.5, gamma=0.9, run_steps=False,
                       plot_steps=False, max_steps=30, save_step_im=False):
    room = me.get_minihack_envirnment(room_id, add_pixel=False)
    room.reset()
    agent = commons.eps_greedy_agent('mca', room.action_space, epsilon)
    task = commons.Custom_RLTask_Learning_TD_OffPolicy_Dyna(room, agent, alpha=alpha, discount_factor=gamma,
                                                            roomID=room_id)
    av_returns = task.interact(num_episodes)

    if run_steps:
        room2 = me.get_minihack_envirnment(room_id, add_pixel=True)
        room2.reset()
        task2 = commons.Custom_RLTask_Learning_TD_OffPolicy_Dyna(room2, agent, alpha=alpha, discount_factor=gamma,
                                                                 roomID=room_id, Qvalues=task.Qmatrix)
        rewards = task2.visualize_episode(max_number_steps=max_steps, save_im=save_step_im, plot_steps=plot_steps)
        return av_returns, rewards
    return av_returns


def run_MC(room_id=me.EMPTY_ROOM, num_episodes=300, epsilon=0.2, gamma=0.9, run_steps=False, plot_steps=False,
           max_steps=30, save_step_im=False):
    room = me.get_minihack_envirnment(room_id, add_pixel=False)
    room.reset()
    agent = commons.eps_greedy_agent('mca', room.action_space, epsilon)
    task = commons.Custom_RLTask_Learning_MC(room, agent, roomID=room_id, discountF=gamma)

    av_returns = task.interact(num_episodes)

    if run_steps:
        room2 = me.get_minihack_envirnment(room_id, add_pixel=True)
        room2.reset()
        task2 = commons.Custom_RLTask_Learning_MC(room2, agent, roomID=room_id, discountF=gamma, Qvalues=task.Qmatrix)
        eps_reward = task2.visualize_episode(max_number_steps=max_steps, save_im=save_step_im, plot_steps=plot_steps)
        return av_returns, eps_reward
    return av_returns


# Task 1.1
# env=commons.Custom_GridEnv(size=(5,5))
# randagent = commons.Custom_RandomAgent("random_agent",action_space=env.action_space)
# rltask = commons.Custom_RLTask(env, randagent)
# av_returns = rltask.interact(10000)
# plt.plot(av_returns)
# plt.title("average returns")
# plt.savefig('21.05/avg_return_task1.png')
# plt.show()
#
# rltask.visualize_episode(max_number_steps=10,saveFig=False)

# Task 1.2
# id = me.EMPTY_ROOM
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# state = empty_room_env.reset()
# agent = commons.FixedAgent("fixed_agent",empty_room_env.action_space)
# rltask = commons.Custom_RLTask(empty_room_env, agent)
# # show inital state:
# plt.imshow(commons.get_crop_pixel_from_observation(state))
# plt.savefig('avg_return_EmptyRoom_task1.2.png')
#
# plt.show()
# rltask.visualize_episode(max_number_steps=10,saveFig=True)

# id = me.ROOM_WITH_LAVA
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# state = empty_room_env.reset()
# agent = commons.FixedAgent("fixed_agent",empty_room_env.action_space)
# rltask = commons.Custom_RLTask(empty_room_env, agent)
# #show inital state:
# plt.imshow(commons.get_crop_pixel_from_observation(state))
# plt.show()
# rltask.visualize_episode(max_number_steps=10,saveFig=True)


# print("Initial state", commons.get_crop_chars_from_observation(state))
# empty_room_env.step(1)
# newstate = empty_room_env._get_observation(empty_room_env.last_observation)
# print("new state", commons.get_crop_chars_from_observation(newstate))
# empty_room_env.step(1)
# newstate = empty_room_env._get_observation(empty_room_env.last_observation)
# print("new state", commons.get_crop_chars_from_observation(newstate))
# plt.imshow(commons.get_crop_pixel_from_observation(state))
# plt.show()
# plt.imshow(commons.get_crop_pixel_from_observation(newstate))
# plt.show()


# Taks 2.1
# experiment TD on VS off and TD on VS MC
id_lava = me.ROOM_WITH_LAVA
id_empty = me.EMPTY_ROOM
id_cliff = me.CLIFF
id_monster = me.ROOM_WITH_MONSTER
id_lava_mod = me.ROOM_WITH_LAVA_MODIFIED

#
# id=id_lava
# num_episodes = 500
# epsilon=0.3
# gamma=0.9
# alpha = 0.5
# repeat=3
#
# on_returns = []
# mc_returns=[]
# for _ in range(repeat):
#     returns_on = run_OnPolicy(id,num_episodes,epsilon,alpha, gamma)
#     returns_mc = run_MC(id, num_episodes, epsilon, gamma)
#
#     on_returns.append(returns_on)
#     mc_returns.append(returns_mc)
#
# on_returns = np.array(on_returns)
# av_on_returns = np.mean(on_returns,axis=0)
# mc_returns = np.array(mc_returns)
# av_mc_returns = np.mean(mc_returns,axis=0)
#
# # plt.plot(av_returns_mc, label= "MC")
# plt.plot(av_mc_returns, label= "MC")
# plt.plot(av_on_returns, label= "TD On Policy")
# plt.title("average returns "+id+"( epsilon="+str(epsilon)+", gamma="+str(gamma)+", alpha="+str(alpha)+" )")
# plt.legend()
# plt.xlabel("episodes")
# plt.ylabel("return")
# plt.savefig("21.05/MC_VS_OnPol_"+id+".png")
# plt.show()

# experiment: different LRs:
# id = id_lava
# num_episodes = 600
# epsilon=0.3
# gamma=0.9
# alpha = 0.5
# repeat=3
# LRs=[0.1,0.3,0.5,0.8,1]
# results= []
# for LR in LRs:
#     off_returns = []
#     for _ in range(repeat):
#         returns_off = run_OffPolicy(id,num_episodes,epsilon,LR, gamma)
#         off_returns.append(returns_off)
#     off_returns = np.array(off_returns)
#     av_returns_off = np.mean(off_returns,axis=0)
#     results.append(av_returns_off)
#
# plt.plot(results[0], label= "LR "+str(LRs[0]))
# plt.plot(results[1], label= "LR "+str(LRs[1]))
# plt.plot(results[2], label= "LR "+str(LRs[2]))
# plt.plot(results[3], label= "LR "+str(LRs[3]))
# plt.plot(results[4], label= "LR "+str(LRs[4]))
# # plt.plot(av_returns_off, label= "TD Off Policy")
# plt.title("average returns LR tests"+id+"( epsilon="+str(epsilon)+", gamma="+str(gamma)+")")
# plt.legend()
# plt.xlabel("episodes")
# plt.ylabel("return")
# plt.savefig("offPol_LRtest_"+id+"2.png")
# plt.show()

# experiment with different epsilon
# id = id_lava
# num_episodes = 600
# gamma=0.9
# alpha = 0.5
# repeat=3
# epsilons=[0.1,0.2,0.5,0.8,1]
# results= []
# for epsilon in epsilons:
#     off_returns = []
#     for _ in range(repeat):
#         returns_off = run_OffPolicy(id,num_episodes,epsilon,alpha, gamma)
#         off_returns.append(returns_off)
#     off_returns = np.array(off_returns)
#     av_returns_off = np.mean(off_returns,axis=0)
#     results.append(av_returns_off)
# 
# plt.plot(results[0], label= "epsilon "+str(epsilons[0]))
# plt.plot(results[1], label= "epsilon "+str(epsilons[1]))
# plt.plot(results[2], label= "epsilon "+str(epsilons[2]))
# plt.plot(results[3], label= "epsilon "+str(epsilons[3]))
# plt.plot(results[4], label= "epsilon "+str(epsilons[4]))
# # plt.plot(av_returns_off, label= "TD Off Policy")
# plt.title("Off Policy average returns epsilon tests"+id+"( alpha="+str(alpha)+", gamma="+str(gamma)+")")
# plt.legend()
# plt.xlabel("episodes")
# plt.ylabel("return")
# plt.savefig("offPol_Epstest_"+id+".png")
# plt.show()


# task 2.2
# id = me.CLIFF
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=False)
# state = empty_room_env.reset()
# eps=0.3
# agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,eps)
# mc_ltask = commons.Custom_RLTask_Learning_MC(empty_room_env, agent, roomID=id, discountF=0.9)
# av_returns_mc = mc_ltask.interact(2000)
#
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# state = empty_room_env.reset()
# mc_ltask2 = commons.Custom_RLTask_Learning_MC(empty_room_env, agent, roomID=id, discountF=0.9, Qvalues = mc_ltask.Qmatrix)
#
# mc_ltask2.visualize_episode(max_number_steps=30,save_im=True)


# task 2.3

# id = id_lava
# id = id_lava_mod
# num_episodes = 300
# epsilon=0.2
# gamma=0.9
# alpha = 0.3
#
# repeat=10
# avg_returns = []
# eps_returns = []
# for _ in range(repeat):
#     # returns_on,eps_reward = run_OffPolicy(id,num_episodes,epsilon,alpha, gamma,run_steps=True,plot_steps=False)
#     # returns_on,eps_reward = run_dyna_OffPolicy(id,num_episodes,epsilon,alpha, gamma,run_steps=True,plot_steps=False)
#     returns_on,eps_reward = run_OnPolicy(id,num_episodes,epsilon,alpha, gamma,run_steps=True,plot_steps=False)
#     # av_returns_mc, eps_reward = run_MC(id, num_episodes,epsilon,gamma,run_steps=True,plot_steps=False)
#
#     avg_returns.append(returns_on)
#     eps_returns.append(eps_reward)
#
# avg_returns = np.array(avg_returns)
# avg_returns = np.mean(avg_returns,axis=0)
#
# fig, (ax1, ax2) = plt.subplots(1,2)
# fig.suptitle("On pol"+id+"( alpha="+str(alpha)+", gamma="+str(gamma)+", epsilon="+str(epsilon)+")")
# ax1.plot(avg_returns)
# ax1.set_title("average returns")
# ax1.set_xlabel("episodes")
# ax1.set_ylabel("average returns")
# ax2.plot(eps_returns)
# ax2.set_title("episode returns")
# ax2.set_xlabel("different runs")
# ax2.set_ylabel("returns")
#
# fig.savefig('task2.3/onPoll_'+str(id)+"2.png")


# task 2.4
# id = id_empty
# id = id_cliff
# id = id_lava
id = id_monster
# id = id_lava_mod
num_episodes = 500
epsilon = 0.2
gamma = 0.9
alpha = 0.3

repeat = 5
avg_returns_off = []
eps_returns_off = []
avg_returns_dynaoff = []
eps_returns_dynaoff = []
for _ in range(repeat):
    returns_off, eps_reward_off = run_OffPolicy(id, num_episodes, epsilon, alpha, gamma, run_steps=True,
                                                plot_steps=False)
    returns_dynaoff, eps_reward_dynaoff = run_dyna_OffPolicy(id, num_episodes, epsilon, alpha, gamma, run_steps=True,
                                                             plot_steps=False)
    avg_returns_off.append(returns_off)
    eps_returns_off.append(eps_reward_off)
    avg_returns_dynaoff.append(returns_dynaoff)
    eps_returns_dynaoff.append(eps_reward_dynaoff)

avg_returns_off = np.array(avg_returns_off)
avg_returns_off = np.mean(avg_returns_off, axis=0)
avg_returns_dynaoff = np.array(avg_returns_dynaoff)
avg_returns_dynaoff = np.mean(avg_returns_dynaoff, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(
    "Q VS dyna-Q" + id + "( alpha=" + str(alpha) + ", gamma=" + str(gamma) + ", epsilon=" + str(epsilon) + ")")
ax1.plot(avg_returns_off, label="Q")
ax1.plot(avg_returns_dynaoff, label="dyna-Q")
ax1.set_title("average returns")
ax1.set_xlabel("episodes")
ax1.set_ylabel("average returns")
ax1.legend()
ax2.plot(eps_returns_off, label="Q")
ax2.plot(eps_returns_dynaoff, label="dyna-Q")
ax2.set_title("episode returns")
ax2.set_xlabel("different runs")
ax2.legend()
ax2.set_ylabel("returns")

fig.savefig('21.05/QvsDynaQ_' + str(id) + "2.png")
plt.show()

#
# plt.plot(av_returns_Q, label= "dynaQ")
# plt.plot(av_returns_Qdyna, label= "Q")
# plt.title("average returns")
# plt.legend()
# plt.savefig("Q_VS_dynaQ_"+id+".png")
# plt.show()


# works for MC empty room: epsilon=0.3, episodes=300
# works for Onpolicy empty room: epsilon=0.3, episodes=300
# works for Onpolicy cliff : epsilon=0.3, episodes=500
# works for onpolicy room_with_monster: epsilon=0.2
# works for offPolicy empty room: epsilon=0.3, episodes=300
# works for offPolicy room with lava: epsilon=0.3, episodes=300, alpha=0.5, gamma=0.9
# on policy with room_with_lava and with room_with_lava_modified doesnt work (eps=0.3, 500 episoes, alpha=0.1, gamma=0.9)
#

# epsilon=0.3 #0.2
# agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,epsilon)
# # rltask = commons.Custom_RLTask_Learning_MC(empty_room_env, agent, id) #works, shortest path found
# rltask = commons.Custom_RLTask_Learning_TD_OnPolicy(empty_room_env, agent,alpha=0.5, discount_factor=0.9,roomID=id)
# # rltask = commons.Custom_RLTask_Learning_TD_OffPolicy(empty_room_env,agent,alpha=0.5,discount_factor=0.9,id)
# # rltask.visualize_episode(max_number_steps=10)
# av_returns = rltask.interact(1000)
#
# print(rltask.Qmatrix)
# plt.plot(av_returns)
# plt.title("average returns")
# plt.show()
#
# rltask.visualize_episode(max_number_steps=30)
