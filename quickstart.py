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


def run_OnPolicy(room_id=me.EMPTY_ROOM, num_episodes=300, epsilon=0.2, alpha=0.5, gamma=0.9, plot_steps=False, max_steps=30, save_step_im=False):
    room = me.get_minihack_envirnment(room_id, add_pixel=False)
    room.reset()
    agent = commons.MonteCarloAgent('mca', room.action_space, epsilon)
    task = commons.Custom_RLTask_Learning_TD_OnPolicy(room, agent, alpha=alpha, discount_factor=gamma, roomID=room_id)
    av_returns = task.interact(num_episodes)

    if plot_steps:
        room2 = me.get_minihack_envirnment(room_id, add_pixel=True)
        room2.reset()
        task2 = commons.Custom_RLTask_Learning_TD_OnPolicy(room2, agent, alpha=alpha, discount_factor=gamma,
                                                           roomID=room_id, Qvalues=task.Qmatrix)
        task2.visualize_episode(max_number_steps=max_steps, save_im=save_step_im)
    return av_returns


def run_OffPolicy(room_id=me.EMPTY_ROOM, num_episodes=300, epsilon=0.2, alpha=0.5, gamma=0.9, plot_steps=False, max_steps=30, save_step_im=False):
    room = me.get_minihack_envirnment(room_id, add_pixel=False)
    room.reset()
    agent = commons.MonteCarloAgent('mca', room.action_space, epsilon)
    task = commons.Custom_RLTask_Learning_TD_OffPolicy(room, agent, alpha=alpha, discount_factor=gamma, roomID=room_id)
    av_returns = task.interact(num_episodes)

    if plot_steps:
        room2 = me.get_minihack_envirnment(room_id, add_pixel=True)
        room2.reset()
        task2 = commons.Custom_RLTask_Learning_TD_OffPolicy(room2, agent, alpha=alpha, discount_factor=gamma,
                                                            roomID=room_id, Qvalues=task.Qmatrix)
        task2.visualize_episode(max_number_steps=max_steps, save_im=save_step_im)
    return av_returns


def run_MC(room_id=me.EMPTY_ROOM, num_episodes=300, epsilon=0.2, gamma=0.9, plot_steps=False, max_steps=30, save_step_im=False):
    room = me.get_minihack_envirnment(room_id, add_pixel=False)
    room.reset()
    agent = commons.MonteCarloAgent('mca', room.action_space, epsilon)
    task = commons.Custom_RLTask_Learning_MC(room, agent, roomID=room_id, discountF=gamma)

    av_returns = task.interact(num_episodes)

    if plot_steps:
        room2 = me.get_minihack_envirnment(room_id, add_pixel=True)
        room2.reset()
        task2 = commons.Custom_RLTask_Learning_MC(room, agent, roomID=room_id, discountF=gamma, Qvalues=task.Qmatrix)
        task2.visualize_episode(max_number_steps=max_steps, save_im=save_step_im)
    return av_returns


#Task 1.1
# env=commons.Custom_GridEnv(size=(5,5))
# randagent = commons.Custom_RandomAgent("random_agent",action_space=env.action_space)
# rltask = commons.Custom_RLTask(env, randagent)
# av_returns = rltask.interact(10000)
# print(av_returns)
# plt.plot(av_returns)
# plt.title("average returns")
# plt.savefig('avg_return_task1.12.png')
# plt.show()

# rltask.visualize_episode(max_number_steps=10,saveFig=False)

#Task 1.2
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


#Taks 2.1
#experiment TD on VS off and TD on VS MC
id_lava = me.ROOM_WITH_LAVA
id_empty = me.EMPTY_ROOM
id_cliff = me.CLIFF
id_monster = me.ROOM_WITH_MONSTER
id_lava_mod = me.ROOM_WITH_LAVA_MODIFIED

# empty_room_env = me.get_minihack_envirnment(id, add_pixel=False)
# state = empty_room_env.reset()
# epsilon=0.1
# agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,epsilon)
# mc_ltask = commons.Custom_RLTask_Learning_MC(empty_room_env, agent, roomID=id, discountF=0.9)
# av_returns_mc = mc_ltask.interact(1000)

# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# empty_room_env.reset()
# mc_ltask2 = commons.Custom_RLTask_Learning_MC(empty_room_env, agent, roomID=id, discountF=0.9,Qvalues = mc_ltask.Qmatrix)
# mc_ltask2.visualize_episode(max_number_steps=30)




# empty_room_env = me
# empty_room_env2= me.get_minihack_envirnment(id, add_pixel=False)
# state = empty_room_env2.reset()
# agent2 = commons.MonteCarloAgent('mca',empty_room_env2.action_space,epsilon)
# off_task = commons.Custom_RLTask_Learning_TD_OnPolicy(empty_room_env2,agent2,alpha=0.1,discount_factor=0.9,roomID=id)
# av_returns_on = off_task.interact(1000)

# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# empty_room_env.reset()
# off_task2 = commons.Custom_RLTask_Learning_TD_OnPolicy(empty_room_env,agent2,alpha=0.5,discount_factor=0.9,roomID=id,Qvalues = mc_ltask.Qmatrix)
# off_task2.visualize_episode(max_number_steps=30,save_im=False)


# empty_room_env2= me.get_minihack_envirnment(id, add_pixel=False)
# state = empty_room_env2.reset()
# agent2 = commons.MonteCarloAgent('mca',empty_room_env2.action_space,epsilon)
# off_task = commons.Custom_RLTask_Learning_TD_OffPolicy(empty_room_env2,agent2,alpha=0.1,discount_factor=0.9,roomID=id)
# av_returns_off = off_task.interact(1000)

id = id_cliff
num_episodes = 500
epsilon=0.3
gamma=0.9
alpha = 0.5

repeat=5
on_returns = []
for _ in range(repeat):
    returns_on = run_OnPolicy(id,num_episodes,epsilon,alpha, gamma)
    on_returns.append(returns_on)
on_returns = np.array(on_returns)
av_returns_on = np.mean(on_returns,axis=0)


off_returns = []
for _ in range(repeat):
    returns_off = run_OffPolicy(id,num_episodes,epsilon,alpha, gamma)
    off_returns.append(returns_off)
off_returns = np.array(off_returns)
av_returns_off = np.mean(off_returns,axis=0)
# av_returns_mc = run_MC(id, num_episodes,epsilon,gamma)

# plt.plot(av_returns_mc, label= "MC")
print(av_returns_on)
plt.plot(av_returns_on, label= "TD On Policy")
plt.plot(av_returns_off, label= "TD Off Policy")
plt.title("average returns "+id+"( epsilon="+str(epsilon)+", gamma="+str(gamma)+", alpha="+str(alpha)+" )")
plt.legend()
plt.xlabel("episodes")
plt.ylabel("return")
plt.savefig("On_VS_off_"+id+".png")
plt.show()

#experiment: different LRs:
# id = me.ROOM_WITH_LAVA
# id = me.EMPTY_ROOM
# id = me.CLIFF
# id = me.ROOM_WITH_MONSTER
#
# epsilons = [0.1,0.2,0.3,0.5,0.8,1]
# plt.figure()
# results = []
# for eps in epsilons:
#     empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
#     state = empty_room_env.reset()
#     # epsilon=0.3
#
#     agent2 = commons.MonteCarloAgent('mca',empty_room_env.action_space,eps)
#     # off_task = commons.Custom_RLTask_Learning_TD_OffPolicy(empty_room_env,agent2,alpha=LR,discount_factor=0.9,roomID=id)
#     on_ltask = commons.Custom_RLTask_Learning_TD_OnPolicy(empty_room_env, agent2,alpha=0.5, discount_factor=0.9,roomID=id)
#
#     av_returns_mc = on_ltask.interact(200)
#     results.append(av_returns_mc)
#     on_ltask.visualize_episode(max_number_steps=30)
#
# plt.plot(results[0], label= "LR "+str(epsilons[0]))
# plt.plot(results[1], label= "LR "+str(epsilons[1]))
# plt.plot(results[2], label= "LR "+str(epsilons[2]))
# plt.plot(results[3], label= "LR "+str(epsilons[3]))
# plt.plot(results[4], label= "LR "+str(epsilons[4]))
# plt.plot(results[5], label= "LR "+str(epsilons[5]))
#
# plt.title("average returns")
# plt.legend()
# plt.savefig("OnPolicy_epsilonTest_roomWithMonster.png")
# plt.show()

#task 2.2
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

#task 2.3
# id = me.ROOM_WITH_LAVA
# id2 = me.ROOM_WITH_LAVA_MODIFIED
# eps=0.2
#
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=False)
# state = empty_room_env.reset()
# agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,eps)
# on_task = commons.Custom_RLTask_Learning_TD_OffPolicy_Dyna(empty_room_env,agent,alpha=0.4,discount_factor=0.9,roomID=id)
# av_returns_lava = on_task.interact(200)
#
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# state = empty_room_env.reset()
# on_task2 = commons.Custom_RLTask_Learning_TD_OffPolicy_Dyna(empty_room_env,agent,alpha=0.4,discount_factor=0.9,roomID=id,Qvalues = on_task.Qmatrix)
# # on_task2.visualize_episode(max_number_steps=30,save_im=True)
#
#
#
#
# empty_room_env = me.get_minihack_envirnment(id2, add_pixel=False)
# state = empty_room_env.reset()
# agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,eps)
# rltask = commons.Custom_RLTask_Learning_TD_OffPolicy_Dyna(empty_room_env,agent,alpha=0.4,discount_factor=0.9,roomID=id2)
# av_returns_lavamod = rltask.interact(200)
#
# empty_room_env = me.get_minihack_envirnment(id2, add_pixel=True)
# state = empty_room_env.reset()
# rltask2 = commons.Custom_RLTask_Learning_TD_OffPolicy_Dyna(empty_room_env,agent,alpha=0.4,discount_factor=0.9,roomID=id2,Qvalues = rltask.Qmatrix)
# # rltask2.visualize_episode(max_number_steps=30,save_im=True)
#
# plt.plot(av_returns_lava, label= "lava normal")
# plt.plot(av_returns_lavamod, label= "lava modified")
# plt.title("average returns")
# plt.legend()
# plt.savefig("dynyQ_normalVSmodLava"+id+"2.png")
# plt.show()

# state = empty_room_env.reset()
# agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,eps)
# off_task = commons.Custom_RLTask_Learning_TD_OffPolicy(empty_room_env,agent,alpha=0.5,discount_factor=0.9,roomID=id)
# av_returns_off = off_task.interact(500)
# off_task.visualize_episode(max_number_steps=30,save_im=True)
#
# state = empty_room_env.reset()
# agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,eps)
# on_ltask = commons.Custom_RLTask_Learning_TD_OnPolicy(empty_room_env, agent,alpha=0.5, discount_factor=0.9,roomID=id)
# av_returns_on = on_ltask.interact(500)
# on_ltask.visualize_episode(max_number_steps=30,save_im=True)




#task 2.4
# # id = me.ROOM_WITH_LAVA
# # id = me.EMPTY_ROOM
# id = me.CLIFF
# # id = me.ROOM_WITH_MONSTER
# # id = me.ROOM_WITH_LAVA_MODIFIED
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=False)
# state = empty_room_env.reset()
#
# eps=0.2
# agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,eps)
#
# task1 = commons.Custom_RLTask_Learning_TD_OffPolicy_Dyna(empty_room_env,agent,alpha=0.4,discount_factor=0.9,roomID=id)
# av_returns_Q = task1.interact(400)
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# state = empty_room_env.reset()
# task12 = commons.Custom_RLTask_Learning_TD_OffPolicy_Dyna(empty_room_env,agent,alpha=0.4,discount_factor=0.9,roomID=id,Qvalues = task1.Qmatrix)
# task12.visualize_episode(max_number_steps=30,save_im=False)
#
#
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=False)
# state = empty_room_env.reset()
# task2 = commons.Custom_RLTask_Learning_TD_OffPolicy(empty_room_env,agent,alpha=0.4,discount_factor=0.9,roomID=id)
# av_returns_Qdyna = task2.interact(400)
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# state = empty_room_env.reset()
# task22 = commons.Custom_RLTask_Learning_TD_OffPolicy(empty_room_env,agent,alpha=0.4,discount_factor=0.9,roomID=id,Qvalues = task2.Qmatrix)
# task22.visualize_episode(max_number_steps=30,save_im=False)
#
#
# plt.plot(av_returns_Q, label= "dynaQ")
# plt.plot(av_returns_Qdyna, label= "Q")
# plt.title("average returns")
# plt.legend()
# plt.savefig("Q_VS_dynaQ_"+id+".png")
# plt.show()












#works for MC empty room: epsilon=0.3, episodes=300
# works for Onpolicy empty room: epsilon=0.3, episodes=300
# works for Onpolicy cliff : epsilon=0.3, episodes=500
#works for onpolicy room_with_monster: epsilon=0.2
# works for offPolicy empty room: epsilon=0.3, episodes=300
# works for offPolicy room with lava: epsilon=0.3, episodes=300, alpha=0.5, gamma=0.9
#on policy with room_with_lava and with room_with_lava_modified doesnt work (eps=0.3, 500 episoes, alpha=0.1, gamma=0.9)
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


