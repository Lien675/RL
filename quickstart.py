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


#Task 1.1
# env=commons.Custom_GridEnv(size=(5,5))
# randagent = commons.Custom_RandomAgent("random_agent",action_space=env.action_space)
# rltask = commons.Custom_RLTask(env, randagent)
# av_returns = rltask.interact(10000)
# print(av_returns)
# plt.plot(av_returns)
# plt.title("average returns")
# plt.savefig('avg_return_task1.1.png')
# plt.show()
#
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
# id = me.ROOM_WITH_LAVA
id = me.EMPTY_ROOM
# id = me.CLIFF

empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
state = empty_room_env.reset()
epsilon=0.15
agent = commons.MonteCarloAgent('mca',empty_room_env.action_space,epsilon)
# on_ltask = commons.Custom_RLTask_Learning_TD_OnPolicy(empty_room_env, agent,alpha=0.5, discount_factor=0.9,roomID=id)
mc_ltask = commons.Custom_RLTask_Learning_MC(empty_room_env, agent, roomID=id, discountF=0.9)

empty_room_env2= me.get_minihack_envirnment(id, add_pixel=True)
state = empty_room_env2.reset()
agent2 = commons.MonteCarloAgent('mca',empty_room_env2.action_space,epsilon)
off_task = commons.Custom_RLTask_Learning_TD_OffPolicy(empty_room_env2,agent2,alpha=0.5,discount_factor=0.9,roomID=id)

# rltask = commons.Custom_RLTask_Learning_MC(empty_room_env, agent)
av_returns_mc = mc_ltask.interact(500)
mc_ltask.visualize_episode(max_number_steps=30)

av_returns_off = off_task.interact(500)
off_task.visualize_episode(max_number_steps=30,save_im=False)

plt.plot(av_returns_mc, label= "MC")
plt.plot(av_returns_off, label= "TD Off Policy")
plt.title("average returns")
plt.legend()
plt.savefig("OffPolicy_VS_MC_EmptyRoom4.png")
plt.show()



#task 2.2
# # id = me.ROOM_WITH_MONSTER
# # id = me.ROOM_WITH_LAVA
# id = me.EMPTY_ROOM
# # id = me.CLIFF
# # id=me.ROOM_WITH_LAVA_MODIFIED
# empty_room_env = me.get_minihack_envirnment(id, add_pixel=True)
# state = empty_room_env.reset()

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


