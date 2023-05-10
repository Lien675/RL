import copy
import random
from typing import List
import numpy as np
import gym
from gym import spaces
import pygame
from matplotlib import pyplot as plt

actions = ['up', 'down', 'left', 'right']
# actions_vec =[[-1, 0], [1, 0], [0, 1], [0, -1]]
actions_pos_cal = {'up':[0,1], 'down':[0,-1], 'left':[-1,0], 'right':[1,0]}
action_idx = random.randint(0, 3)
action = actions[action_idx]

#state space
size=(5,4)
(n,m)=size
states = []
for i in range(n):
  for j in range(m):
    state = [i+1,j+1]
    states.append(state)



class Custom_GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None,size=(5,5)):
        super(Custom_GridEnv).__init__()

        m,n = size
        self.m=m
        self.n=n

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0]), high=np.array([n, m]), dtype=int),
                "target": spaces.Box(low=np.array([0, 0]), high=np.array([n, m]), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.start_state = [0,0]
        self.terminal_state=[n-1,m-1]
        self.step_reward = -1

        self._target_location = [n-1,m-1]
        self._agent_location = self.start_state


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}


    def reset(self):
        self.state = self.start_state
        self._agent_location = self.start_state
        return self.state

    def step(self, action):
        assert self.action_space.contains(action)
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, [self.n - 1,self.m-1]
        )
        reward = self.step_reward
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()


        return observation, reward, terminated, info



    def render(self, action=0, reward=-1):
        print("action: "+str(action)+": new agent location: "+str(self._agent_location)+" reward: "+str(reward))
         # print(f"{action}: ({self._agent_location}) reward = {reward}")

class Custom_RandomAgent():

    def __init__(self, id, action_space):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = action_space

        # Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        # You may want to disable some behaviour when not learning (e.g. no update rule, no exploration eps = 0, etc.)
        self.learning = True

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        next_state_int = random.randint(0, 3)
        return next_state_int
        # raise NotImplementedError()


    def onEpisodeEnd(self, reward, episode):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward: the reward obtained in the last step
        :param episode: the episode number
        :return:
        """
        pass




class AbstractAgent():

    def __init__(self, id, action_space):
        """
        An abstract interface for an agent.

        :param id: it is a str-unique identifier for the agent
        :param action_space: some representation of the action that an agents can do (e.g. gym.Env.action_space)
        """
        self.id = id
        self.action_space = action_space

        # Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        # You may want to disable some behaviour when not learning (e.g. no update rule, no exploration eps = 0, etc.)
        self.learning = True

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        raise NotImplementedError()


    def onEpisodeEnd(self, reward, episode):
        """
        This function can be exploited to allow the agent to perform some internal process (e.g. learning-related) at the
        end of an episode.
        :param reward: the reward obtained in the last step
        :param episode: the episode number
        :return:
        """
        pass


class FixedAgent(AbstractAgent):
    def __int__(self, id, action_space):
        super().__init__(id, action_space)

    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        # next_state_int = random.randint(0, 3)
        agent = 64
        sol = np.argwhere(state==agent)
        agent_pos = sol[0]
        index_row = agent_pos[0]
        index_column = agent_pos[1]
        env_height, env_width = len(state),len(state[0])

        if index_row+1<env_height and state[index_row+1][index_column]==46:
            return 2
        else:
            return 1


class MonteCarloAgent(AbstractAgent):
    def __init__(self, id, action_space,epsilon):
        super().__init__(id, action_space)
        self.epsilon=epsilon


    def act(self, state, reward, Qs,statelist):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        #my code:
        row, column = extract_user_location_from_state(state)
        state_index = statelist.index((row,column))

        u=random.random()

        if self.learning==False:
            x=0

        # code from tutorial:
        # if episode_index<80 and self.learning:
        #     return random.randint(0, 3)
        # if episode_index>80 and self.learning:
        #     self.epsilon=0.9*self.epsilon



        if u<self.epsilon and self.learning:
            action = random.randint(0, 3)

        else: #no training or u <1-epsilon
            # greey selection
            # action = argmax_a Q(q)

            # my own code
            # possible_Qs = [row[state_index] for row in Qs]
            # max=np.max(possible_Qs)
            # ids=[]
            # for i in range(0,len(possible_Qs)):
            #     Q=possible_Qs[i]
            #     if Q==max:
            #         max=Q
            #         ids.append(i)
            # action = np.random.choice(ids)
            # action = np.argmax(possible_Qs)

            #code from tutorial:
            max = np.max(Qs[state_index,:])
            temp_array = np.array(Qs[state_index][:])
            return np.random.choice(np.where(temp_array==max)[0])

        return action
class AbstractRLTask():

    def __init__(self, env, agent):
        """
        This class abstracts the concept of an agent interacting with an environment.


        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """

        self.env = env
        self.agent = agent

    def interact(self, n_episodes):
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode avergae returns  (see assignment for a definition
        """
        raise NotImplementedError()


    def visualize_episode(self, max_number_steps = None):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """

        raise NotImplementedError()

class Custom_RLTask_Learning_MC(AbstractRLTask):
    def __init__(self, env, agent,roomID, discountF, Qvalues=None):
        super().__init__(env, agent)
        action_count = self.env.action_space.n
        self.roomid = roomID
        self.useDoulbeEnv=True
        if self.roomid=="room-with-lava" or self.roomid=="empty-room":
            self.useDoulbeEnv=False

        if not self.useDoulbeEnv:
            start_state= get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation))
        else:
            start_state= get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation))

        env_height, env_width = len(start_state),len(start_state[0])

        states_list=[[(i,j) for j in range(0,env_width)] for i in range(0,env_height)]
        self.states_list = [item for subList in states_list for item in subList]
        self.Qs = [[0 for _ in range(env_width*env_height)] for _ in range(action_count)]
        # self.Returns = [[[] for _ in range(env_width*env_height)] for _ in range(action_count)]
        self.Returns = [[[] for _ in range(action_count)] for _ in range(env_width*env_height)]

        if Qvalues is None:
            self.Qmatrix = np.zeros((env_width*env_height, 4))
        else:
            self.Qmatrix=Qvalues
        self.visit_counts = np.zeros((env_width*env_height, 4))


        # self.Qdict = {}
        # self.visit_counts = {}
        # for state_id in range(len(states_list)):
        #     self.Qdict[state_id] ={}
        #     self.visit_counts[state_id] ={}
        #     for action in range(action_count):
        #         self.Qdict[state_id][action] = 0
        #         self.visit_counts[state_id][action] = 0


        self.discountF = discountF

    def interact(self, n_episodes):
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode avergae returns  (see assignment for a definition
        """
        average_returns = []


        # rewards = 0
        for i in range(n_episodes):
            curr_state = self.env.reset()
            curr_state = copy.deepcopy(get_crop_chars_from_observation(curr_state))
            # average_return=0
            sum_rewards = 0 # = G = return
            curr_reward=0

            #generate episode:
            episodes=[]
            while True:
                # if not self.useDoulbeEnv:
                #
                #     curr_state =copy.deepcopy(get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation)))
                # else:
                #     curr_state = copy.deepcopy(
                #         get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation)))


                # let agent choose action
                action = self.agent.act(curr_state, curr_reward, self.Qmatrix, self.states_list)
                # perform action on env and see results and
                next_observation, reward, terminated, info = self.env.step(action)
                curr_reward=reward
                sum_rewards += reward

                episodes.append((curr_state, action,reward))
                curr_state = copy.deepcopy(get_crop_chars_from_observation(next_observation))

                if terminated:
                    break

            T=len(episodes)-1
            G=0
            counter = 0
            for St,At,Rt in reversed(episodes):
                G = G + self.discountF*Rt
                firstVisit = True
                for j in range(counter+1, len(episodes)):
                    S = episodes[::-1][j][0]
                    A = episodes[::-1][j][1]
                    if np.array_equal(S,St) and A==At:
                        firstVisit=False
                        break
                if firstVisit:
                    user_row, user_col = extract_user_location_from_state(St)
                    state_index = self.states_list.index((user_row, user_col))

                    # self.Returns[state_index][At].append(G)
                    self.visit_counts[state_index][At] +=1
                    n=self.visit_counts[state_index][At]
                    self.Qmatrix[state_index, At] = self.Qmatrix[state_index, At] +(1/n)*(G - self.Qmatrix[state_index, At])

                counter+=1
            # for t in range(T-1,0,-1):
            #     _,_,Rt1 = episodes[t+1]
            #     St,At,Rt = episodes[t]
            #     G = G + self.discountF *Rt1
            #     #test if its first visit
            #     test=False
            #     for S,A,_ in episodes[:t]:
            #         if np.array_equal(S,St) and np.array_equal(A,At):
            #             test=True
            #             break
            #     if not test: #first visit MC
            #         user_row,user_col = extract_user_location_from_state(St)
            #         state_index = self.states_list.index((user_row, user_col))
            #
            #         self.Returns[state_index][At].append(G)
            #         n = len(self.Returns[state_index][At])
            #         Qn=self.Qmatrix[state_index, At]
            #         update = Qn +(Rt-Qn)/n
            #         # self.Qmatrix[state_index, action] = update #incremental approach --> doesnt seem to work (not converging)
            #         # self.Qmatrix[At][ state_index] = np.average(self.Returns[At][state_index]) #naive average
            #         # self.Qmatrix[state_index, action] = np.average(self.Returns[state_index][At])
            #         self.Qmatrix[state_index, action] = update

            average_return = (sum(average_returns)+sum_rewards)/(i+1)
            average_returns.append(average_return)
            print(sum_rewards)
            print("episode "+str(i)+" done")

        return average_returns


    def visualize_episode(self, max_number_steps = None,save_im=False):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """

        self.agent.learning = False
        curr_reward=0
        timestep=0
        state = self.env.reset()
        sum_rewards=0
        curr_state = get_crop_chars_from_observation(state)
        while True:
            # if not self.useDoulbeEnv:
            #     curr_state = get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation))
            # else:
            #     curr_state = get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation))

            # let agent choose action
            action = self.agent.act(curr_state, curr_reward, self.Qmatrix, self.states_list)

            # perform action on env and see results and
            observation, reward, terminated, info = self.env.step(action)
            curr_reward = reward
            timestep+=1

            sum_rewards+=reward
            curr_state = get_crop_chars_from_observation(observation)
            # self.env.render(action, reward)
            # print("Initial state", commons.get_crop_chars_from_observation(state))
            plt.imshow(get_crop_pixel_from_observation(observation))

            # if not self.useDoulbeEnv:
            #     plt.imshow(get_crop_pixel_from_observation(self.env._get_observation(self.env.last_observation)))
            # else:
            #     plt.imshow(get_crop_pixel_from_observation(self.env.env._get_observation(self.env.last_observation)))
            if save_im:
                plt.savefig("experiment_results2/step"+str(timestep)+"MC_cliff.png")

            plt.show()

            if terminated or (max_number_steps!=None and timestep==max_number_steps):
                print("episode terminated with a reward of "+str(sum_rewards))
                break

class Custom_RLTask_Learning_TD_OnPolicy(AbstractRLTask):
    def __init__(self, env, agent,alpha,discount_factor,roomID):
        super().__init__(env, agent)
        self.agent.learning=True
        action_count = self.env.action_space.n
        self.roomid = roomID
        self.useDoulbeEnv=True
        if self.roomid=="room-with-lava" or self.roomid=="empty-room":
            self.useDoulbeEnv=False
        if not self.useDoulbeEnv:
            start_state= get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation))
        else:
            start_state= get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation))

        env_height, env_width = len(start_state),len(start_state[0])

        states_list=[[(i,j) for j in range(0,env_width)] for i in range(0,env_height)]
        self.states_list = [item for subList in states_list for item in subList]
        self.Qs = [[0 for _ in range(env_width*env_height)] for _ in range(action_count)]
        # self.Returns = [[[] for _ in range(env_width*env_height)] for _ in range(action_count)]
        self.alpha = alpha
        self.discountF = discount_factor #gamma

        self.actionNumber = env.action_space.n
        self.Qmatrix = np.zeros((env_width*env_height, self.actionNumber))


    def interact(self, n_episodes):
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode avergae returns  (see assignment for a definition
        """
        average_returns = []


        # rewards = 0
        counter=0
        for i in range(n_episodes):
            if i==50:
                x=2
            state = self.env.reset()
            curr_reward=0
            curr_state = copy.deepcopy(get_crop_chars_from_observation(state))
            action = self.agent.act(curr_state, curr_reward, self.Qmatrix, self.states_list)

            sum_rewards = 0 # = G = return
            done=False

            #generate episode:
            episodes=[]
            terminated=False
            update_count = 0
            while True:
                if terminated:
                    break
                # if not self.useDoulbeEnv:
                #     curr_state =copy.deepcopy(get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation)))
                # else:
                #     curr_state = copy.deepcopy(
                #         get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation)))

                # perform action on env and see results and

                next_observation, reward, terminated, info = self.env.step(action) #take action, observe R and S'
                curr_reward=reward
                sum_rewards += reward

                next_state = get_crop_chars_from_observation(next_observation)


                if not terminated:
                    next_action = self.agent.act(next_state, curr_reward, self.Qmatrix, self.states_list)#choose A' from S'
                    next_user_row, next_user_col = extract_user_location_from_state(next_state)
                    next_state_index = self.states_list.index((next_user_row, next_user_col))

                # next_action = self.agent.act(next_state, curr_reward, self.Qs, self.states_list)#choose A' from S'
                user_row, user_col = extract_user_location_from_state(curr_state)
                state_index = self.states_list.index((user_row, user_col))

                if not terminated:
                    term = reward + self.discountF*self.Qmatrix[next_state_index,next_action]-self.Qmatrix[state_index, action]
                    self.Qmatrix[state_index,action] = self.Qmatrix[state_index,action]+self.alpha*term
                else:
                    term = reward - self.Qmatrix[state_index,action]
                    self.Qmatrix[state_index,action]=self.Qmatrix[state_index,action] + self.alpha*term
                    break
                #
                # user_row, user_col = extract_user_location_from_state(curr_state)
                # next_user_row, next_user_col = extract_user_location_from_state(next_state)
                # state_index = self.states_list.index((user_row, user_col))
                # next_state_index = self.states_list.index((next_user_row, next_user_col))
                # Q = self.Qs[action][state_index]
                # Qnext = self.Qs[next_action][next_state_index]
                # term = reward + self.discountF*Qnext - Q
                # self.Qs[action][state_index] =  Q+ self.alpha*term

                done  = terminated
                action = next_action
                curr_state = copy.deepcopy(next_state)
                update_count+=1
                if terminated:
                    xdd=2
                    print(xdd)



            average_return = (sum(average_returns)+sum_rewards)/(i+1)
            average_returns.append(average_return)
            print("episode "+str(i)+" done, sum rewards: "+str(sum_rewards))
            # counter+=1
            # if counter==12:
            #     self.agent.epsilon= self.agent.epsilon - 0.1
            #     counter=0
            #     print("new epsilon: "+str(self.agent.epsilon))

        return average_returns


    def visualize_episode(self, max_number_steps = None,save_im=False):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """

        self.agent.learning = False
        curr_reward=0
        timestep=0
        state = self.env.reset()
        curr_state = get_crop_chars_from_observation(state)
        sum_rewards=0
        while True:

            # let agent choose action
            action = self.agent.act(curr_state, curr_reward, self.Qmatrix, self.states_list)

            # perform action on env and see results and
            observation, reward, terminated, info = self.env.step(action)
            curr_reward = reward
            timestep+=1

            sum_rewards+=reward
            curr_state = get_crop_chars_from_observation(observation)
            # self.env.render(action, reward)
            # print("Initial state", commons.get_crop_chars_from_observation(state))
            plt.imshow(get_crop_pixel_from_observation(observation))
            if save_im:
                plt.savefig("experiment_results/step"+str(timestep)+"_OnPolicy_cliff.png")

            plt.show()

            if terminated or (max_number_steps!=None and timestep==max_number_steps):
                print("episode terminated with a reward of "+str(sum_rewards))
                break


class Custom_RLTask_Learning_TD_OffPolicy(AbstractRLTask):
    def __init__(self, env, agent,alpha,discount_factor,roomID):
        super().__init__(env, agent)
        action_count = self.env.action_space.n

        self.roomid = roomID
        self.useDoulbeEnv = True
        if self.roomid == "room-with-lava" or self.roomid == "empty-room":
            self.useDoulbeEnv = False

        if not self.useDoulbeEnv:
            start_state= get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation))
        else:
            start_state = get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation))

        env_height, env_width = len(start_state),len(start_state[0])

        states_list=[[(i,j) for j in range(0,env_width)] for i in range(0,env_height)]
        self.states_list = [item for subList in states_list for item in subList]
        self.Qs = [[0 for _ in range(env_width*env_height)] for _ in range(action_count)]
        # self.Returns = [[[] for _ in range(env_width*env_height)] for _ in range(action_count)]
        self.alpha = alpha
        self.discountF = discount_factor

        self.actionNumber = env.action_space.n
        self.Qmatrix = np.zeros((env_width*env_height, self.actionNumber))

    def interact(self, n_episodes):
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode avergae returns  (see assignment for a definition
        """
        average_returns = []


        # rewards = 0
        for i in range(n_episodes):
            self.env.reset()

            sum_rewards = 0 # = G = return
            curr_reward=0

            episodes=[]
            while True:
                if not self.useDoulbeEnv:
                    curr_state =copy.deepcopy(get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation)))
                else:
                    curr_state = copy.deepcopy(
                        get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation)))

                action = self.agent.act(curr_state, curr_reward, self.Qmatrix, self.states_list)
                # perform action on env and see results and
                next_observation, reward, terminated, info = self.env.step(action) #take action, observe R and S'
                curr_reward=reward
                sum_rewards += reward
                next_state = get_crop_chars_from_observation(next_observation)
                if terminated:
                    max_Q_next=0
                else:
                    next_user_row, next_user_col = extract_user_location_from_state(next_state)
                    next_state_index = self.states_list.index((next_user_row, next_user_col))
                    max_Q_next = np.max(self.Qmatrix[next_state_index, :])

                user_row, user_col = extract_user_location_from_state(curr_state)
                state_index = self.states_list.index((user_row, user_col))
                Q = self.Qmatrix[state_index, action]


                # max_Q_next = max( [row[next_state_index] for row in self.Qs] )
                term = reward + self.discountF*max_Q_next - Q

                self.Qmatrix[state_index, action]  =  Q+ self.alpha*term
                # self.Qs[action][state_index] =  Q+ self.alpha*term

                curr_state = next_state

                if terminated:
                    break


            average_return = (sum(average_returns)+sum_rewards)/(i+1)
            average_returns.append(average_return)
            print(sum_rewards)
            print("episode "+str(i)+" done")

        return average_returns


    def visualize_episode(self, max_number_steps = None,save_im=False):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """

        self.agent.learning = False
        curr_reward=0
        timestep=0
        self.env.reset()
        sum_rewards=0
        while True:
            if not self.useDoulbeEnv:
                curr_state = get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation))
            else:
                curr_state = get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation))

            # let agent choose action
            action = self.agent.act(curr_state, curr_reward, self.Qmatrix, self.states_list)

            # perform action on env and see results and
            observation, reward, terminated, info = self.env.step(action)
            curr_reward = reward
            timestep+=1

            sum_rewards+=reward
            # self.env.render(action, reward)
            # print("Initial state", commons.get_crop_chars_from_observation(state))
            if not self.useDoulbeEnv:
                plt.imshow(get_crop_pixel_from_observation(self.env._get_observation(self.env.last_observation)))
            else:
                plt.imshow(get_crop_pixel_from_observation(self.env.env._get_observation(self.env.last_observation)))
            if save_im:
                plt.savefig("experiment_results/step"+str(timestep)+"_OffPolicy_cliff.png")

            plt.show()

            if terminated or (max_number_steps!=None and timestep==max_number_steps):
                print("episode terminated with a reward of "+str(sum_rewards))
                break


class Custom_RLTask_Learning_TD_OffPolicy_Dyna(AbstractRLTask):
    def __init__(self, env, agent,alpha,discount_factor,roomID):
        super().__init__(env, agent)
        action_count = self.env.action_space.n

        self.roomid = roomID
        self.useDoulbeEnv = True
        if self.roomid == "room-with-lava" or self.roomid == "empty-room":
            self.useDoulbeEnv = False

        if not self.useDoulbeEnv:
            start_state= get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation))
        else:
            start_state = get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation))

        env_height, env_width = len(start_state),len(start_state[0])

        states_list=[[(i,j) for j in range(0,env_width)] for i in range(0,env_height)]
        self.states_list = [item for subList in states_list for item in subList]
        self.Qs = [[0 for _ in range(env_width*env_height)] for _ in range(action_count)]
        # self.Returns = [[[] for _ in range(env_width*env_height)] for _ in range(action_count)]
        self.alpha = alpha
        self.discountF = discount_factor

        self.actionNumber = env.action_space.n
        self.Qmatrix = np.zeros((env_width*env_height, self.actionNumber))

        self.n = 10 #planning_steps
        self.model = {}
        self.state_actions = []
    def interact(self, n_episodes):
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode avergae returns  (see assignment for a definition
        """
        average_returns = []


        # rewards = 0
        for i in range(n_episodes):
            self.env.reset()

            sum_rewards = 0 # = G = return
            curr_reward=0

            episodes=[]
            while True:
                if not self.useDoulbeEnv:
                    curr_state =copy.deepcopy(get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation)))
                else:
                    curr_state = copy.deepcopy(
                        get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation)))


                action = self.agent.act(curr_state, curr_reward, self.Qmatrix, self.states_list)

                self.state_actions.append((curr_state,action))

                # perform action on env and see results and
                next_observation, reward, terminated, info = self.env.step(action) #take action, observe R and S'
                curr_reward=reward
                sum_rewards += reward
                next_state = get_crop_chars_from_observation(next_observation)

                if terminated:
                    max_Q_next=0
                else:
                    next_user_row, next_user_col = extract_user_location_from_state(next_state)
                    next_state_index = self.states_list.index((next_user_row, next_user_col))
                    max_Q_next = np.max(self.Qmatrix[next_state_index, :])

                user_row, user_col = extract_user_location_from_state(curr_state)
                state_index = self.states_list.index((user_row, user_col))
                Q = self.Qmatrix[state_index, action]


                # max_Q_next = max( [row[next_state_index] for row in self.Qs] )
                term = reward + self.discountF*max_Q_next - Q

                self.Qmatrix[state_index, action]  =  Q+ self.alpha*term
                # self.Qs[action][state_index] =  Q+ self.alpha*term

                #update model:
                if state_index not in self.model.keys():
                    self.model[state_index]={}
                self.model[state_index]= (reward, next_state_index)


                #loop n times tu do random updates to Q values:
                #volledig volgens: towardsdatascience reinforcement learning-model based lanning methods
                for ni in range(self.n):
                    random_index = np.random.choice(range(len(self.model.keys())))
                    random_state = list(self.model)[random_index]
                    random_index2 = np.random.choice(range(len(self.model[state].keys())))
                    random_action = list(self.model[state])[random_index2]
                    rand_reward, rand_newstate = self.model[random_state][random_action]

                    max_Q_next = np.max(self.Qmatrix[rand_newstate, :])

                    self.Qmatrix[random_state][random_action]+= self.alpha*(rand_reward +max_Q_next-self.Qmatrix[random_state][random_action])



                if terminated:
                    break


            average_return = (sum(average_returns)+sum_rewards)/(i+1)
            average_returns.append(average_return)
            print(sum_rewards)
            print("episode "+str(i)+" done")

        return average_returns


    def visualize_episode(self, max_number_steps = None):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """

        self.agent.learning = False
        curr_reward=0
        timestep=0
        self.env.reset()
        sum_rewards=0
        while True:
            if not self.useDoulbeEnv:
                curr_state = get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation))
            else:
                curr_state = get_crop_chars_from_observation(self.env.env._get_observation(self.env.last_observation))

            # let agent choose action
            action = self.agent.act(curr_state, curr_reward, self.Qmatrix, self.states_list)

            # perform action on env and see results and
            observation, reward, terminated, info = self.env.step(action)
            curr_reward = reward
            timestep+=1

            sum_rewards+=reward
            # self.env.render(action, reward)
            # print("Initial state", commons.get_crop_chars_from_observation(state))
            if not self.useDoulbeEnv:
                plt.imshow(get_crop_pixel_from_observation(self.env._get_observation(self.env.last_observation)))
            else:
                plt.imshow(get_crop_pixel_from_observation(self.env.env._get_observation(self.env.last_observation)))

            plt.show()

            if terminated or (max_number_steps!=None and timestep==max_number_steps):
                print("episode terminated with a reward of "+str(sum_rewards))
                break


class Custom_RLTask(AbstractRLTask):
    def __init__(self, env, agent):
        super().__init__(env, agent)

    def interact(self, n_episodes):
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode avergae returns  (see assignment for a definition
        """
        average_returns = []


        # rewards = 0
        for i in range(n_episodes):
            self.env.reset()
            # average_return=0
            sum_rewards = 0
            curr_reward=0

            while True:
                # let agent choose action
                action = self.agent.act(self.env.state, curr_reward)
                # perform action on env and see results and
                observation, reward, terminated, info = self.env.step(action)
                curr_reward=reward
                sum_rewards += reward
                # rewards+=reward

                if terminated:
                    break

            average_return = (sum(average_returns)+sum_rewards)/(i+1)
            average_returns.append(average_return)
            print("episode "+str(i)+" sum_rewards: "+str(sum_rewards))

        return average_returns


    def visualize_episode(self, max_number_steps = None,saveFig=False):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """

        curr_reward=0
        timestep=0
        self.env.reset()
        while True:
            curr_state = get_crop_chars_from_observation(self.env._get_observation(self.env.last_observation))
            # let agent choose action
            action = self.agent.act(curr_state, curr_reward)
            # action = self.agent.act(self.env.state, curr_reward)
            # perform action on env and see results and
            observation, reward, terminated, info = self.env.step(action)
            curr_reward = reward
            timestep+=1

            # self.env.render(action, reward)
            # print("Initial state", commons.get_crop_chars_from_observation(state))

            test = self.env.render("string")
            print(test)

            plt.imshow(get_crop_pixel_from_observation(self.env._get_observation(self.env.last_observation)))
            if saveFig:
                plt.savefig('step'+str(timestep)+'_1.2.png')

            plt.show()
            if terminated or (max_number_steps!=None and timestep==max_number_steps):
                break


def extract_user_location_from_state(state):
    agent = 64
    sol = np.argwhere(state == agent)
    agent_pos = sol[0]
    index_row = agent_pos[0]
    index_column = agent_pos[1]
    return index_row, index_column


blank = 32
def get_crop_chars_from_observation(observation):
    chars = observation["chars"]
    coords = np.argwhere(chars != blank)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    chars = chars[x_min:x_max + 1, y_min:y_max + 1]
    return chars


size_pixel = 16
def get_crop_pixel_from_observation(observation):
    coords = np.argwhere(observation["chars"] != blank)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    non_empty_pixels = observation["pixel"][x_min * size_pixel : (x_max + 1) * size_pixel, y_min * size_pixel : (y_max + 1) * size_pixel]
    return non_empty_pixels