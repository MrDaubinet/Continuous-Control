from collections import deque
import time
import numpy as np
import torch

class DDPG():
    @staticmethod
    def train(env, agent, n_episodes=500, max_t=int(1000), train_mode=True, solved_score=30.0, consec_episodes=100, print_every=1, actor_path='actor_ckpt.pth', critic_path='critic_ckpt.pth'):
        """Deep Deterministic Policy Gradient (DDPG)

        Params
        ======
                n_episodes (int)      : maximum number of training episodes
                max_t (int)           : maximum number of timesteps per episode
                train_mode (bool)     : if 'True' set environment to training mode
                solved_score (float)  : min avg score over consecutive episodes
                consec_episodes (int) : number of consecutive episodes used to calculate score
                print_every (int)     : interval to display results
                actor_path (str)      : directory to store actor network weights
                critic_path (str)     : directory to store critic network weights
        """

        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        best_score = -np.inf
        mean_scores = []											      # list containing scores from each episode
        mean_score_window = deque(maxlen=100)							  # last 100 scores
        mean_scores_window_avg = []										  # average of last 100 scores

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]   	      # reset environment
            states = env_info.vector_observations                         # get current state for each agent
            num_agents = len(env_info.agents)
            agent.reset()
            
            scores = np.zeros(num_agents)                           		  # initialize score for each agent
            start_time = time.time()
            
            for t in range(max_t):
                actions = agent.act(states)								  # select an action
                env_info = env.step(actions)[brain_name]                   # send actions to environment
                next_states = env_info.vector_observations                # get next state
                rewards = env_info.rewards							      # get reward
                dones = env_info.local_done                               # see if episode has finished

                # save experience to replay buffer,
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    agent.step(state, action, reward, next_state, done, t)  

                states = next_states
                scores += rewards                                	
                if np.any(dones):                                       			
                    break

            duration = time.time() - start_time  						  # update time
            mean_scores.append(np.mean(scores))							  # save most recent score
            mean_score_window.append(mean_scores[-1])                     # add to consecutive scores
            mean_scores_window_avg.append(np.mean(mean_score_window))              # save the most recent consecutive score

            print('\rEpisode {} \tDuration: {:.1f} \tMean Episode Scores: {:.2f} \tMean Consecutive Scores: {:.2f}'.format(
                i_episode, round(duration), mean_scores[-1], mean_scores_window_avg[-1]))

            if train_mode and mean_scores[-1] > best_score:
                torch.save(agent.actor_local.state_dict(), actor_path)
                torch.save(agent.critic_local.state_dict(), critic_path)
                best_score = mean_scores[-1]

            if mean_scores_window_avg[-1] >= solved_score and i_episode >= consec_episodes:
                print('\nEnvironment SOLVED : \tMoving Average ={:.1f} over last {} episodes'.format(\
                                        mean_scores_window_avg[-1], consec_episodes))            
                if train_mode:
                    torch.save(agent.actor_local.state_dict(), actor_path)
                    torch.save(agent.critic_local.state_dict(), critic_path)  
                break

        return mean_scores, mean_scores_window_avg

    @staticmethod
    def test(agent, env, actor_path='actor_ckpt.pth', critic_path='critic_ckpt.pth'):
        # load best performing models
        agent.actor_local.load_state_dict(torch.load(actor_path))
        agent.critic_local.load_state_dict(torch.load(critic_path))

        brain_name = env.brain_names[0]

        # load environment variables
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        states = env_info.vector_observations              # get the current state
        scores = np.zeros(len(env_info.agents))            # initialize the score
        while True:
            actions = agent.act(states)                    # select an action
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished
            states = next_states                           # roll over the state to next time step
            scores += rewards                              # update the score
            if np.any(dones):                              # exit loop if episode finished
                break
            
        return scores
    