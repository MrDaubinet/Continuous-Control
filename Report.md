[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: images/realWorld.gif "Robots"
[image3]: https://user-images.githubusercontent.com/10624937/42135610-c37e0292-7d12-11e8-8228-4d3585f8c026.gif "Pendulum"
[image4]: images/scores.png "Graph"
[image5]: images/agent_result.gif "Agent"

# Report: Continuous Control


## Implemented Algorithm
Due to the nature of the environment being a continuous control problem. The reinforcemenr learning agorithm needs to be able to work in a continuous space. This hard requirement means we have to use a deep learning approach where neural networks are used for continuous function approximation. When considering between Policy-based vs Value-based Methods. Policy-based methods are better suited for continuous action spaces. Udacity suggest using either the [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf) or [D4PG](https://openreview.net/pdf?id=SyZipzbCb). I chose to implement the [Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971.pdf), which is describes as an extension of Deep Q-learning to continuous tasks.

I based my code off of [this](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) repository, an implementation of DDPG with OpenAI Gym's Pendulum environment. 

![Pendulum][image3]

I copied the Actor and Critic models, as [found here](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py), but I adapted the number of hidden unites to 256 and added another layer of batch normalization. I copied the agent code, [found here](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py), then changed it to accomidate 20 environments. 

The ```Agent()``` code can be [found here](https://github.com/MrDaubinet/Continuous-Control/blob/master/agent.py) resulting DDPG algorithm can be seen below:
```python
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
        # get env information
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        # set the best score to negative infinity
        best_score = -np.inf
        # list containing scores from each episode
        mean_scores = []
        # last 100 scores											      
        mean_score_window = deque(maxlen=100)
        # average of last 100 scores							  
        mean_scores_window_avg = []										  
        # for each episode
        for i_episode in range(1, n_episodes+1):
            # reset environment
            env_info = env.reset(train_mode=True)[brain_name]   
            # get current state for each agent	      
            states = env_info.vector_observations                         
            num_agents = len(env_info.agents)
            agent.reset()
            # initialize score for each agent
            scores = np.zeros(num_agents)                           		  
            start_time = time.time()
            # for each iteration
            for t in range(max_t):
                # select an action
                actions = agent.act(states)	
                # send actions to environment							  
                env_info = env.step(actions)[brain_name]
                # get next state                   
                next_states = env_info.vector_observations
                # get reward                
                rewards = env_info.rewards
                # check if episode has finished							      
                dones = env_info.local_done                               
                # save experience to replay buffer,
                for state, action, reward, next_state, done in \
                  zip(states, actions, rewards, next_states, dones):
                    # Step through the experiences given by each environment
                    agent.step(state, action, reward, next_state, done, t)  
                # set the next states
                states = next_states
                # save the agents performance
                scores += rewards    
                # break if any environment is completed                            	
                if np.any(dones):                                       			
                    break
            # if the average score of the last 100 episodes, over all environments, is greater than the goal 
            # and 100 episode
            if mean_scores_window_avg[-1] >= solved_score and i_episode >= consec_episodes:
              if train_mode:
                    torch.save(agent.actor_local.state_dict(), actor_path)
                    torch.save(agent.critic_local.state_dict(), critic_path)  
                break

```

## Results
My algorithm was able to solve the environment in 23 episodes with an average of 31.8 over the first 100 episodes. Check the graph below to see how it trained.
```
Episode 90      Duration: 290.0         Mean Episode Scores: 38.52      Mean Consecutive Scores: 31.10
Episode 91      Duration: 290.0         Mean Episode Scores: 38.89      Mean Consecutive Scores: 31.19
Episode 92      Duration: 289.0         Mean Episode Scores: 39.22      Mean Consecutive Scores: 31.27
Episode 93      Duration: 289.0         Mean Episode Scores: 38.32      Mean Consecutive Scores: 31.35
Episode 94      Duration: 291.0         Mean Episode Scores: 35.74      Mean Consecutive Scores: 31.39
Episode 95      Duration: 287.0         Mean Episode Scores: 38.17      Mean Consecutive Scores: 31.47
Episode 96      Duration: 292.0         Mean Episode Scores: 39.21      Mean Consecutive Scores: 31.55
Episode 97      Duration: 294.0         Mean Episode Scores: 38.78      Mean Consecutive Scores: 31.62
Episode 98      Duration: 296.0         Mean Episode Scores: 38.03      Mean Consecutive Scores: 31.69
Episode 99      Duration: 301.0         Mean Episode Scores: 39.09      Mean Consecutive Scores: 31.76
Episode 100     Duration: 301.0         Mean Episode Scores: 38.99      Mean Consecutive Scores: 31.83
```

![real robots][image4]

The final agent at work:
![Result Agent][image5]

## Ideas for Future Work
* **Hyperparameter optimization** - Most algorithms can be tweeked to perform better for specific environments when by changeing the various hyper parameters. This could be investigated to find a more effective agent.
* **Priority Experience Replay** - Prioritized experience replay selects experiences based on a priority value that is correlated with the magnitude of error. This replaces the random selection of experiences with an approach that is more intelligent, as described in [this paper](https://arxiv.org/pdf/1511.05952.pdf). 
