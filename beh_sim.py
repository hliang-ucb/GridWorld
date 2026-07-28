class NewWorldEnv(gym.Env):
    
    def __init__(self):
        self.size = 4
        self.n_states = self.size ** 2   # 16
        self.n_actions = 4                # 0=Up 1=Right 2=Down 3=Left
        self.action_space = spaces.Discrete(self.n_actions)
        
        self._action_to_delta = {
            0: ( 0, -1),   # Left
            1: ( 0,  1),   # Right
            2: ( 1,  0),   # Up
            3: ( -1, 0),   # Down
        }

        # Build once at construction time
        self.T = self._build_transition_matrix()

    
    def _build_transition_matrix(self):
        """
        T[s, a, s'] = probability of reaching s' from s via action a.

        Rules:
          - Terminal states (goal): agent stays put (absorbing).
          - Wall collisions: agent stays in current cell.
          - Otherwise: deterministic move → probability 1.0.
        """
        
        T = np.zeros((self.n_states, self.n_actions, self.n_states))

        for s in range(self.n_states):

            for a, (dr, dc) in self._action_to_delta.items():
                row, col = divmod(s, self.size)
                new_row  = max(0, min(self.size - 1, row + dr))
                new_col  = max(0, min(self.size - 1, col + dc))
                s_next   = new_row * self.size + new_col
                T[s, a, s_next] = 1.0   # deterministic

                if (row + dr < 0) | (col + dc < 0) | (row + dr > (self.size-1)) | (col + dc > (self.size-1)): 
                    T[s, a, s_next] = 0   # deterministic

        # remove the transition when there is no connection between two states
        T[0,2,4] = 0
        T[4,3,0] = 0
        T[2,2,6] = 0
        T[6,3,2] = 0
        T[4,1,5] = 0
        T[5,0,4] = 0
        T[10,1,11] = 0
        T[11,0,10] = 0
        T[11,2,15] = 0
        T[15,3,11] = 0
        T[12,1,13] = 0
        T[13,0,12] = 0
        
        return T

    
    def _build_reward_vector(self):
        """R[s] — reward received upon *entering* state s."""
        R = np.full(self.n_states, -0.1)
        R[self.goal] = 10.0
        return R


    def reset(self, goal, start, seed=None, options=None):
        super().reset(seed=seed)
        self.goal = goal
        self.agent_pos = start   
        self.R = self._build_reward_vector()
        self.steps = 0
        obs = self.agent_pos
        info = {}
        return obs, info

    
    def step(self, action, cutoff):

        self.agent_pos = np.where(self.T[self.agent_pos,action,:])[0][0]
        self.steps += 1

        # Rewards & termination
        reward = self.R[self.agent_pos]
        if self.agent_pos == self.goal:
            terminated = True
        else:
            terminated = False   

        truncated = self.steps >= cutoff          # episode cut-off, time-limit is 30s, 1 step is 
        obs  = self.agent_pos
        info = {"steps": self.steps}
        return obs, reward, terminated, truncated, info

    
    def close(self):
        pass

def simulate_a_session(data, model, nrep = 100):

    # this is the common structure across null models - mostly create a workable dataframe
        
    df = data.copy()
    ntrials = len(df)
    sim_trial = pd.DataFrame()
    sim_room = pd.DataFrame()
    env = NewWorldEnv()

    for n in range(nrep):
        
        if model is model3:
            Qa = np.zeros(4)
            
        if model is model4:
            Qsa = np.zeros((16,4))
            Qsa[env.T.sum(axis=2)==0] = np.nan

        for trial in range(ntrials):
                
            target = df.iloc[trial].target
            start = df.iloc[trial].start
            trial_df = df.iloc[[trial]][['Session','BlockNumber','EngagedTrialsInBlock', 'target', 'start','minus_step']].copy()
            cutoff = trial_df['minus_step'].values[0]
            
            obs, info = env.reset(target,start)
            action = []
            
            temp = trial_df.copy()
            temp[['nrep','step', 'current', 'graph_distance', 'degree']] = [n,0,obs, 
                                                nx.shortest_path_length(G, source=obs, target=target),G.degree[obs]]
            sim_room = pd.concat([sim_room,temp])
            
            while True:
                if model is model1:
                    obs, reward, terminated, truncated, info = model1(env, obs, info, cutoff=cutoff)
                if model is model2:
                    obs, reward, terminated, truncated, info, action = model(env,obs,info, action,cutoff=cutoff)
                if model is model3:
                    Qa, obs, reward, terminated, truncated, info = model(Qa,env,obs,cutoff=cutoff)
                if model is model4:
                    Qsa, obs, reward, terminated, truncated, info = model(Qsa,env,obs,cutoff=cutoff)

                temp = trial_df.copy()
                temp[['nrep','step', 'current', 'graph_distance', 'degree']] = [n,info['steps'],obs, 
                                                    nx.shortest_path_length(G, source=obs, target=target),G.degree[obs]]
                sim_room = pd.concat([sim_room,temp])
                
                if terminated:
                    break
                    
                if truncated:
                    break
            

            temp = df.iloc[[trial]][['Session','BlockNumber','EngagedTrialsInBlock', 'target', 'start', 'graph_distance', 'degree']].copy()
            temp[['nrep','minus_step', 'success', 'detour_index']] = [n,info['steps'],terminated, info['steps']/temp.graph_distance.values[0]]
            sim_trial = pd.concat([sim_trial,temp])

    return sim_trial, sim_room


