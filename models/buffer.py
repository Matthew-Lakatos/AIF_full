class PPORolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.state_values, self.is_terminals = [], [], []

    def store(self, state, action, reward, done, logprob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.logprobs.append(logprob)
        self.state_values.append(value)

    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:]
        del self.rewards[:], self.state_values[:], self.is_terminals[:]
