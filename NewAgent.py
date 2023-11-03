
class NewAgent:


    def update(self, state, action, reward, next_state):
        # Get the old value
        old_value = self.q_table.get((state, action), 0)

        # Go through the different states, and get the max value of available next states
        next_max = max([self.q_table.get((next_state, a), 0) for a in range(self.action_space_size)])

        new_value = (old_value +
                     self.learning_rate * (reward + self.discount_factor * next_max - old_value))
