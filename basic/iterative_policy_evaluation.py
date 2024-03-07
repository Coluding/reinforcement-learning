import numpy as np
from grid_world import standard_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3  # threshold for convergence


def print_values(V, g):
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")


if __name__ == '__main__':

    ### define transition probabilities and grid ###
    # the key is (s, a, s'), the value is the probability
    # that is, transition_probs[(s, a, s')] = p(s' | s, a)
    # any key NOT present will considered to be impossible (i.e. probability 0)
    transition_probs = {}

    # to reduce the dimensionality of the dictionary, we'll use deterministic
    # rewards, r(s, a, s')
    # note: you could make it simpler by using r(s') since the reward doesn't
    # actually depend on (s, a)
    rewards = {}

    grid = standard_grid()
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    # compute transition probability of going from state s with action a to state s2
                    # here it has probability of 1 since it is deterministic
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]

    ### fixed policy ###
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L',
    }
    print_policy(policy, grid)

    # initialize V(s) = 0
    V = {}
    for s in grid.all_states():
        V[s] = 0

    gamma = 0.9  # discount factor
    # The action space shows all possible actions that can be taken in each state
    # The policy suggests which action to take in each step
    # In our example we set the policy to fixed because we want to demonstrate iterative policy evaluation,
    # where we develop the value function over time instead of choosing the correct policy
    # repeat until convergence
    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0  # we will accumulate the answer

                # Loop through each action in the action space to evaluate its impact on the state's value
                for a in ACTION_SPACE:
                    # Loop through all possible next states (s2) for each action
                    for s2 in grid.all_states():
                        # Determine the action probability based on the current policy
                        # If the action 'a' is the one suggested by the policy for state 's', this probability is 1 (deterministic policy)
                        # Otherwise, the action is not considered by the policy for state 's', and its probability is set to 0
                        action_prob = 1 if policy.get(s) == a else 0

                        # Get the reward for transitioning from state 's' to state 's2' via action 'a'
                        # If this specific transition does not have a defined reward, default to 0
                        r = rewards.get((s, a, s2), 0)

                        # Update the new value for state 's' based on the Bellman equation
                        # The update includes the action probability, transition probability (chance of going from 's' to 's2' via 'a'),
                        # the immediate reward 'r', and the discounted value of the next state 's2'
                        # This calculation is only significant if the action 'a' is part of the policy for state 's' (action_prob > 0)
                        # and if there is a nonzero chance of transitioning to 's2' from 's' via 'a'
                        # By doing that we ensure that only the value of s2 is considered if and only if s2 is a result
                        # of taking action a in state s
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])

                # after done getting the new value, update the value table
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        print("iter:", it, "biggest_change:", biggest_change)
        print_values(V, grid)
        it += 1

        if biggest_change < SMALL_ENOUGH:
            break
    print("\n\n")
