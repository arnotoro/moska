from math import log, sqrt

from moskaengine.utils.card_utils import choose_random_action


class Node:
    """

    """
    W = 0 # Score
    N = 0 # Number of visits

    def __init__(self, parent=None, game_state=None):
        if game_state is None:
            self.game_state = None
            self.is_end_state = None
        else:
            self.game_state = game_state
            self.is_end_state = game_state.is_end_state

        self.is_explored = False
        self.parent = parent
        self.children = {}

    def get_game_state(self):
        """For preventing from finding all unsearched game states"""
        if self.game_state is not None:
            return self.game_state
        assert self.parent is not None
        for action_to_perform, child in self.parent.children.items():
            if id(child) == id(self):
                game = self.parent.get_game_state().make_deepcopy()
                game.execute_action(action_to_perform)
                self.is_end_state = game.is_end_state
                return game
        else:
            raise BaseException('Unable to find child in the children of its parent')

    def uct_select(self, expl_const):
        child = []
        for action_played, mcts_node in self.children.items():
            N, W = mcts_node.N, mcts_node.W
            if N > 0:
                child.append((action_played, N, W))

        assert self.N > 0 # Check if node is explored
        assert len(child) > 0 # At least one child must be explored

        # Calculate the UCT value for each child
        const = expl_const * sqrt(log(self.N))
        def uct_value(args):
            _, N, W = args
            return W / N + const * N**(-0.5)

        # Choose best child
        best = max(child, key=uct_value)
        return best[0], self.children[best[0]]


class MCTS:
    """

    """
    def __init__(self):
        self.node = None
        self.expl_rate = 0.7
        self.player = None

    def do_rollout(self, game_state, rollouts=1000, expl_rate=0.7):
        """

        """
        self.expl_rate = expl_rate

        # Create a new node
        self.node = Node(game_state=game_state)

        # Perform rollouts
        for rollout in range(rollouts):
            print(f"Doing rollout {rollout} for {game_state.player_to_play.name}", end="\r")

            leaf_node = self.select()
            self.expand(leaf_node)
            lost_name = self.simulate(leaf_node)
            self.backpropagate(leaf_node, lost_name)
        print(' ' * 50, end='\r')

        # Return the information of each action with their performance
        dct = {}
        for action_played, child in self.node.children.items():
            dct[action_played] = (child.W, child.N)

        # Clear the node
        self.node = None
        return dct

    def select(self):
        """Select a leaf node from the tree"""

        mcts_node = self.node

        while True:
            # Check if we stop traversing
            if not mcts_node.is_explored:
                return mcts_node

            if mcts_node.is_end_state is None:
                mcts_node.get_game_state()

            if mcts_node.is_end_state:
                return mcts_node

            # Otherwise, traverse to an unexplored child
            for action, node in mcts_node.children.items():
                if not node.is_explored:
                    break
            else:
                action, node = mcts_node.uct_select(self.expl_rate)

            mcts_node = node

    def expand(self, leaf_node):
        """Expand the leaf node from the tree, i.e. simulate all possible actions"""
        if leaf_node.is_end_state is None:
            leaf_node.get_game_state()

        if leaf_node.is_end_state:
            # Nothing to do
            return None

        # Simulate all possible actions
        game = leaf_node.get_game_state()
        # Check which actions we're allowed to do
        allowed = game.allowed_plays()
        # Initialize the children of the leaf node
        leaf_node.children = {action: Node(parent=leaf_node) for action in allowed}
        leaf_node.is_explored = True

        return None

    def simulate(self, leaf_node):
        """Play random game until an end state is reached, return loser"""
        # TODO: Check to stop after N steps

        assert leaf_node.is_end_state is not None
        if leaf_node.is_end_state:
            return leaf_node.get_game_state().loser.name

        action = choose_random_action(list(leaf_node.children.keys()))
        # Execute
        game = leaf_node.get_game_state().make_deepcopy()
        game.execute_action(action)

        # Traverse the tree randomly
        while not game.is_end_state:
            # Execute a random action
            allowed = game.allowed_plays()
            action = choose_random_action(allowed)
            game.execute_action(action)

        # Return the loser
        return game.loser.name

    def backpropagate(self, mcts_node, lost_name):
        """Backpropagate the result of the simulation to the root node"""
        # Traverse the tree back to the root
        while mcts_node is not None:
            # Update the number of visits
            mcts_node.N += 1

            # Increase the score
            if mcts_node.parent is not None:
                if mcts_node.parent.get_game_state().player_to_play.name != lost_name:
                    mcts_node.W += 1
            mcts_node = mcts_node.parent