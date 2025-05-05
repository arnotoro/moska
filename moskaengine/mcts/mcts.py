from math import log, sqrt

from moskaengine.utils.card_utils import choose_random_action


class Node:
    """

    """


    def __init__(self, parent=None, game_state=None):
        self.game_state = game_state
        self.is_end_state = game_state.is_end_state if game_state else None
        self.is_explored = False
        self.parent = parent
        self.children = {}
        self.W = 0  # Score
        self.N = 0  # Number of visits

    def get_game_state(self):
        """For preventing from finding all unsearched game states"""
        if self.game_state is not None:
            return self.game_state

        parent_state = self.parent.get_game_state().clone_for_rollout()
        for action_to_perform, child in self.parent.children.items():
            if child is self:
                parent_state.execute_action(action_to_perform)
                self.is_end_state = parent_state.is_end_state
                self.game_state = parent_state
                return parent_state
        else:
            raise BaseException("Unable to find child in the parent's children")

    def uct_select(self, expl_const):
        assert self.N > 0 and self.children

        const = expl_const * sqrt(log(self.N))
        best_value = float('-inf')
        best_action = None
        best_node = None

        for action, child in self.children.items():
            if child.N == 0:
                continue
            uct = (child.W / child.N) + const / sqrt(child.N)
            if uct > best_value:
                best_value = uct
                best_action = action
                best_node = child

        assert best_node is not None
        return best_action, best_node

class MCTS:
    """

    """
    def __init__(self):
        self.node = None
        self.expl_rate = 0.7
        # self.player = None

    def do_rollout(self, game_state, rollouts=1000, expl_rate=0.7):
        """

        """
        self.expl_rate = expl_rate
        # Create a new node
        self.node = Node(game_state=game_state)

        # Perform rollouts
        for rollout in range(rollouts):
            print(f"Rollout {rollout}/{rollouts} for {game_state.player_to_play.name}", end="\r")

            leaf_node = self.select()
            self.expand(leaf_node)
            loser_name = self.simulate(leaf_node)
            self.backpropagate(leaf_node, loser_name)

        print(' ' * 50, end='\r')

        result = {action: (child.W, child.N) for action, child in self.node.children.items()}
        self.node = None
        return result

    def select(self):
        """Select a leaf node from the tree"""
        node = self.node

        while True:
            # Check if we stop traversing
            if not node.is_explored or node.is_end_state:
                return node

            # Otherwise, traverse to an unexplored child
            for child in node.children.values():
                if not child.is_explored:
                    return child

            _, node = node.uct_select(self.expl_rate)

    def expand(self, node):
        """Expand the leaf node from the tree, i.e. simulate all possible actions"""
        if node.is_end_state is None:
            node.get_game_state()

        if node.is_end_state:
            # Nothing to do
            return None

        game = node.get_game_state()
        allowed = game.allowed_plays()
        node.children = {action: Node(parent=node) for action in allowed}
        node.is_explored = True
        return None

    def simulate(self, node):
        """Play random game until an end state is reached, return loser"""
        # TODO: Check to stop after N steps
        if node.is_end_state:
            return node.get_game_state().loser.name

        game = node.get_game_state().clone_for_rollout()
        actions = list(node.children.keys())
        game.execute_action(choose_random_action(actions))

        # Traverse the tree
        while not game.is_end_state:
            game.execute_action(choose_random_action(game.allowed_plays()))

        # Return the loser
        return game.loser.name

    def backpropagate(self, node, loser_name):
        """Backpropagate the result of the simulation to the root node"""
        # Traverse the tree back to the root
        while node:
            # Update the number of visits
            node.N += 1
            parent = node.parent

            # Increase the score
            if parent:
                if parent.get_game_state().player_to_play.name != loser_name:
                    node.W += 1
            node = node.parent