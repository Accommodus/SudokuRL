import gym
from gym import spaces
import numpy as np
import SudokuBoard


class SudokuEnv(gym.Env):
    """
    Custom Gym Environment for a Sudoku game.

    This environment allows an AI agent to interact with a Sudoku puzzle,
    providing actions to fill cells with numbers and receiving feedback on the validity of these actions.

    Attributes:
        sudoku (SudokuBoard): Instance of the SudokuBoard class representing the puzzle.
        row_length (int): The size of the Sudoku board (number of rows and columns).
        removed_cells (int): The number of cells initially empty in the puzzle.
        total_reward (int): The cumulative reward earned by the agent.
        max_actions (int): The maximum number of possible actions.
        action_space (spaces.Discrete): The action space of the environment.
        observation_space (spaces.Box): The observation space of the environment, representing the state of the board.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, removed_cells=20, row_length=9):
        super(SudokuEnv, self).__init__()
        self.sudoku = SudokuBoard.SudokuBoard(removed_cells, row_length)
        self.row_length = row_length
        self.removed_cells = removed_cells
        self.total_reward = 0

        self.max_actions = row_length * row_length * 9
        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Box(low=0, high=9, shape=(row_length, row_length), dtype=np.uint8)

    def map_action_to_cell(self, action):
        """
        Maps an action index to a cell (row, column) and a number to be placed in the cell.

        Args:
            action (int): The index of the action.

        Returns:
            tuple: The row and column indices and the number to be placed in the cell.
        """
        empty_cells = [(i, j) for i in range(self.row_length) for j in range(self.row_length) if
                       self.sudoku.board[i][j] == 0]
        if not empty_cells:
            return None, None, None

        cell_idx = action % len(empty_cells)
        row, col = empty_cells[cell_idx]
        number = (action // len(empty_cells)) % 9 + 1

        return row, col, number

    def valid_actions_count(self):
        """
        Counts the number of valid actions based on the current state of the board.

        Returns:
            int: The number of valid actions.
        """
        return sum(
            1 for i in range(self.row_length) for j in range(self.row_length) if self.sudoku.board[i][j] == 0) * 9

    def step(self, action):
        """
        Executes one step in the environment with the given action.

        Args:
            action (int): The index of the action to execute.

        Returns:
            numpy.array: The current state of the Sudoku board.
            int: The reward for the action.
            bool: Whether the current episode is done.
            dict: Additional information.
        """
        row, col, number = action

        if self.sudoku.board[row][col] == 0 and self.sudoku.is_valid(row, col, number):
            self.sudoku.set_value(number, row, col)
            self.total_reward += 1
            reward = 1
        else:
            reward = -1

        done = self.sudoku.is_full()
        return self._get_obs(), reward, done, {}

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            numpy.array: The initial state of the Sudoku board.
        """
        self.sudoku.reset_board()
        self.total_reward = 0
        return self._get_obs()

    def render(self, mode='console'):
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode in which to render the environment.
        """
        if mode != 'console':
            raise NotImplementedError()
        self.sudoku.print_board()

    def _get_obs(self):
        """
        Gets the current observation of the environment.

        Returns:
            numpy.array: The current state of the Sudoku board.
        """
        return np.array(self.sudoku.board)

    def close(self):
        pass
