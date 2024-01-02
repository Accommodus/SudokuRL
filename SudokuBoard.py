import random
import math


class SudokuBoard:
    """
    SudokuBoard class for generating and managing a Sudoku puzzle.

    Attributes:
        board (list[list[int]]): The Sudoku board represented as a 2D list.
        removed_cells (int): Number of cells to remove from the complete board to create the puzzle.
        row_length (int): The size of the board (number of rows and columns).
        box_length (int): The size of each small box in the board.
        initial_board (list[list[int]]): A copy of the initial state of the board for reset purposes.

    Methods:
        __init__(removed_cells, row_length=9): Constructor that initializes the board and sets up the puzzle.
        is_valid(row, col, num): Checks if a number can be placed at a given row and column.
        set_value(value, row, column): Assigns a value to a specified cell on the board.
        is_full(): Checks if the board is completely filled.
        reset_board(): Resets the board to its initial state.
        print_board(): Prints the current state of the board to the console.
        _valid_in_row(row, num): Checks if a number can be placed in a given row (private).
        _valid_in_col(col, num): Checks if a number can be placed in a given column (private).
        _valid_in_box(row_start, col_start, num): Checks if a number can be placed in a specific 3x3 box (private).
        _fill_box(row, col): Fills a 3x3 box with numbers 1-9 (private).
        _fill_diagonal(): Fills the diagonal 3x3 boxes to start puzzle generation (private).
        _remove_cells(): Removes a specified number of cells from the board to create the puzzle (private).
        _fill_remaining(row, col): Fills the remaining cells of the board after the diagonal boxes (private).
        _fill_values(): Initiates the process of filling the board (private).
        _is_valid_board(): Checks if the current board configuration is valid (private).
        _is_valid_box(box_row, box_col): Checks if a 3x3 box is valid (private).
    """
    def __init__(self, removed_cells, row_length=9):
        self.board = [[0] * row_length for _ in range(row_length)]
        self.removed_cells = removed_cells
        self.row_length = row_length
        self.box_length = int(math.sqrt(row_length))

        self._fill_values()
        self._remove_cells()
        self.initial_board = self.board

    def _valid_in_row(self, row, num):
        return num not in self.board[row]

    def _valid_in_col(self, col, num):
        return all(self.board[row][col] != num for row in range(self.row_length))

    def _valid_in_box(self, row_start, col_start, num):
        for i in range(3):
            for j in range(3):
                if self.board[row_start + i][col_start + j] == num:
                    return False
        return True

    def is_valid(self, row, col, num):
        return all((self._valid_in_row(row, num),
                    self._valid_in_col(col, num),
                    self._valid_in_box(row - row % 3, col - col % 3, num)
                    ))

    def _fill_box(self, row, col):
        nums = list(range(1, self.row_length + 1))
        random.shuffle(nums)
        for i in range(3):
            for j in range(3):
                self.board[row + i][col + j] = nums.pop()

    def _fill_diagonal(self):
        for i in range(0, self.row_length, 3):
            self._fill_box(i, i)

    def _remove_cells(self):
        for _ in range(self.removed_cells):
            row = random.randint(0, self.row_length - 1)
            col = random.randint(0, self.row_length - 1)
            while self.board[row][col] == 0:
                row = random.randint(0, self.row_length - 1)
                col = random.randint(0, self.row_length - 1)
            self.board[row][col] = 0

    def _fill_remaining(self, row, col):
        if col >= self.row_length and row < self.row_length - 1:
            row += 1
            col = 0
        if row >= self.row_length and col >= self.row_length:
            return True
        if row < self.box_length:
            if col < self.box_length:
                col = self.box_length
        elif row < self.row_length - self.box_length:
            if col == int(row // self.box_length * self.box_length):
                col += self.box_length
        else:
            if col == self.row_length - self.box_length:
                row += 1
                col = 0
                if row >= self.row_length:
                    return True

        for num in range(1, self.row_length + 1):
            if self.is_valid(row, col, num):
                self.board[row][col] = num
                if self._fill_remaining(row, col + 1):
                    return True
                self.board[row][col] = 0
        return False

    def _fill_values(self):
        self._fill_diagonal()
        self._fill_remaining(0, self.box_length)

    def set_value(self, value, row, column):
        try:
            value = int(value)
            if not (1 <= value <= 9):
                raise ValueError('Value must be between 0 & 9')
            self.board[row][column] = value
        except Exception as exc:
            return exc

    def is_full(self):
        for row in self.board:
            if 0 in row:
                return False
        return True

    def _is_valid_board(self):
        for i in range(self.row_length):
            row_set = set()
            col_set = set()
            for j in range(self.row_length):
                if self.board[i][j] != 0:
                    if self.board[i][j] in row_set:
                        return False
                    row_set.add(self.board[i][j])
                if self.board[j][i] != 0:
                    if self.board[j][i] in col_set:
                        return False
                    col_set.add(self.board[j][i])
        for box_row in range(0, self.row_length, self.box_length):
            for box_col in range(0, self.row_length, self.box_length):
                if not self._is_valid_box(box_row, box_col):
                    return False
        return True

    def _is_valid_box(self, box_row, box_col):
        box_set = set()
        for i in range(self.box_length):
            for j in range(self.box_length):
                num = self.board[box_row + i][box_col + j]
                if num != 0:
                    if num in box_set:
                        return False
                    box_set.add(num)
        return True

    def reset_board(self):
        self.board = self.initial_board

    def print_board(self):
        for row in self.board:
            print(" ".join(str(num) for num in row))
