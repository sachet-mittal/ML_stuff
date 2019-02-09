"""
Refences: 
 - https://en.wikipedia.org/wiki/Expectiminimax
 - https://en.wikipedia.org/wiki/Minimax
 - https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
"""
import numpy as np
import math


MAX_PLAYER = 1
MIN_PLAYER = 2


def available_cols_to_move(board):
    valid_cols = []
    for col in range(board.shape[1]):
        if 0 in board[:,col]:
            valid_cols.append(col)
    return valid_cols


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def is_terminal_state(self, board):
        """
        Returns true if a player can win 
        """
        return len(available_cols_to_move(board)) == 0
    
    def get_available_row(self, board, column):
        """
        Returns the row available in the given column
        """
        for i in range(len(board) - 1, -1, -1):
            if board[i][column] == 0:
                return i
        
    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        alpha = -math.inf
        beta = math.inf
        depth = 5
        if self.player_number == MAX_PLAYER:
            column, _ = self.max_value_minimax(depth, board, alpha, beta)
        else:
            column, _ = self.min_value_minimax(depth, board, alpha, beta)
        return column

    def max_value_minimax(self, depth, board, alpha, beta, player=None):
        if player is None:
            player = self.player_number
        opponent = 3 - player
        if depth == 0 or not available_cols_to_move(board):
            result = self.evaluation_function(board)
            return None, result
        value = -math.inf
        column = 0
        columns = available_cols_to_move(board)
        #np.random.shuffle(columns)
        for col in columns:
            row = self.get_available_row(board, col)
            board_copy = board.copy()
            board_copy[row][col] = player
            new_score = self.min_value_minimax(depth-1, board_copy, alpha, beta, opponent)[1]
            if new_score >= value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if value >= beta:
                break
        return column, value

    def min_value_minimax(self, depth, board, alpha, beta, player=None):
        if player is None:
            player = self.player_number
        opponent = 3 - player
        if depth == 0 or not available_cols_to_move(board):
            result = self.evaluation_function(board)
            return None, result
        value = math.inf
        column = 0
        
        columns = available_cols_to_move(board)
        # np.random.shuffle(columns)
        for col in columns:
            row = self.get_available_row(board, col)
            board_copy = board.copy()
            board_copy[row][col] = player
            new_score = self.max_value_minimax(depth-1, board_copy, alpha, beta, opponent)[1]
            if new_score <= value:
                value = new_score
                column = col
            beta = min(beta, value)
            if value <= alpha:
                break
        return column, value

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them
        RETURNS:
        The 0 based index of the column that represents the next move
        Note: Probablity of each move is the same
        """
        depth = 4
        return self.max_value_expectimax(depth, board)[0]
    
    def max_value_expectimax(self, depth, board, player=None):
        if player is None:
            player = self.player_number
        opponent = 3 - player

        if depth == 0 or not available_cols_to_move(board):
            result = self.evaluation_function(board)
            return None, result
        
        value = -math.inf
        column = 0
        
        columns = available_cols_to_move(board)
        # np.random.shuffle(columns)
        for col in columns:
            row = self.get_available_row(board, col)
            board_copy = board.copy()
            board_copy[row][col] = player
            new_score = self.exp_value_expectimax(depth-1, board_copy, opponent)[1]
            if new_score >= value:
                value = new_score
                column = col
        return column, value
    
    def exp_value_expectimax(self, depth, board, player=None):
        if player is None:
            player = self.player_number
        opponent = 3 - player
        if depth == 0 or not available_cols_to_move(board):
            result = self.evaluation_function(board)
            return None, result
        value = 0
        columns = available_cols_to_move(board)
        # np.random.shuffle(columns)
        for col in columns:
            row = self.get_available_row(board, col)
            board_copy = board.copy()
            board_copy[row][col] = player
            # As probablity of each move is equal, multiply by 1/no.of columns
            value += self.max_value_expectimax(depth-1, board_copy, opponent)[1]/len(columns)
        return None, value

    def search_pattern_row(self, board, pattern):
        """
        Returns 1 if the specified pattern is found in the rows of the board.
        Else return 0
        """
        for row in board:
            row = "".join(map(str,row))
            if pattern in row:
                return 1
        return 0

    def search_pattern_diagonal(self, board, pattern):
        """
        Returns 1 if the specified pattern is found in board diagonally.
        Else return 0
        Refer: ConnectFour.py::check_diagonal
        """
        root_diag = np.diagonal(board, offset=0).astype(np.int)
        diagonal_str = "".join(map(str, root_diag))
        if pattern in diagonal_str:
            return 1

        for i in range(1, board.shape[1]-3):
            for offset in [i, -i]:
                diag = np.diagonal(board, offset=offset)
                diag = "".join(map(str, diag))
                if pattern in diag:
                    return 1
        return 0

    def score_player_position(self, board, player_number, count):
        """
        Returns 1 if the player player_number has the count pieces filled in continously.
        Else returns 0
        """
        str_to_search = "{}".format(player_number) * count
        # For rows
        c1 = self.search_pattern_row(board, str_to_search)
        # For Columns
        c2 = self.search_pattern_row(board.T, str_to_search)
        # For Diagonals
        c3 = self.search_pattern_diagonal(board, str_to_search)
        return c1 + c2 + c3
        
    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        h = 0
        player = self.player_number
        opponent = 3 - player #if player==2, oppnent=1 elif player=1, opponent=2

        # Does this board has 4 connected for player
        h += 1000 * self.score_player_position(board, player, 4)
        # Does this board has 3 connected for player
        h += 700 * self.score_player_position(board, player, 3)
        # Does this board has 2 connected for player
        h += 100 * self.score_player_position(board, player, 2)

        # Does this board has 4 connected for opponent
        h -= 800 * self.score_player_position(board, opponent, 4)
        # Does this board has 3 connected for opponent
        h -= 500 * self.score_player_position(board, opponent, 3)
        # Does this board has 2 connected for opponent
        h -= 50 * self.score_player_position(board, opponent, 2)
        return h

class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = available_cols_to_move(board)        
        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move