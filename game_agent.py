"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import isolation


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

#class SearchDepthError(Exception):
#    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    own_weight = 1.0
    opp_weight = 2.0
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_weight*own_moves - opp_weight*opp_moves)
    #raise NotImplementedError


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left


        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        
        if not legal_moves:
            return (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            # Initialisation : s and m1 are for score and best moves for the current depth of search
            #best_score and m2 are for best score of the over all search
            s = float("-inf")
            m1 = legal_moves[0]
            best_score = float("-inf")
            m2 = legal_moves[0]
            


            if self.iterative == True:
                if self.method == 'minimax':
                
                    depth = 0
                    while m2 != (-1,-1):# Iterate depths until there is no more moves left
                        for m in legal_moves:
                            # Implement minimax search for every potential moves
                            score, _ = self.minimax(game.forecast_move(m), depth)
                            # Return with the best moves of the current depth of search
                            if score > s:
                                s = score
                                m1 = m
                        # Store the score and best moves so far before funthering to the next search depth
                        best_score = s  
                        m2 = m1
                        depth = depth + 1
                        #If time is running out, return the best move so far
                        if self.time_left() < self.TIMER_THRESHOLD:
                            raise Timeout()
                        
                elif self.method == 'alphabeta':
                
                    depth = 0
                    while m2 != (-1,-1):# Iterate depths until there is no more moves left
                        for m in legal_moves:
                            # Implement alpha-beta pruning for every potential moves
                            score, _ = self.alphabeta(game.forecast_move(m), depth)
                            # Return with the best moves of the current depth of search
                            if score > s:
                                s = score
                                m1 = m
                        # Store the score and best moves so far before funthering to the next search depth        
                        best_score = s
                        m2 = m1
                        depth = depth + 1
                        #If time is running out, return the best move so far
                        if self.time_left() < self.TIMER_THRESHOLD:
                            raise Timeout()
                                       
                      
                    
            else:
                #Complete search without iterative deepning
                
                if self.method == 'minimax':
                    for m in legal_moves:
                        #Implement minimax search, return with the best moves found
                        score, _ = self.minimax(game.forecast_move(m), self.search_depth)
                        if score > s:
                            s = score
                            m2 = m

                elif self.method == 'alphabeta':
                    for m in legal_moves:
                        #Implement alpha-beta pruning, return with the best moves found                       
                        score, _ = self.alphabeta(game.forecast_move(m), self.search_depth)
                        if score > s:
                            s = score
                            m2 = m

                
                
            
            

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return m1

        return m2
        # Return the best move from the last completed search iteration


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """


        # TODO: finish this function!

        #Initialisation: find all the possible moves, and store them in "moves" list
        moves = game.get_legal_moves()
        
        #If there is no possible moves any more, return the end-of-game values
        if not moves:
            return float("-inf"), (-1,-1)
        
        #If the current depth is root, return the current best score //
        #   and first move from the left end of the search tree
        if depth == 0:
            return self.score(game, self), moves[0]
        
        
        if maximizing_player == True: #Max level search
            best_score = float("-inf")
            move = ()
            
            for m in moves:
                #Take values from all the child nodes and store the node with maximum score
                v, _ = self.minimax(game.forecast_move(m), depth-1, maximizing_player = False)
                if v > best_score:
                    best_score = v
                    move = m


        
        else: # Min level Search
            best_score = float("inf")
            move = ()
            for m in moves: 
                #Take values from all the child nodes and store the node with minimum score 
                v, _ = self.minimax(game.forecast_move(m), depth-1, maximizing_player = True)
                if v < best_score:
                    best_score = v
                    move = m
       

        
        return best_score, move



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        #Initialisation: find all the possible moves, and store them in "moves" list
        moves = game.get_legal_moves()
        
        #If there is no possible moves any more, return the end-of-game values
        if not moves:
            return float("-inf"), (-1,-1)
        
        #If the current depth is root, return the current best score //
        #   and first move from the left end of the search tree
        if depth == 0:
            return self.score(game, self), moves[0]
            
        if maximizing_player == True: # Max level search
            move = ()
            for m in moves:
                #Take values from all the child nodes and store the node that has score higher than
                #lower boundary value alpha
                v, _ = self.alphabeta(game.forecast_move(m), depth-1, alpha, beta, maximizing_player = False)
                if v > alpha:
                    alpha = v #reset alpha with the new value
                    move = m
                
                #if any child branch has its highest score smaller than alpha, prune that branch             
                if beta <= alpha:
                    break

            return alpha, move

        
        else:
            move = ()
            for m in moves:
                #Take values from all the child nodes and store the node that has score lower than
                #upper boundary value beta
                v, _ = self.alphabeta(game.forecast_move(m), depth-1, alpha, beta, maximizing_player = True)
                if v < beta:
                    beta = v #reset beta with the new value
                    move = m
                
                #if any child branch has its lowest score larger than beta, prune that branch                             
                if beta <= alpha:
                    break
       
            return beta, move

