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


def custom_score(game, player):
    return combined_score(game, player)



def custom_score_1(game, player):
    return current_candidate(game,player)

def custom_score_2(game, player):
    return center_focused_score(game,player)




        
def combined_score_1(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"




    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 1.0
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score
      

def combined_score_2(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"


    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 2.0
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score 
   
     

def combined_score_3(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"


    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 3.0
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score 

     

def combined_score_4(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"


    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 4.0
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score 

     

def combined_score_5(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"


    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 4.5
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score 

     

def combined_score_6(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"


    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 6.0
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score 

     

def combined_score_7(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"


    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 7.0
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score         
    
     

def combined_score_8(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"


    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 8.0
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score 
     

def combined_score_9(game, player):
    #This heuristics combines the move score and center focused score.
    #It gives different weights to the above two at different stages of the game
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    blank_spaces = game.get_blank_spaces()
    board_size = game.height*game.width
    
    #First half of the game
    if 0<=len(blank_spaces)<= board_size/2.0:
        game_status = "First Half"
    #Second half of the game
    elif board_size/2.0 < len(blank_spaces) <= board_size:
        game_status = "Last Half"


    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    game_weight = 9.0
    
    #Set the scores with different emphasis.
    f_score = float( game_weight*score1 + score2 )
    l_score = float(score1 + game_weight*score2)
        
    if game_status == "First Half":
        return f_score
    elif game_status == "Last Half":
        return l_score         
def combo_score(game, player):    
    score3 = away_from_blocked_score(game,player)
    score1 = center_focused_score(game, player)
    score2 = moves_score(game, player)
    score = float(score1+2*score2)
    
    return score   
    

def moves_score(game, player ):    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        

    own_moves = len(game.get_legal_moves(player)) #Player's moves left
    opp_moves = len(game.get_legal_moves(game.get_opponent(player))) #Opponent's moves left
    own_weight = 1.0 #The weight of the player's moves left
    opp_weight  = 4.0 #The weight of the opponent's moves left

    score = float(own_weight*own_moves - opp_weight*opp_moves)

    return score



def away_from_blocked_score(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    opp_r, opp_c = game.get_player_location(game.get_opponent(player))
    r, c = game.get_player_location(player)
    blank_spaces = game.get_blank_spaces()
    neighbours = []
    for (square_r, square_c) in blank_spaces:
        if max(abs(square_r - r), abs(square_c - c)) == 1:
            neighbours.append( (square_r, square_c))
    score = float(len(neighbours))
    return score


    
def center_focused_score(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    r, c = game.get_player_location(player)
    
    #Find center cell
    rows = game.height
    cols = game.width
    center_r = int(rows/2)
    center_c = int(cols/2)
    assert center_r == center_c
    
    max_score = center_r +0.5 # Set up center score


    delta_r = abs(r - center_r)
    delta_c = abs(c - center_c)
    
    #The score will decrease as it gets closer and closer to the edges
    layer  = max(delta_r,delta_c)
    score = max_score - layer
    return score
    
    

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

    #def whatever()
    # building a book of open moves using board rotation and stuff
    
    
    
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
            move =()
            max_depth = game.height*game.width
            center_square = int(game.height/2), int(game.width/2)
            if center_square in legal_moves:
                m1 = center_square
            else:
                m1 = legal_moves[0]
            best_score = float("-inf")
            m2 = m1
            score = float("-inf")


            if self.iterative == True:
                if self.method == 'minimax':
                
                    depth = 1
                    while depth <= max_depth:# Iterate depths until there is no more moves left
                        #for m in legal_moves:
                            # Implement minimax search for every potential moves
                        score, move = self.minimax(game, depth)
                            # Return with the best moves of the current depth of search
                        if score > s:
                            s = score
                            m1 = move
                        # Store the score and best moves so far before funthering to the next search depth
                        best_score = s  
                        m2 = m1
                        depth = depth + 1
                        #If time is running out, return the best move so far
                        if self.time_left() < self.TIMER_THRESHOLD:
                            raise Timeout()
                        
                elif self.method == 'alphabeta':
                
                    depth = 0
                    while depth <= max_depth:# Iterate depths until there is no more moves left
                        #for m in legal_moves:
                            # Implement alpha-beta pruning for every potential moves
                        score, move = self.alphabeta(game, depth)
                            # Return with the best moves of the current depth of search
                        if score > s:
                            s = score
                            m1 = move
                        # Store the score and best moves so far before funthering to the next search depth        
                        best_score = s
                        m2 = m1
                        depth = depth + 1
                        #print (depth)
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
            #print (s, score, m2, m1)

                
                
            
            

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return m2

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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        moves = game.get_legal_moves()

        #If there is no possible moves any more, return the end-of-game values
        if not moves:
            return game.utility(self), (-1, -1)
        
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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        moves = game.get_legal_moves()
        
        #If there is no possible moves any more, return the end-of-game values
        if not moves:
            return game.utility(self), (-1, -1)
        
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

