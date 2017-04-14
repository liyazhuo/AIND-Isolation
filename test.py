from isolation import Board
from sample_players import RandomPlayer, GreedyPlayer
from game_agent import CustomPlayer
from scipy.spatial.distance import pdist
import numpy

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [(a, b) for a in A for b in B]
player1 = CustomPlayer()
player2 = GreedyPlayer()
game = Board(player1, player2)

score = numpy.array([
    [0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0]])
for i in range (9):
    for j in range (9):
        score[i][j] = 0




rows = game.height
cols = game.width

print (rows)
for i in range (9):
    for j in range(9):
        game.apply_move((i,j))
        score[i][j] = len(game.get_legal_moves(player1))
print (game.to_string())
print (score)
    
    
    

#new_board = game.forecast_move((2,3))
#copy_board = game.copy()
#copy_board.apply_move((2,1))
#moves = game.get_legal_moves(player1)
#moves_new = new_board.get_legal_moves()
#print (moves)

#blankey = game.get_blank_spaces()
#print (blankey)
#game.apply_move((2,1))
#print (game.active_player())
#t1,t2 = game.get_player_location(player1)
#print (t1)
#print (t2)
#game.apply_move((3,3))
#print (game.get_player_location(player2))
#print (int(game.width/2))
#blankey = game.get_blank_spaces()
#d = pdist([(0,0),(4,4)])
#print (d)
#print (game.to_string())
#r, c = game.get_player_location(player2)
#print (r,c)

#center_r = int(rows/2)
#center_c = int(cols/2)
#assert center_r == center_c
#max_score = center_r
#print (max_score)



