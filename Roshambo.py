import random

#Dict of moves
moves = {'R': "Rock",
         'P': "Paper",
         'S': "Scissors"}

#Game
def roshambo():
#Bot makes a move    
    bot = random.choice(['R', 'P', 'S']) #bot = input("R, P or S?") 
    print("Bot chose " + bot)   
#Opponent makes a move
    opp = random.choice(['R', 'P', 'S'])
    print("Opp chose " + opp)    
#Results
    if bot == opp:
        print("It's a TIE")
        return 0
    elif (bot == 'S' and opp == 'P') or (bot == 'R' and opp == 'S') or (bot == 'P' and opp == 'R'): 
        print("It's a WIN")
        return 1
    else:  
        print("It's a LOSE")
        return -1   
        
#Simulate n rounds 
def simul(n):
    score = 0
    for i in range(n):
        result = roshambo()
        score += result
        print("Score is ", score)
        print("======================")
    if score > 0:
        print('Bot wins the game')
    elif score == 0:
        print('Tie')
    else:
        print('Opp wins the game')    
