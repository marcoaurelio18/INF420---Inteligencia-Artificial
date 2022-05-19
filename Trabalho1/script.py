import sys
import math
        
my_team_id = int(input())  # if 0 you need to score on the right of the map, if 1 you need to score on the left

# coordinates of each goal
if my_team_id == 1:
    goal = [0,3750]
else:
    goal = [16000, 3750]

# class to save the methods of each entity
class Entities:
    def __init__(self, game_data):
        self.entity_id = game_data['entity_id']
        self.entity_type = game_data['entity_type']
        self.x = game_data['x']
        self.y = game_data['y']
        self.vx = game_data['vx']
        self.vy = game_data['vy']
        self.state = game_data['state']

# inheritance entities class, to manipulate wizards moves
class Wizard(Entities):
    @staticmethod
    def throw(x, y, power):
        print('THROW {} {} {}'.format(x, y, power))

    @staticmethod
    def move(x, y, thruster):
        print('MOVE {} {} {}'.format(x, y, thruster))

# inheritance entities class, it only serves to separate what is snaffle from what is wizard
class Snaffle(Entities):
    pass

'''
It is not necessary to save the opposing wizard, because if our wizard moves behind the snaffle and launches towards the goal,
he automatically prevents the enemy from scoring goals, as the enemy's intention is to move behind the snaffle too.
Avoiding memory waste and algorithm efficiency
'''
flag = True
# most important function, where will calculate which snaffle is closest to the wizard,
# and verified if other wizard is with the snaffle (or is in radius of other wizard) 
def closest_calculate(snaffles, wizard_cordinate, other_wizard):
    closest = snaffles[0]
    shortest_distance = 1000000
    for snaffle in snaffles:
        snaffle_cordinate = [snaffle.x, snaffle.y]
        dist = math.dist(snaffle_cordinate, wizard_cordinate)
        if other_wizard.state == 1:
            other_wizard_cordinate = [other_wizard.x, other_wizard.y]
            other_dist = math.dist(snaffle_cordinate, other_wizard_cordinate)
            if other_dist > 2000:
                if dist < shortest_distance:
                    shortest_distance = dist
                    closest = snaffle
        elif dist < shortest_distance:
            shortest_distance = dist
            closest = snaffle

    # returns the object that indicates the nearest snaffle
    return closest

# game loop
while True:
    wizards = list()
    opponent_wizzards = list()
    snaffles = list()
    bludgers = list()
    my_score, my_magic = [int(i) for i in input().split()]
    opponent_score, opponent_magic = [int(i) for i in input().split()]
    entities = int(input())  # number of entities still in game

    for i in range(entities):
        game_data = dict()
        inputs = input().split()
        game_data['entity_id'] = int(inputs[0])  # entity identifier
        game_data['entity_type'] = inputs[1]  # "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" (or "BLUDGER" after first league)
        game_data['x'] = int(inputs[2])  # position
        game_data['y'] = int(inputs[3])  # position
        game_data['vx'] = int(inputs[4])  # velocity
        game_data['vy'] = int(inputs[5])  # velocity
        game_data['state'] = int(inputs[6])  # 1 if the wizard is holding a Snaffle, 0 otherwise
        
        # save the wizards
        if game_data['entity_type'] == 'WIZARD':
            wizards.append(Wizard(game_data))

        # save the snaffles
        if game_data['entity_type'] == 'SNAFFLE':
            snaffles.append(Snaffle(game_data))
    
    for i in range(2):
        if i == 0:
            other_wizard = 1
        else:
            other_wizard = 0 
        
        wizard_cordinate = [wizards[i].x,wizards[i].y]
        
        # verify if the wizard is with the snaffle
        if wizards[i].state == 0:
            if opponent_score > my_score:
                opponent_snaffle = closest_calculate()
            closest = closest_calculate(snaffles, wizard_cordinate, wizards[other_wizard]) # find the nearest snaffle
            wizards[i].move(closest.x, closest.y, 130) # moves quickly behind the snaffle
        
        # if the wizard has the snaffle, he throw towards the goal with an ideal force 
        # (not too strong, not to isolate, nor too weak to take too long)
        else:
            if math.dist(wizard_cordinate, goal) > 3500:
                wizards[i].throw(goal[0], goal[1], 300)
            else:
                wizards[i].throw(goal[0], goal[1], 500)
