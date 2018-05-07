# If you change these, you won't affect the server, so you can't cheat
PROJECT_FLAG = 3
KILL_POINTS = 0
#jferguson -> Modify sight range to allow for exact knowledge of opponents
if PROJECT_FLAG < 4:
  SIGHT_RANGE = 500 # Manhattan distance
else:
  SIGHT_RANGE = 5 # Manhattan distance
MIN_FOOD = 2
TOTAL_FOOD = 60 #Total food in the maze (both sides combined)

DUMP_FOOD_ON_DEATH = True # whether or not we dump dots on death

COLLISION_TOLERANCE = 0.7 # How close ghosts must be to Pacman to kill

LEGAL_POWERS = ['laser', 'speed', 'capacity', 'invisibility', 'respawn']
LASER_RANGE = 5 # max distance for shooting without full upgrade
FOOD_CAPACITY = [5, 10, float('inf')] # max dots that an agent can carry
SPEED_FROM_POINTS = [1.0, 1.67, 2.0]
INVISIBILITY_RANGE = [float('inf'), 5, 2]
RESPAWN_TIMES = [50, 10, 0]

BLAST_RADIUS = 2      # blast radius
SCARED_TIME = 20      # Timesteps ghosts are scared
SONAR_TIME = 40       # Timesteps that sonar power lasts
ARMOUR_TIME = 30      # Timesteps that armour power lasts
JUGGERNAUT_TIME = 10  # Timesteps that juggernaut power lasts
