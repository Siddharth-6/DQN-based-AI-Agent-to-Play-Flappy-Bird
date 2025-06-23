from enum import Enum

class GameState(Enum):
    SPLASH = 0
    PLAYING = 1
    GAME_OVER = 2

class Config:
    # Game dimensions
    GAME_WIDTH = 400                 
    GAME_HEIGHT = 700  
    scaling_parameter = 1
    FPS = 60*scaling_parameter
    
    # Colors
    SKY_COLOR = (78, 192, 202)      # Sky blue
    GROUND_COLOR = (222, 216, 149)  # Ground color
    PIPE_COLOR = (0, 128, 0)        # Green pipes
    TEXT_COLOR = (255, 255, 255)    # White text
    
    # Bird config
    BIRD_CONFIG = {
        'width': 45,           
        'height': 32,          
        'x': 100,             
        'flapSpeed': -8*scaling_parameter,       # Upward velocity when flapping
        'maxVelY': 10*scaling_parameter,          # Maximum downward velocity
        'minVelY': -8*scaling_parameter,          # Maximum upward velocity
        'accY': 0.5*scaling_parameter,            # Downward acceleration (gravity)
        'rot': 45,              # Rotation angle
        'velRot': 3*scaling_parameter,            # Angular velocity
        'rotThr': 20*scaling_parameter,           # Rotation threshold
        'flapAcc': -9,          # Flap acceleration
    }
    
    # Pipe config
    PIPE_CONFIG = {
        'width': 52,                 # Increased from 52
        'height': 320,               # Increased from 288
        'min_gap': 130,
        'max_gap': 200,
        'min_spawn_time': 1500/scaling_parameter,
        'max_spawn_time': 2500/scaling_parameter,
        'speed': 2*scaling_parameter,
        'spawn_x': GAME_WIDTH + 10,  # Spawn just outside new screen width
        'min_height': 80,            # Increased from 50
        'max_height': 450            # Increased from 350
    }
    
    # Ground config
    GROUND_CONFIG = {
        'height': 150,         # Increased from 112
        'speed': 2*scaling_parameter,
        'color': (222, 216, 149)
    }
    
    # Game config
    GAME_CONFIG = {
        'score_sound': True,
        'collision_sound': True,
        'flap_sound': True
    }
