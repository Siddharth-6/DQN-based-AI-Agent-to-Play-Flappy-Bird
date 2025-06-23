import pygame
import random
from config import Config

PIPE_IMG = pygame.image.load("pipes.png")
PIPE_FLIP_IMG = pygame.transform.flip(PIPE_IMG, False, True)  # Flip vertically

class Pipe:
    def __init__(self, x, height, inverted=False):
        self.config = Config.PIPE_CONFIG
        self.x = x
        self.height = height
        self.inverted = inverted
        self.passed = False
        
        # Get image dimensions
        self.img_height = PIPE_IMG.get_height()
        
        if inverted:
            # For inverted pipe, position it relative to top of screen
            # Offset y by negative height so pipe extends upward from top
            self.y = -(self.img_height - height)
        else:
            # For normal pipe, calculate y from bottom up
            self.y = Config.GAME_HEIGHT - Config.GROUND_CONFIG['height'] - height

    def get_rect(self):
        """Get collision rectangle"""
        if self.inverted:
            # For inverted pipe, collision rect starts at y=0 
            return pygame.Rect(self.x, 0, self.config['width'], self.height)
        return pygame.Rect(self.x, self.y, self.config['width'], self.height)
    
    def update(self, dt):
        """Update pipe position"""
        self.x -= self.config['speed']
    
    def draw(self, screen):
        # """Draw pipe on screen"""
        # screen.blit(self.surface, (self.x, self.y))
        image = PIPE_FLIP_IMG if self.inverted else PIPE_IMG
        screen.blit(image, (self.x, self.y))
        # for pipe in pipes:
        #     if pipe.bottom >= SCREEN_HEIGHT:
        #         screen.blit(PIPE_IMG, (self.x, self.y))
        #     else:
        #         screen.blit(PIPE_FLIP_IMG, pipe)
    
    def is_off_screen(self):
        """Check if pipe is completely off screen"""
        return self.x + self.config['width'] < 0

class PipeManager:
    def __init__(self):
        self.config = Config.PIPE_CONFIG
        self.pipes = []
        self.last_pipe_time = 0
        self.pipe_timer = 0
        self.min_spawn_time = self.config['min_spawn_time']  # Minimum time between pipes (ms)
        self.max_spawn_time = self.config['max_spawn_time']  # Maximum time between pipes (ms)
        self.next_spawn_time = random.randint(self.min_spawn_time, self.max_spawn_time)
        self.gap = random.randint(self.config['min_gap'],self.config['max_gap'])
    def update(self, dt):
        """Update all pipes"""
        self.pipe_timer += dt
        
        # Add new pipe pair
        if self.pipe_timer >= self.next_spawn_time:
            self.add_pipe_pair()
            self.pipe_timer = 0
            self.next_spawn_time = random.randint(self.min_spawn_time, self.max_spawn_time)
            
        # Update existing pipes
        for pipe in self.pipes[:]:
            pipe.update(dt)
            
            # Remove off-screen pipes
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
    
    def add_pipe_pair(self):
        """Add a pair of pipes (upper and lower)"""
        # Calculate vertical position for gap
        min_y = self.config['min_height']  # Minimum height from top of screen
        max_y = Config.GAME_HEIGHT - Config.GROUND_CONFIG['height'] - self.gap - self.config['min_height']
        
        # Ensure valid range
        if max_y <= min_y:
            max_y = min_y + 1
        
        gap_y = random.randint(min_y, max_y)
        
        # Upper pipe (inverted)
        upper_height = gap_y
        upper_pipe = Pipe(self.config['spawn_x'], upper_height, inverted=True)
        
        # Lower pipe
        lower_y = gap_y + self.gap
        lower_height = Config.GAME_HEIGHT - Config.GROUND_CONFIG['height'] - lower_y
        lower_pipe = Pipe(self.config['spawn_x'], lower_height, inverted=False)
        
        self.pipes.extend([upper_pipe, lower_pipe])
        self.gap = random.randint(self.config['min_gap'], self.config['max_gap'])
    def get_next_pipes(self, bird_x):
        """Get next pipe pair that bird will encounter"""
        for i in range(0, len(self.pipes), 2):
            if i + 1 < len(self.pipes):
                upper_pipe = self.pipes[i]
                lower_pipe = self.pipes[i + 1]
                
                if upper_pipe.x + upper_pipe.config['width'] > bird_x:
                    return upper_pipe, lower_pipe
        return None, None
    
    def check_collision(self, bird_rect):
        """Check collision with any pipe"""
        for pipe in self.pipes:
            if bird_rect.colliderect(pipe.get_rect()):
                return True
        return False
    
    def check_score(self, bird_x):
        """Check if bird passed pipes for scoring"""
        score_gained = 0
        
        for i in range(0, len(self.pipes), 2):
            if i + 1 < len(self.pipes):
                upper_pipe = self.pipes[i]
                
                if (not upper_pipe.passed and 
                    upper_pipe.x + upper_pipe.config['width'] < bird_x):
                    upper_pipe.passed = True
                    self.pipes[i + 1].passed = True  # Mark lower pipe as passed too
                    score_gained += 1
                    
        return score_gained
    
    def draw(self, screen):
        """Draw all pipes"""
        for pipe in self.pipes:
            pipe.draw(screen)
    
    def reset(self):
        """Reset pipe manager"""
        self.pipes.clear()
        self.pipe_timer = 0
