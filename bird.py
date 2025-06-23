import pygame
import math
from config import Config

BIRD_IMG = pygame.image.load("bird.png")

class Bird:
    def __init__(self):
        self.config = Config.BIRD_CONFIG
        
        # Position and physics (original Flappy Bird variable names)
        self.x = self.config['x']
        self.y = Config.GAME_HEIGHT // 2
        self.velY = 0  # Vertical velocity
        self.accY = self.config['accY']  # Gravity
        
        # Rotation
        self.rot = 0
        self.velRot = self.config['velRot']
        
        # Animation
        self.frame = 0
        self.animation_time = 0
        
        # Create bird surface
        # self.create_surface()
        
    # def create_surface(self):
    #     """Create bird visual representation"""
    #     self.surface = pygame.Surface((self.config['width'], self.config['height']))
    #     self.surface.fill(self.config['color'])
    #     # Add simple wing pattern
    #     pygame.draw.ellipse(self.surface, (255, 200, 0), 
    #                       (5, 5, self.config['width']-10, self.config['height']-10))
        
    def update(self, dt):
        """Update bird physics and animation"""
        # Update vertical velocity and position
        self.velY += self.accY
        
        # Clamp velocity
        if self.velY > self.config['maxVelY']:
            self.velY = self.config['maxVelY']
        elif self.velY < self.config['minVelY']:
            self.velY = self.config['minVelY']
            
        # Update position
        self.y += self.velY
        
        # Update rotation based on velocity
        if self.velY < 0:
            self.rot = max(-25, -25 * (-self.velY / self.config['minVelY']))
        else:
            self.rot = min(90, 25 * (self.velY / self.config['maxVelY']))
        
        # Update animation
        self.animation_time += dt
        if self.animation_time > 100:  # Change frame every 100ms
            self.frame = (self.frame + 1) % 3
            self.animation_time = 0
    
    def flap(self):
        """Make bird flap (jump)"""
        self.velY = self.config['flapSpeed']
        
    def get_rect(self):
        """Get collision rectangle"""
        return pygame.Rect(self.x, self.y, self.config['width'], self.config['height'])
    
    def draw(self, screen):
        """Draw bird on screen"""
        # Rotate image based on bird's rotation
        rotated_surface = pygame.transform.rotate(BIRD_IMG, self.rot)
        
        # Get new rect to center the rotated image
        rotated_rect = rotated_surface.get_rect(center=(
            self.x + self.config['width'] // 2,
            self.y + self.config['height'] // 2
        ))
        
        screen.blit(rotated_surface, rotated_rect)
    
    def reset(self):
        """Reset bird to initial state"""
        self.y = Config.GAME_HEIGHT // 2
        self.velY = 0
        self.rot = 0
        self.frame = 0
        self.animation_time = 0
