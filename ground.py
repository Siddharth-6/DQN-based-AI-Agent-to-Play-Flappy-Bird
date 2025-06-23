import pygame
from config import Config

BASE_IMG = pygame.image.load("base.png")

class Ground:
    def __init__(self):
        self.config = Config.GROUND_CONFIG
        self.width = Config.GAME_WIDTH
        self.height = self.config['height']
        self.y = Config.GAME_HEIGHT - self.height
        
        # For scrolling effect
        self.x1 = 0
        self.x2 = self.width
        
        # Create ground surface
        # self.create_surface()
    
    # def create_surface(self):
    #     """Create ground visual representation"""
    #     self.surface = pygame.Surface((self.width, self.height))
    #     self.surface.fill(self.config['color'])
        
    #     # Add some texture lines
    #     for i in range(0, self.width, 20):
    #         pygame.draw.line(self.surface, (200, 180, 120), 
    #                        (i, 0), (i, self.height), 2)
    
    def update(self, dt):
        """Update ground scrolling"""
        self.x1 -= self.config['speed']
        self.x2 -= self.config['speed']
        
        # Reset positions for continuous scrolling
        if self.x1 <= -self.width:
            self.x1 = self.width
        if self.x2 <= -self.width:
            self.x2 = self.width
    
    def get_rect(self):
        """Get collision rectangle"""
        return pygame.Rect(0, self.y, Config.GAME_WIDTH, self.height)
    
    def draw(self, screen):
        """Draw ground on screen"""
        screen.blit(BASE_IMG, (self.x1, self.y))
        screen.blit(BASE_IMG, (self.x2, self.y))
    
    def reset(self):
        """Reset ground position"""
        self.x1 = 0
        self.x2 = self.width
