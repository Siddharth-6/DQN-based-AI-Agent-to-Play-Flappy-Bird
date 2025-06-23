import pygame
import sys
from game_runner import FlappyBirdGame
from config import GameState
from config import Config         

def main():
    """Main function to run the game manually (for testing)"""
    game = FlappyBirdGame()
    
    print("Flappy Bird Game")
    print("Controls:")
    print("  SPACE or CLICK - Flap")
    print("  ESC - Quit")
    print("\nClick to start...")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.flap()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                game.flap()
                
        # Update game
        dt = game.clock.tick(Config.FPS)
        game.update(dt)
        
        # Draw game
        game.draw()
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
