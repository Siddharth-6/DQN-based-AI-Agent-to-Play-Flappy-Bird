import pygame
import numpy as np
from config import Config, GameState
from bird import Bird
from pipes import PipeManager
from ground import Ground

class FlappyBirdGame:
    def __init__(self, width=None, height=None):
        # Game dimensions
        self.width = width or Config.GAME_WIDTH
        self.height = height or Config.GAME_HEIGHT
        self.high_score = 0
        self.high_score_file = "high_score.txt"  # Add this line
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird - RL Training")
        self.clock = pygame.time.Clock()
        
        # Game objects
        self.bird = Bird()
        self.pipe_manager = PipeManager()
        self.ground = Ground()
        
        # Game state (original Flappy Bird variable names)
        self.state = GameState.SPLASH
        self.score = 0
        self.game_time = 0
        self.frame_count = 0
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)

    def load_high_score(self):
        """Load high score from file"""
        try:
            with open(self.high_score_file, "r") as file:
                self.high_score = int(file.read().strip())
        except (FileNotFoundError, ValueError):
            # If file doesn't exist or contains invalid data, start with 0
            self.high_score = 0

    def save_high_score(self):
        """Save high score to file"""
        try:
            with open(self.high_score_file, "w") as file:
                file.write(str(self.high_score))
        except IOError:
            print("Unable to save high score")

    def update_high_score(self):
        """Update high score if current score is higher"""
        if self.score > self.high_score:
            self.high_score = self.score
            self.save_high_score()
            return True  # Return True if new high score achieved
        return False

        
    def update(self, dt):
        """Update game state"""
        self.game_time += dt
        self.frame_count += 1
        
        if self.state == GameState.PLAYING:
            # Update game objects
            self.bird.update(dt)
            self.pipe_manager.update(dt)
            self.ground.update(dt)
            
            # Check collisions
            if self.check_collisions():
                self.game_over()
                
            # Check score
            score_gained = self.pipe_manager.check_score(self.bird.x)
            if score_gained > 0:
                self.score += score_gained
                # Check for new high score immediately when score increases
                self.update_high_score()
            
    def check_collisions(self):
        """Check all collision conditions"""
        bird_rect = self.bird.get_rect()
        
        # Check ground collision
        if bird_rect.colliderect(self.ground.get_rect()):
            return True
            
        # Check ceiling collision
        if self.bird.y < 0:
            return True
            
        # Check pipe collision
        if self.pipe_manager.check_collision(bird_rect):
            return True
            
        return False
    
    def flap(self):
        """Make bird flap"""
        if self.state == GameState.PLAYING:
            self.bird.flap()
        elif self.state == GameState.SPLASH:
            self.start_game()
        elif self.state == GameState.GAME_OVER:
            self.restart_game()
    
    def start_game(self):
        """Start the game"""
        self.state = GameState.PLAYING
        
    def game_over(self):
        """Handle game over"""
        self.state = GameState.GAME_OVER
        self.update_high_score()
    def restart_game(self):
        """Restart the game"""
        self.bird.reset()
        self.pipe_manager.reset()
        self.ground.reset()
        self.score = 0
        self.game_time = 0
        self.frame_count = 0
        self.state = GameState.PLAYING
    
    def draw(self):
        """Draw game on screen"""
        # Clear screen
        self.screen.fill(Config.SKY_COLOR)
        
        # Draw game objects
        self.pipe_manager.draw(self.screen)
        self.ground.draw(self.screen)
        self.bird.draw(self.screen)
        
        # Draw UI
        self.draw_ui()
        
        # Update display
        pygame.display.flip()
    
    
    def draw_ui(self):
        """Draw user interface"""
        # Draw current score (always visible)
        score_text = self.font_large.render(str(self.score), True, Config.TEXT_COLOR)
        score_rect = score_text.get_rect(center=(self.width // 2, 50))
        self.screen.blit(score_text, score_rect)
        
        # Draw high score in top right corner
        high_score_text = self.font_medium.render(f"Best: {self.high_score}", True, Config.TEXT_COLOR)
        high_score_rect = high_score_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(high_score_text, high_score_rect)
        
        if self.state == GameState.SPLASH:
            title_text = self.font_large.render("FLAPPY BIRD", True, Config.TEXT_COLOR)
            title_rect = title_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
            self.screen.blit(title_text, title_rect)
            
            start_text = self.font_medium.render("Click to Start", True, Config.TEXT_COLOR)
            start_rect = start_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
            self.screen.blit(start_text, start_rect)
            
        elif self.state == GameState.GAME_OVER:
            game_over_text = self.font_large.render("GAME OVER", True, Config.TEXT_COLOR)
            game_over_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 60))
            self.screen.blit(game_over_text, game_over_rect)
            
            # Show final score
            final_score_text = self.font_medium.render(f"Score: {self.score}", True, Config.TEXT_COLOR)
            final_score_rect = final_score_text.get_rect(center=(self.width // 2, self.height // 2 - 20))
            self.screen.blit(final_score_text, final_score_rect)
            
            # Show high score with special message if new record
            if self.score == self.high_score and self.score > 0:
                new_high_text = self.font_medium.render("NEW HIGH SCORE!", True, (255, 215, 0))  # Gold color
                new_high_rect = new_high_text.get_rect(center=(self.width // 2, self.height // 2 + 10))
                self.screen.blit(new_high_text, new_high_rect)
            
            restart_text = self.font_medium.render("Click to Restart", True, Config.TEXT_COLOR)
            restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 40))
            self.screen.blit(restart_text, restart_rect)

    def get_state(self):
        """Get current game state for RL agent"""
        # Get next pipe information
        upper_pipe, lower_pipe = self.pipe_manager.get_next_pipes(self.bird.x)
        
        if upper_pipe and lower_pipe:
            pipe_x = upper_pipe.x
            pipe_top_y = upper_pipe.height
            pipe_bottom_y = lower_pipe.y
            gap_center_y = (pipe_top_y + pipe_bottom_y) / 2
        else:
            pipe_x = Config.GAME_WIDTH
            pipe_top_y = 0
            pipe_bottom_y = Config.GAME_HEIGHT
            gap_center_y = Config.GAME_HEIGHT / 2
        
        state = {
            'bird_y': self.bird.y,
            'bird_vel_y': self.bird.velY,
            'pipe_x': pipe_x,
            'pipe_top_y': pipe_top_y,
            'pipe_bottom_y': pipe_bottom_y,
            'gap_center_y': gap_center_y,
            'score': self.score,
            'game_over': self.state == GameState.GAME_OVER
        }
        
        return state
    
    def get_state_array(self):
        """Get state as numpy array for RL training"""
        state = self.get_state()
        maxvel = max(abs(Config.BIRD_CONFIG['maxVelY']),abs(Config.BIRD_CONFIG['minVelY']))
        # Normalize values
        normalized_state = np.array([
            (state['bird_vel_y'] /maxvel),  # Bird velocity (-1 to 1 approx)
            (state['pipe_x'] - self.bird.x) / self.width,  # Horizontal distance to pipe
            (state['bird_y'] - state['gap_center_y'])/self.height,  # Vertical distance to gap center
            (state['pipe_bottom_y']-state['pipe_top_y'])/Config.PIPE_CONFIG['max_gap']
        ])
        
        return normalized_state
    
    def step(self, action):
        """Perform one step with given action (for RL interface)"""
        # Action: 0 = do nothing, 1 = flap
        if action == 1:
            self.flap()
        
        dt = self.clock.tick(Config.FPS)
        self.update(dt)
        
        # Initialize tracking variables if they don't exist
        if not hasattr(self, '_last_score'):
            self._last_score = 0
        if not hasattr(self, '_steps_alive'):
            self._steps_alive = 0
        
        reward = 0
        
        if self.state == GameState.GAME_OVER:
            # Death penalty
            reward = -10
            # Reset counters
            self._steps_alive = 0
        else:
            # Increment steps alive
            self._steps_alive += 1
            
            # Get current state info
            state = self.get_state()
            bird_y = self.bird.y
            gap_center_y = state['gap_center_y']
            gap_size = state['pipe_bottom_y'] - state['pipe_top_y']
            
            # 1. SURVIVAL REWARD (small but consistent)
            reward += 0.1
            
            # 2. BOUNDARY PENALTIES (critical safety)
            safety_margin = 50
            if bird_y < safety_margin:  # Too high
                penalty = -((safety_margin - bird_y) / safety_margin) * 2  # Progressive penalty
                reward += penalty
            elif bird_y > (self.height - safety_margin):  # Too low  
                penalty = -((bird_y - (self.height - safety_margin)) / safety_margin) * 2
                reward += penalty
            
            # 3. GAP POSITIONING REWARD (encourage centering)
            if safety_margin <= bird_y <= (self.height - safety_margin):  # Only if in safe zone
                distance_to_center = abs(bird_y - gap_center_y)
                max_distance = gap_size / 2
                
                if distance_to_center <= max_distance:
                    # Normalized distance: 0 (perfect center) to 1 (edge of gap)
                    normalized_distance = distance_to_center / max_distance
                    positioning_reward = (1 - normalized_distance) * 0.5  # 0 to 0.5 reward
                    reward += positioning_reward
            
            # 4. VELOCITY CONTROL (encourage smooth flight)
            abs_velocity = abs(self.bird.velY)
            max_safe_velocity = Config.BIRD_CONFIG['maxVelY'] * 0.7  # 70% of max
            
            if abs_velocity > max_safe_velocity:
                velocity_penalty = -((abs_velocity - max_safe_velocity) / max_safe_velocity) * 0.3
                reward += velocity_penalty
            
            # 5. HORIZONTAL PROGRESS (encourage moving toward pipes)
            pipe_distance = state['pipe_x'] - self.bird.x
            if 0 < pipe_distance < Config.GAME_WIDTH:
                # Small reward for being close to pipes (shows progress)
                progress_reward = 0.1 * (1 - pipe_distance / Config.GAME_WIDTH)
                reward += progress_reward
        
        # 6. MAJOR SCORING BONUS (pipe passed)
        if self.score > self._last_score:
            pipe_bonus = 10  # Significant but not overwhelming
            reward += pipe_bonus
            print(f"PIPE PASSED! +{pipe_bonus} reward - Score: {self.score}")
        
        # Update tracking
        self._last_score = self.score
        done = self.state == GameState.GAME_OVER
        
        # Optional: Clip rewards to prevent extreme values
        reward = np.clip(reward, -15, 15)
        
        return self.get_state_array(), reward, done