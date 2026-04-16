import gymnasium as gym
import pygame
import math
import sys

# 1. Initialize Gymnasium Environment
env = gym.make('CartPole-v1')
obs, info = env.reset()

# 2. Initialize Pygame
pygame.init()
screen_width = 600
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Playable CartPole - Use Left/Right Arrows")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CART_COLOR = (50, 50, 200)
POLE_COLOR = (200, 150, 100)

# Dimensions
cart_width = 50
cart_height = 30
pole_length = 120
pole_width = 10

running = True
# Default action (0 = push left, 1 = push right)
current_action = 0 

while running:
    # 3. Handle Human Input via Pygame Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    # Check which keys are currently being pressed down
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        current_action = 0  # Push Left
    elif keys[pygame.K_RIGHT]:
        current_action = 1  # Push Right

    # 4. Environment Step Logic
    # We pass the human-selected action to the environment
    obs, reward, terminated, truncated, info = env.step(current_action)

    # If you drop the pole or drive off-screen, it resets automatically
    if terminated or truncated:
        print("You dropped it! Resetting...")
        obs, info = env.reset()

    cart_x, cart_v, pole_angle, pole_v = obs

    # 5. Rendering Logic
    screen.fill(WHITE)

    # Scale the Gym coordinates to screen pixels
    scale_x = screen_width / 4.8 
    cart_pixel_x = int(screen_width / 2 + cart_x * scale_x)
    cart_pixel_y = int(screen_height / 2)

    # Draw the Cart
    cart_rect = pygame.Rect(
        cart_pixel_x - cart_width // 2,
        cart_pixel_y - cart_height // 2,
        cart_width,
        cart_height
    )
    pygame.draw.rect(screen, CART_COLOR, cart_rect)

    # Draw the Pole
    pole_end_x = cart_pixel_x + pole_length * math.sin(pole_angle)
    pole_end_y = cart_pixel_y - pole_length * math.cos(pole_angle)

    pygame.draw.line(
        screen,
        POLE_COLOR,
        (cart_pixel_x, cart_pixel_y),
        (pole_end_x, pole_end_y),
        pole_width
    )

    # Draw a ground line
    pygame.draw.line(screen, BLACK, (0, cart_pixel_y + cart_height // 2), 
                     (screen_width, cart_pixel_y + cart_height // 2), 2)

    # Draw instructions on the screen
    font = pygame.font.SysFont(None, 24)
    text = font.render("Use Left / Right arrow keys to balance!", True, BLACK)
    screen.blit(text, (20, 20))

    pygame.display.flip()
    
    # Run at 30 FPS. Standard is 50, but 30 gives humans slightly more reaction time!
    clock.tick(30) 

# Cleanup
env.close()
pygame.quit()
sys.exit()
