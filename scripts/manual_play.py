import pygame
import numpy as np
from src.Env import UTTTEnv

def main():
    # Initialize environment
    env = UTTTEnv()
    obs, info = env.reset()
    
    # Render once to open the window
    env.render()
    
    # Set window caption
    pygame.display.set_caption("Ultimate Tic Tac Toe - Manual Play")

    # Game variables
    running = True
    game_over = False
    
    # You are Player 1 (X), Agent is Player -1 (O)
    human_player = 1
    
    print(f"Game Start! You are {'X' if human_player == 1 else 'O'}.")
    if env.current_player == human_player:
        print("Your turn!")
    else:
        print("Agent's turn!")

    while running:
        # 1. Handle Pygame Events (Human Input)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Only handle clicks if it's human's turn and game isn't over
            if not game_over and env.current_player == human_player:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get mouse coordinates
                    x, y = event.pos
                    
                    # Convert pixel coordinates to grid index (0-80)
                    # Grid is 9x9, window is 540x540 -> 60px per cell
                    cell_size = env.window_size / 9
                    col = int(x // cell_size)
                    row = int(y // cell_size)
                    
                    action = (row * 9) + col
                    
                    # Check if move is valid
                    action_mask = env._get_action_mask_grid().flatten()
                    
                    if action_mask[action] == 1:
                        print(f"Human plays at: {action} (Row {row}, Col {col})")
                        obs, reward, terminated, truncated, info = env.step(action)
                        env.render()
                        print(f"Board States: \n {info['board_states'].reshape(3,3)}")
                        
                        if terminated:
                            game_over = True
                            check_winner(env)
                    else:
                        print("Invalid move! That tile is either occupied or not in the active board.")

        # 2. Agent Turn (Random)
        if not game_over and env.current_player != human_player:
            # Add a small delay so we can see the moves happen
            pygame.time.wait(500)
            pygame.event.pump() # Keep window responsive during wait

            # Get valid actions
            action_mask = env._get_action_mask_grid().flatten()
            valid_actions = np.where(action_mask == 1)[0]
            
            if len(valid_actions) > 0:
                # Pick a random valid action
                action = np.random.choice(valid_actions)
                print(f"Agent plays at: {action}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                
                if terminated:
                    game_over = True
                    check_winner(env)

        # Limit frame rate
        if env.clock:
            env.clock.tick(30)

    env.close()

def check_winner(env):
    # Check the macro board state
    macro_state = env.mini_board_states.reshape(3, 3)
    winner = env._check_3x3_state(macro_state)
    
    if winner == 1: print("\nGAME OVER: Player 1 (X) Wins!")
    elif winner == -1: print("\nGAME OVER: Player -1 (O) Wins!")
    else: print("\nGAME OVER: It's a Tie!")

if __name__ == "__main__":
    main()