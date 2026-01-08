import pygame
import numpy as np
import os
from pathlib import Path
import sys
import tyro
from dataclasses import dataclass
from typing import Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.Env import UTTTEnv
from src.Opponent import FrozenAgentOpponent

@dataclass
class Args:
    model_path: Path
    """The model being inferred (plays as O)."""
    human_player: bool = True
    """Set to true to play as the Human (X) against the model (O)."""
    opponent_model_path: Optional[Path] = None
    """If not playing as a human, this is the path to the second model (plays as X)."""
    num_games: int = 3
    """The number of games to be played."""
    viewer: bool = True
    """If set, the pygame window will not be displayed. Human player requires the viewer."""
    slow_moves: bool = True
    """Determines if there's a pause before models make move so human can see it"""

def main():
    args = tyro.cli(Args)

    if args.human_player and not args.viewer:
        print("Error: A human player requires the viewer. Cannot use --human_player and set --viewer to false.")
        sys.exit(1)

    if not args.human_player and not args.opponent_model_path:
        print("Error: For Model vs. Model mode, --opponent_model_path must be provided.")
        sys.exit(1)
        
    use_viewer = args.viewer

    # Initialize environment
    env = UTTTEnv(render_mode="human" if use_viewer else None)

    # Setup Players
    model_o = FrozenAgentOpponent(name="Model_O", path_to_model=str(args.model_path))
    model_x = None
    if not args.human_player:
        model_x = FrozenAgentOpponent(name="Model_X", path_to_model=str(args.opponent_model_path))
    
    player_names = {
        1: "Human" if args.human_player else model_x.name,
        -1: model_o.name
    }

    scores = {player_names[1]: 0, player_names[-1]: 0, "Ties": 0}

    main_running = True
    for game_num in range(1, args.num_games + 1):
        if not main_running:
            break

        obs, info = env.reset()
        if use_viewer:
            env.render()
            pygame.display.set_caption(f"Ultimate Tic Tac Toe - Game {game_num}/{args.num_games}")

        game_over = False
        
        print("\n" + "="*20)
        print(f"Starting Game {game_num}/{args.num_games}")
        print(f"Player 1 (X): {player_names[1]}")
        print(f"Player 2 (O): {player_names[-1]}")
        print("="*20)

        while not game_over and main_running:
            # Check for quit event first
            if use_viewer:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        main_running = False
                        break
            if not main_running:
                break

            current_player_id = env.game.current_player
            player_name = get_player_name(env, player_names)
            
            is_human_turn = args.human_player and current_player_id == 1

            action = -1

            if is_human_turn:
                print(f"\nYour turn ({player_name})")
                # Handle Human Input
                action_taken = False
                while not action_taken and main_running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            main_running = False
                            break
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            x, y = event.pos
                            cell_size = env.game.window_size / 9
                            row, col = int(y // cell_size), int(x // cell_size)
                            action = (row * 9) + col
                            
                            legal_moves = env.game._get_legal_moves().flatten()
                            if legal_moves[action] == 1:
                                action_taken = True
                            else:
                                print("Invalid move! Please try again.")
                                action = -1
                    if not main_running:
                        break
            else:
                # Handle Model Input
                print(f"\n{player_name}'s turn...")
                if use_viewer and args.slow_moves:
                    pygame.time.wait(500)
                    pygame.event.pump()

                obs = env._get_obs()
                model_to_use = model_x if current_player_id == 1 else model_o
                action = model_to_use.pick_action(obs)

            if not main_running:
                break

            if action != -1:
                row, col = env.game._int_to_entry(action)
                print(f"{player_name} plays at: {action} (Row {row}, Col {col})")
                game_over = env.game.apply_action(action)
                if use_viewer:
                    env.render()

            if game_over:
                winner_name = check_winner(env, player_names)
                if winner_name == player_names[1]:
                    scores[player_names[1]] += 1
                elif winner_name == player_names[-1]:
                    scores[player_names[-1]] += 1
                else:
                    scores["Ties"] += 1
                
                print("\n" + "-"*20)
                print(f"Game {game_num} Over!")
                print(f"Winner: {winner_name}")
                print("Current Scores:")
                print(f"  {player_names[1]} (X): {scores[player_names[1]]}")
                print(f"  {player_names[-1]} (O): {scores[player_names[-1]]}")
                print(f"  Ties: {scores['Ties']}")
                print("-"*20)

            if use_viewer and env.game.clock:
                env.game.clock.tick(30)
        
        # Wait for 'n' key press for next game
        if use_viewer and main_running and game_num < args.num_games:
            print("\nPress 'N' to play the next game, or close the window to quit.")
            waiting_for_next = True
            while waiting_for_next and main_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        main_running = False
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                        waiting_for_next = False
                if not main_running:
                    break
    
    print("\nAll games finished!")
    env.close()
    
    print("\nAll games finished!")
    env.close()

def get_player_name(env, player_names):
    return player_names.get(env.game.current_player, "Unknown")

def check_winner(env, player_names):
    macro_state = env.game.mini_board_states.reshape(3, 3)
    winner_id = env.game._check_3x3_state(macro_state)
    
    if winner_id == 1:
        return player_names[1]
    elif winner_id == -1:
        return player_names[-1]
    else:
        return "Tie"

if __name__ == "__main__":
    main()