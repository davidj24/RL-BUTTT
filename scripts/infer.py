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
    """The model being inferred"""
    human_player: bool = True
    """Determines whether the game is between two models or a model and human. 0 for human, 1 for model"""
    opponent_model_path: str = ""
    """If two model's are playing against each other, this is the path to the second model"""
    num_games: int = 3
    """The number of games to be played"""
    viewer: bool = True
    """Whether pygame window should be displayed, needed for human player"""



def main():
    args = tyro.cli(Args)

    # Initialize environment
    env = UTTTEnv(render_mode="human" if args.viewer else None)

    # Setup Players
    model_p2 = FrozenAgentOpponent(name="Inferring_model", path_to_model=str(args.model_path))
    model_p1 = None
    if not args.human_player:
        if not args.opponent_model_path:
            print("Error: For Model vs. Model mode, --opponent_model_path must be provided.")
            sys.exit(1)
        model_p1 = FrozenAgentOpponent(name="Opponent_model", path_to_model=args.opponent_model_path)
    
    player_names = {
        1: "Human" if args.human_player else model_p1.name,
        -1: model_p2.name
    }

    scores = {player_names[1]: 0, player_names[-1]: 0, "Ties": 0}

    for game_num in range(1, args.num_games + 1):
        obs, info = env.reset()
        if args.viewer:
            env.render()
            pygame.display.set_caption(f"Ultimate Tic Tac Toe - Game {game_num}/{args.num_games}")

        game_over = False
        running = True
        
        print("\n" + "="*20)
        print(f"Starting Game {game_num}")
        print(f"Player 1 (X): {player_names[1]}")
        print(f"Player 2 (O): {player_names[-1]}")
        print("="*20)

        while not game_over and running:
            current_player_id = env.game.current_player
            player_name = get_player_name(env, player_names)
            
            is_human_turn = args.human_player and current_player_id == 1

            if is_human_turn:
                print(f"\nYour turn ({player_name})")
                # Handle Human Input
                action_taken = False
                while not action_taken:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            action_taken = True
                            break
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            x, y = event.pos
                            cell_size = env.game.window_size / 9
                            row, col = int(y // cell_size), int(x // cell_size)
                            action = (row * 9) + col
                            
                            legal_moves = env.game._get_legal_moves().flatten()
                            if legal_moves[action] == 1:
                                print(f"{player_name} plays at: {action} (Row {row}, Col {col})")
                                game_over = env.game.apply_action(action)
                                env.render()
                                action_taken = True
                            else:
                                print("Invalid move!")
                    if not running:
                        break
            else:
                # Handle Model Input
                print(f"\n{player_name}'s turn...")
                if args.viewer:
                    pygame.time.wait(500)
                    pygame.event.pump()

                obs = env._get_obs()
                model_to_use = model_p1 if current_player_id == 1 else model_p2
                action = model_to_use.pick_action(obs)
                
                row, col = env.game._int_to_entry(action)
                print(f"{player_name} plays at: {action} (Row {row}, Col {col})")

                game_over = env.game.apply_action(action)
                if args.viewer:
                    env.render()

            if game_over:
                winner = check_winner(env, player_names)
                if winner == player_names[1]:
                    scores[player_names[1]] += 1
                elif winner == player_names[-1]:
                    scores[player_names[-1]] += 1
                else:
                    scores["Ties"] += 1
                
                print("\n" + "-"*20)
                print(f"Game {game_num} Over!")
                print(f"Winner: {winner}")
                print("Current Scores:")
                print(f"  {player_names[1]}: {scores[player_names[1]]}")
                print(f"  {player_names[-1]}: {scores[player_names[-1]]}")
                print(f"  Ties: {scores['Ties']}")
                print("-"*20)

            # Check for quit event after every move
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if env.game.clock:
                env.game.clock.tick(30)
        
        if not running:
            break

        # Wait for 'n' key press for next game
        if game_num < args.num_games:
            print("\nPress 'N' to play the next game, or close the window to quit.")
            waiting_for_next = True
            while waiting_for_next:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_next = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                        waiting_for_next = False
                if not running:
                    break
    
    print("\nAll games finished!")
    env.close()

def get_player_name(env, player_names):
    return player_names.get(env.game.current_player, "Unknown")

def check_winner(env, player_names):
    # This check is based on the internal state of UTTTGame, which might be what is intended
    # It seems to check the macro board state for a winner
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