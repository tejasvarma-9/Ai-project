import random
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# -------------------------------
# Part (a) Tic-Tac-Toe Game Class
# -------------------------------
class TicTacToe:
    def __init__(self, size=3):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.current_player = 'X'  # Player X starts

    def display_board(self):
        for row in self.board:
            print(' | '.join(row))
            print('-' * (self.size * 2))

    def is_winner(self, player):
        for row in self.board:
            if all(cell == player for cell in row):
                return True

        for col in range(self.size):
            if all(self.board[row][col] == player for row in range(self.size)):
                return True

        if all(self.board[i][i] == player for i in range(self.size)) or \
           all(self.board[i][self.size - i - 1] == player for i in range(self.size)):
            return True

        return False

    def is_draw(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def get_empty_positions(self):
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == ' ']

    def make_move(self, player):
        empty_positions = self.get_empty_positions()
        if empty_positions:
            row, col = random.choice(empty_positions)
            self.board[row][col] = player
            return (row, col)
        return None

    def play_game(self):
        while True:
            self.display_board()
            move = self.make_move(self.current_player)
            print(f"Player {self.current_player} moves to {move}")

            if self.is_winner(self.current_player):
                self.display_board()
                print(f"Player {self.current_player} wins!")
                return self.current_player

            if self.is_draw():
                self.display_board()
                print("It's a draw!")
                return "Draw"

            self.current_player = 'O' if self.current_player == 'X' else 'X'


# -----------------------------------
# Part (b) Simulate 500 Games & Plot
# -----------------------------------
def simulate_games(num_trials=500):
    results = {"X": 0, "O": 0, "Draw": 0}

    for _ in range(num_trials):
        game = TicTacToe(size=3)
        winner = game.play_game()
        results[winner] += 1

    # Save results to a file
    with open("Exercise1.json", "w") as f:
        json.dump(results, f)

    return results


def plot_binomial_distribution(results, num_trials=500):
    total_games = results["X"] + results["O"]
    p_win_X = results["X"] / total_games if total_games > 0 else 0.5  # Avoid division by zero

    # Generate binomial distribution
    x_values = np.arange(0, num_trials + 1)
    y_values = binom.pmf(x_values, num_trials, p_win_X)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, color='b', alpha=0.6, label=f'X Wins Probability: {p_win_X:.2f}')
    plt.xlabel("Number of X Wins")
    plt.ylabel("Probability")
    plt.title("Binomial Distribution of Tic-Tac-Toe Wins (500 Games)")
    plt.legend()
    plt.savefig("Exercise1.png")
    plt.show()


# --------------------------------------------
# Part (c) Implement LLM Move (LLM Integration)
# --------------------------------------------
# Uncomment below if using OpenAI API
# import openai  

def llm_move(board, last_move):
    """
    Function to generate the next move using an LLM.
    Replace this with an actual API call.
    """
    # Simulating LLM move by selecting a random empty position
    empty_positions = [(r, c) for r in range(3) for c in range(3) if board[r][c] == ' ']
    if empty_positions:
        return random.choice(empty_positions)
    return None

    # Example OpenAI API call (requires API key)
    """
    prompt = f"Given this tic-tac-toe board:\n{board}\nLast move was: {last_move}\nWhat's the best move?"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a tic-tac-toe strategist."},
                  {"role": "user", "content": prompt}]
    )
    move = response['choices'][0]['message']['content']
    return tuple(map(int, move.strip("()").split(',')))
    """


# ------------------------------------
# Part (d) Human vs. LLM Tic-Tac-Toe
# ------------------------------------
def human_vs_llm():
    game = TicTacToe(size=3)
    while True:
        game.display_board()

        if game.current_player == 'X':  # Human Player
            row, col = map(int, input("Enter your move (row col): ").split())
            game.board[row][col] = 'X'
        else:  # LLM Player
            move = llm_move(game.board, None)
            if move:
                row, col = move
                game.board[row][col] = 'O'

        if game.is_winner(game.current_player):
            game.display_board()
            print(f"Player {game.current_player} wins!")
            break

        if game.is_draw():
            game.display_board()
            print("It's a draw!")
            break

        game.current_player = 'O' if game.current_player == 'X' else 'X'


# ---------------------------------
# Run the Simulation and Plot Data
# ---------------------------------
if __name__ == "__main__":
    # Run 500 games
    results = simulate_games(num_trials=500)

    # Generate binomial distribution
    plot_binomial_distribution(results)

    # Uncomment to play human vs. LLM
    # human_vs_llm()
