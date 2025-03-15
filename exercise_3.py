import random
import json
import numpy as np
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# =======================
# == TIC-TAC-TOE (Exercise 1)
# =======================

class TicTacToe:
    """
    Simple Tic-Tac-Toe for 2 "LLMs" that move randomly.
    'X' => LLM-1
    'O' => LLM-2
    """
    def __init__(self, size=3):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.current_player = 'X'  # 'X' starts (LLM-1)

    def display_board(self):
        for row in self.board:
            print(' | '.join(row))
            print('-' * (self.size * 2))
        print()

    def is_winner(self, player):
        # Check rows
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        # Check columns
        for col in range(self.size):
            if all(self.board[r][col] == player for r in range(self.size)):
                return True
        # Check diagonals
        if all(self.board[i][i] == player for i in range(self.size)):
            return True
        if all(self.board[i][self.size - 1 - i] == player for i in range(self.size)):
            return True
        return False

    def is_draw(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def get_empty_positions(self):
        return [(r, c) for r in range(self.size)
                for c in range(self.size)
                if self.board[r][c] == ' ']

    def make_random_move(self, player):
        empty_positions = self.get_empty_positions()
        if empty_positions:
            row, col = random.choice(empty_positions)
            self.board[row][col] = player
            return (row, col)
        return None

    def play_game(self):
        """
        Plays a full game with 2 "LLMs" making random moves.
        Returns 'X' if LLM-1 won, 'O' if LLM-2 won, or 'Draw'.
        """
        while True:
            # Make random move for current player
            move = self.make_random_move(self.current_player)
            # Check winner
            if self.is_winner(self.current_player):
                return self.current_player
            # Check draw
            if self.is_draw():
                return "Draw"
            # Switch player
            self.current_player = 'O' if self.current_player == 'X' else 'X'

# ==========================
# == WUMPUS WORLD (Exercise 2)
# ==========================

class WumpusWorld:
    """
    NxN Wumpus World with:
      - 1 gold (G), 1 wumpus (W), random pits (P), empty squares (E).
      - Agent starts at (0,0).
      - Agent can die if it steps into a pit.
    """
    def __init__(self, N=4):
        self.N = N
        self.world = np.full((N, N), 'E')
        
        # Place gold (not at (0,0))
        gold_x, gold_y = self._get_random_empty_cell(exclude=(0,0))
        if gold_x is not None:
            self.world[gold_y, gold_x] = 'G'
        
        # Place wumpus (not at (0,0) or gold)
        wumpus_x, wumpus_y = self._get_random_empty_cell(exclude=(0,0, gold_x, gold_y))
        if wumpus_x is not None:
            self.world[wumpus_y, wumpus_x] = 'W'
        
        # Random pits
        for r in range(N):
            for c in range(N):
                if self.world[r,c] == 'E':
                    if random.random() < 0.15:
                        self.world[r,c] = 'P'
        
        self.agent_pos = (0, 0)
        self.agent_alive = True
        self.visited = {(0,0)}
    
    def _get_random_empty_cell(self, exclude=()):
        empties = []
        for r in range(self.N):
            for c in range(self.N):
                if self.world[r,c] == 'E' and (c,r) not in exclude:
                    empties.append((c, r))
        if not empties:
            return None, None
        return random.choice(empties)
    
    def get_breeze_percept(self, x, y):
        """Breeze if any orth neighbor is a pit."""
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.N and 0 <= ny < self.N:
                if self.world[ny, nx] == 'P':
                    return True
        return False
    
    def is_pit(self, x, y):
        return self.world[y,x] == 'P'
    
    def is_gold(self, x, y):
        return self.world[y,x] == 'G'
    
    def move_agent(self, x, y):
        self.agent_pos = (x,y)
        self.visited.add((x,y))
        if self.is_pit(x,y):
            self.agent_alive = False
        elif self.is_gold(x,y):
            print(f"Agent found GOLD at ({x},{y})!")
    
    def reset_agent_if_dead(self, last_safe_pos):
        if not self.agent_alive:
            print("Agent died by stepping into a pit. Restarting from", last_safe_pos)
            self.agent_pos = last_safe_pos
            self.agent_alive = True
    
    def get_possible_moves(self):
        x, y = self.agent_pos
        moves = []
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.N and 0 <= ny < self.N:
                moves.append((nx, ny))
        return moves
    
    def reached_gold(self):
        return self.is_gold(*self.agent_pos)

# Bayesian Model for Pit inference

class PitBayesModel:
    """
    Same BN as in Exercise 2.  Pit_{x,y} -> Breeze_{neighbors}.
    Inference with VariableElimination to get P(Pit_{x,y}=1 | observed breezes).
    """
    def __init__(self, wumpus_world: WumpusWorld):
        self.world = wumpus_world
        self.N = wumpus_world.N
        
        self.model = BayesianNetwork()
        
        # Build edges
        for y in range(self.N):
            for x in range(self.N):
                pit_name = f"Pit_{x}_{y}"
                breeze_name = f"Breeze_{x}_{y}"
                # Add node for pit and breeze
                self.model.add_node(pit_name)
                self.model.add_node(breeze_name)
                
        for y in range(self.N):
            for x in range(self.N):
                # Breeze_x_y has parents: Pit of neighbors
                neighbors = self._get_neighbors(x,y)
                for (nx, ny) in neighbors:
                    self.model.add_edge(f"Pit_{nx}_{ny}", f"Breeze_{x}_{y}")
        
        # CPDs: Pit_x_y => prior 0.15
        pit_cpds = []
        for y in range(self.N):
            for x in range(self.N):
                pit_name = f"Pit_{x}_{y}"
                cpd = TabularCPD(
                    variable=pit_name,
                    variable_card=2,
                    values=[[0.85],[0.15]]
                )
                pit_cpds.append(cpd)
        
        # Breeze_x_y => deterministic: 1 if ANY neighbor pit=1
        breeze_cpds = []
        for y in range(self.N):
            for x in range(self.N):
                breeze_name = f"Breeze_{x}_{y}"
                neighbors = self._get_neighbors(x,y)
                parent_names = [f"Pit_{nx}_{ny}" for (nx, ny) in neighbors]
                num_parents = len(parent_names)
                combos = 2**num_parents
                vals_b0, vals_b1 = [], []
                for combo in range(combos):
                    bits = self._decode_binary(combo, num_parents)
                    if any(bits):
                        vals_b1.append(1.0)
                        vals_b0.append(0.0)
                    else:
                        vals_b1.append(0.0)
                        vals_b0.append(1.0)
                cpd = TabularCPD(
                    variable=breeze_name,
                    variable_card=2,
                    evidence=parent_names,
                    evidence_card=[2]*num_parents,
                    values=[vals_b0, vals_b1]
                )
                breeze_cpds.append(cpd)
        
        self.model.add_cpds(*pit_cpds, *breeze_cpds)
        self.inference = VariableElimination(self.model)
    
    def _get_neighbors(self, x,y):
        directions = [(0,1),(0,-1),(1,0),(-1,0)]
        res = []
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.N and 0 <= ny < self.N:
                res.append((nx, ny))
        return res
    
    def _decode_binary(self, num, length):
        return [int(x) for x in bin(num)[2:].zfill(length)]
    
    def update_and_infer_pit_probabilities(self, observations):
        """
        observations: dict {f"Breeze_x_y": 0 or 1}
        Return NxN matrix of P(Pit_{x,y}=1).
        """
        pit_prob = np.zeros((self.N,self.N))
        # Convert booleans
        evidence = {}
        for k,v in observations.items():
            evidence[k] = int(bool(v))
        
        for y in range(self.N):
            for x in range(self.N):
                pit_name = f"Pit_{x}_{y}"
                q = self.inference.query([pit_name], evidence=evidence, show_progress=False)
                prob_pit_1 = q.values[1]
                pit_prob[y,x] = prob_pit_1
        return pit_prob

# ===========================
# == Plotting for Wumpus
# ===========================

def plot_world_state(world: WumpusWorld, step_num, suffix=""):
    N = world.N
    ax = plt.gca()
    ax.set_title(f"Wumpus World Step={step_num} {suffix}")
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(-0.5, N-0.5)
    
    for i in range(N):
        ax.axhline(i-0.5, color='black', linewidth=1)
        ax.axvline(i-0.5, color='black', linewidth=1)
    
    for row in range(N):
        for col in range(N):
            val = world.world[row,col]
            if (col,row) == world.agent_pos and world.agent_alive:
                val = "A"
            ax.text(col, row, val, ha='center', va='center', fontsize=12)
    
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

def plot_pit_probability_heatmap(pit_prob, step_num, suffix=""):
    N = pit_prob.shape[0]
    ax = plt.gca()
    im = ax.imshow(pit_prob, origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    ax.set_title(f"Pit Probability Heatmap Step={step_num} {suffix}")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(Pit=1)")
    for y in range(N):
        for x in range(N):
            ax.text(x, y, f"{pit_prob[y,x]:.2f}", ha='center', va='center', color='black')

# ===========================
# == EXERCISE 3 MAIN
# ===========================

def exercise3_main(N=4, random_seed=42):
    """
    1) We repeatedly:
       - Play 1 TicTacToe game between LLM-1 (X) and LLM-2 (O).
       - If X wins => Wumpus agent makes "best move".
         If O wins => Wumpus agent makes "random move".
         If Draw => default to random.
       - Visualize Wumpus world + pit probability after that move.
    2) Stop if agent finds gold or dies too many times or step limit.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create Wumpus world + BN model
    ww = WumpusWorld(N)
    pit_model = PitBayesModel(ww)
    
    # Track breezes
    observations = {}
    
    step_count = 0
    max_steps = 5 * N * N
    
    while not ww.reached_gold() and step_count < max_steps:
        # 1) Play TicTacToe game => see who wins
        ttt = TicTacToe(size=3)
        winner = ttt.play_game()  # 'X', 'O', or 'Draw'
        print(f"TicTacToe result: {winner}")
        
        # 2) In Wumpus world, observe breeze at current cell
        x, y = ww.agent_pos
        b = ww.get_breeze_percept(x,y)
        observations[f"Breeze_{x}_{y}"] = 1 if b else 0
        
        # 3) Infer pit probabilities
        pit_prob = pit_model.update_and_infer_pit_probabilities(observations)
        
        # 4) Decide move type
        if winner == 'X':
            move_type = "BEST"  # LLM-1 => best move
        else:
            # If O or Draw => random move
            move_type = "RANDOM"
        
        # 5) Make one move
        possible = ww.get_possible_moves()
        if not possible:
            print("No possible moves left. Stopping.")
            break
        
        if move_type == "BEST":
            # choose neighbor with lowest pit prob
            best_move = None
            best_p = 1.0
            for (nx, ny) in possible:
                p = pit_prob[ny,nx]
                if p < best_p:
                    best_p = p
                    best_move = (nx, ny)
            last_safe = ww.agent_pos
            ww.move_agent(*best_move)
            ww.reset_agent_if_dead(last_safe)
        else:
            # RANDOM move
            last_safe = ww.agent_pos
            chosen = random.choice(possible)
            ww.move_agent(*chosen)
            ww.reset_agent_if_dead(last_safe)
        
        # 6) Plot after the move
        fig = plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plot_world_state(ww, step_count, suffix=f"({move_type} Move)")
        plt.subplot(1,2,2)
        plot_pit_probability_heatmap(pit_prob, step_count, suffix=f"({move_type} Move)")
        
        plt.tight_layout()
        plt.savefig(f"step_{step_count}_{move_type.lower()}.png")
        plt.show()
        plt.close(fig)
        
        step_count += 1
        
        # Check if agent reached gold or died
        if ww.reached_gold():
            print(f"Agent found gold after {step_count} steps.")
            break
        if not ww.agent_alive:
            print("Agent died in the Wumpus world. Stopping.")
            break
    
    if ww.reached_gold():
        print("SUCCESS: Gold reached!")
    elif step_count >= max_steps:
        print(f"Reached {max_steps} steps without finding gold. Stopping.")
    else:
        print("Stopped (agent died or no moves).")

# ================
# RUN EXERCISE 3
# ================
if __name__ == "__main__":
    user_input = input("Enter Wumpus board size (>=4): ")
    try:
        size = int(user_input)
        if size < 4:
            size = 4
    except:
        size = 4
    
    exercise3_main(N=size, random_seed=42)

