import numpy as np
import random
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class WumpusWorld:
    """
    A flexible NxN Wumpus World.
    We track:
      - The agent's position
      - Which cells contain pits (P), gold (G), wumpus (W), or are empty (E)
      - Whether the agent is alive
      - Which squares have been visited
    """
    def __init__(self, N=4):
        self.N = N
        
        # Initialize all cells as empty
        self.world = np.full((N, N), 'E')  # E means Empty
        
        # Place 1 gold somewhere not at (0,0)
        gold_x, gold_y = self._get_random_empty_cell(exclude=(0,0))
        if gold_x is not None:
            self.world[gold_y, gold_x] = 'G'
        
        # Place 1 wumpus somewhere not at (0,0) or gold
        wumpus_x, wumpus_y = self._get_random_empty_cell(exclude=(0,0, gold_x, gold_y))
        if wumpus_x is not None:
            self.world[wumpus_y, wumpus_x] = 'W'
        
        # Randomly place pits (15% chance) in remaining empty cells
        for r in range(N):
            for c in range(N):
                if self.world[r,c] == 'E':
                    if random.random() < 0.15:
                        self.world[r,c] = 'P'
        
        # Agent starts at (0,0) (bottom-left in the problem statement).
        # We'll interpret row=0 as the bottom row for plotting.
        self.agent_pos = (0, 0)  # (x, y)
        self.agent_alive = True
        self.visited = {(0, 0)}

    def _get_random_empty_cell(self, exclude=()):
        """Return a random (x,y) that is currently 'E' and not in 'exclude' (x,y) coords."""
        empties = []
        for r in range(self.N):
            for c in range(self.N):
                if self.world[r,c] == 'E' and (c, r) not in exclude:
                    empties.append((c, r))
        if not empties:
            return None, None
        return random.choice(empties)
    
    def get_breeze_percept(self, x, y):
        """
        Returns True if the agent perceives a 'Breeze' at (x,y).
        A breeze occurs if ANY of the 4 orth. neighbors is a pit (P).
        """
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.N and 0 <= ny < self.N:
                if self.world[ny, nx] == 'P':
                    return True
        return False
    
    def is_pit(self, x, y):
        return self.world[y, x] == 'P'
    
    def is_gold(self, x, y):
        return self.world[y, x] == 'G'
    
    def move_agent(self, x, y):
        """
        Move the agent to (x,y). Check if we fall into a pit or find gold.
        """
        self.agent_pos = (x, y)
        self.visited.add((x, y))
        
        if self.is_pit(x, y):
            self.agent_alive = False
        elif self.is_gold(x, y):
            print(f"Agent found GOLD at ({x},{y})!")
    
    def reset_agent_if_dead(self, last_safe_position):
        """
        If the agent died by stepping into a pit, we 'restart' it at 'last_safe_position'.
        """
        if not self.agent_alive:
            print(f"Agent fell into a pit! Restarting from {last_safe_position}...")
            self.agent_pos = last_safe_position
            self.agent_alive = True
    
    def get_possible_moves(self):
        """
        Return all valid neighbor coordinates (x,y) from the agent's current position.
        """
        x, y = self.agent_pos
        moves = []
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.N and 0 <= ny < self.N:
                moves.append((nx, ny))
        return moves
    
    def reached_gold(self):
        return self.is_gold(*self.agent_pos)

# ---------------------------------------------------------------------
# Bayesian Network for Pit Probability using pgmpy

class PitBayesModel:
    """
    Builds a BN for each cell's Pit variable => Breeze variables in neighboring cells.
    We do:
      Pit_x_y -> Breeze_(neighbors)
    Then do inference to get P(Pit_x_y=1 | observed Breezes).
    """
    def __init__(self, wumpus_world: WumpusWorld):
        self.world = wumpus_world
        self.N = wumpus_world.N
        
        self.model = BayesianNetwork()
        
        # We'll name each pit var as "Pit_x_y" and each breeze var as "Breeze_x_y"
        pit_vars = []
        breeze_vars = []
        
        # 1) Add edges: Pit_(neighbors) -> Breeze_(x,y)
        for y in range(self.N):
            for x in range(self.N):
                pit_name = f"Pit_{x}_{y}"
                breeze_name = f"Breeze_{x}_{y}"
                pit_vars.append(pit_name)
                breeze_vars.append(breeze_name)
                
                neighbors = self._get_neighbors(x, y)
                for (nx, ny) in neighbors:
                    self.model.add_edge(f"Pit_{nx}_{ny}", breeze_name)
        
        # 2) Add CPDs:
        #    (a) Pit_x_y ~ Bernoulli(0.15)
        pit_cpds = []
        for y in range(self.N):
            for x in range(self.N):
                pit_name = f"Pit_{x}_{y}"
                cpd = TabularCPD(
                    variable=pit_name,
                    variable_card=2,  # 0 or 1
                    values=[[0.85],[0.15]]  # P(notPit)=0.85, P(Pit)=0.15
                )
                pit_cpds.append(cpd)
        
        #    (b) Breeze_x_y depends on whether ANY neighbor's pit=1 (deterministic).
        breeze_cpds = []
        for y in range(self.N):
            for x in range(self.N):
                breeze_name = f"Breeze_{x}_{y}"
                neighbors = self._get_neighbors(x, y)
                parent_names = [f"Pit_{nx}_{ny}" for (nx, ny) in neighbors]
                
                # For up to 4 parents => 2^4=16 combos
                # We'll do a table that says Breeze=1 if ANY parent=1
                num_parents = len(parent_names)
                combos = 2**num_parents
                vals_b0 = []
                vals_b1 = []
                for combo_idx in range(combos):
                    bits = self._decode_binary(combo_idx, num_parents)
                    if any(bits):
                        # If any pit=1 => breeze=1 with prob=1
                        vals_b1.append(1.0)
                        vals_b0.append(0.0)
                    else:
                        # No parent pit => breeze=0 with prob=1
                        vals_b1.append(0.0)
                        vals_b0.append(1.0)
                
                cpd = TabularCPD(
                    variable=breeze_name,
                    variable_card=2,  # 0 or 1
                    evidence=parent_names,
                    evidence_card=[2]*num_parents,
                    values=[vals_b0, vals_b1]
                )
                breeze_cpds.append(cpd)
        
        self.model.add_nodes_from(pit_vars + breeze_vars)
        self.model.add_cpds(*pit_cpds, *breeze_cpds)
        
        # Inference engine
        self.inference = VariableElimination(self.model)
    
    def _get_neighbors(self, x, y):
        """Return orth. neighbors within the grid."""
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        neighbors = []
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.N and 0 <= ny < self.N:
                neighbors.append((nx, ny))
        return neighbors
    
    def _decode_binary(self, num, length):
        return [int(x) for x in bin(num)[2:].zfill(length)]
    
    def update_and_infer_pit_probabilities(self, observations):
        """
        Given a dict of {f"Breeze_x_y":0/1}, return a matrix pit_prob[y,x]
        of P(Pit_{x}_{y}=1 | evidence).
        """
        # Convert bool->0/1 if needed
        evidence = {}
        for k,v in observations.items():
            evidence[k] = int(bool(v))
        
        pit_prob = np.zeros((self.N, self.N))
        for y in range(self.N):
            for x in range(self.N):
                pit_name = f"Pit_{x}_{y}"
                q = self.inference.query([pit_name], evidence=evidence, show_progress=False)
                # Single-variable query => DiscreteFactor
                # If states are [0,1], q.values[1] = P(Pit=1)
                p_pit_1 = q.values[1]
                pit_prob[y,x] = p_pit_1
        
        return pit_prob

# ---------------------------------------------------------------------
# Plotting helper functions

def plot_world_state(world: WumpusWorld, step_num, title_suffix=""):
    """
    Plot the NxN grid with agent, pits, wumpus, gold, etc.
    (0,0) is bottom-left visually.
    """
    N = world.N
    ax = plt.gca()
    ax.set_title(f"Wumpus World (Bottom-Left=(0,0)), Step={step_num}\n{title_suffix}")
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(-0.5, N-0.5)
    
    # Grid lines
    for i in range(N):
        ax.axhline(i-0.5, color='black', linewidth=1)
        ax.axvline(i-0.5, color='black', linewidth=1)
    
    # Label each cell
    for row in range(N):
        for col in range(N):
            val = world.world[row,col]
            # If agent is here (and alive), label 'A'
            if (col, row) == world.agent_pos and world.agent_alive:
                val = "A"
            ax.text(col, row, str(val),
                    ha='center', va='center', color='black', fontsize=12)
    
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # so row=0 is at bottom

def plot_pit_probability_heatmap(pit_prob, step_num, title_suffix=""):
    """
    Show the NxN pit_prob matrix in a heatmap with values overlaid.
    """
    N = pit_prob.shape[0]
    ax = plt.gca()
    im = ax.imshow(pit_prob, origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    ax.set_title(f"Pit Probability Heatmap\nStep={step_num} {title_suffix}")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(Pit=1)", rotation=90)
    
    # Put numeric probability in each cell
    for row in range(N):
        for col in range(N):
            ax.text(col, row, f"{pit_prob[row,col]:.2f}",
                    ha='center', va='center', color='black')

# ---------------------------------------------------------------------
# MAIN: (i) random move, (ii) best move, visualize each step

def main(N=4, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create the Wumpus world
    ww = WumpusWorld(N)
    # Create Bayesian model for pit inference
    pit_model = PitBayesModel(ww)
    
    # Keep track of observed Breezes: {f"Breeze_x_y":0/1}
    observations = {}
    
    step_num = 0
    max_steps = 5 * N * N  # limit to prevent infinite loops
    
    while not ww.reached_gold() and step_num < max_steps:
        x, y = ww.agent_pos
        
        # 1) Observe breeze at current cell
        breeze = ww.get_breeze_percept(x, y)
        observations[f"Breeze_{x}_{y}"] = 1 if breeze else 0
        
        # 2) Update pit probabilities
        pit_prob = pit_model.update_and_infer_pit_probabilities(observations)
        
        # -----------------------------
        # (i) RANDOM MOVE
        # -----------------------------
        possible_moves = ww.get_possible_moves()
        if not possible_moves:
            print("No moves available; stopping.")
            break
        
        random_move = random.choice(possible_moves)
        last_safe_position = (x, y)
        ww.move_agent(*random_move)
        
        # If agent died from random move => reset
        ww.reset_agent_if_dead(last_safe_position)
        
        # Visualize after the random move
        # (Because the instructions say: "Visualize pit prob after the agent is moved")
        fig = plt.figure(figsize=(10,4))
        
        # Left subplot: world state
        plt.subplot(1,2,1)
        plot_world_state(ww, step_num, title_suffix="(After Random Move)")
        
        # Right subplot: pit probability
        plt.subplot(1,2,2)
        plot_pit_probability_heatmap(pit_prob, step_num, title_suffix="(After Random Move)")
        
        plt.tight_layout()
        plt.savefig(f"step_{step_num}_random.png")
        plt.show()
        plt.close(fig)
        
        step_num += 1
        
        # Check if agent reached gold or is dead
        if ww.reached_gold():
            print(f"Reached gold after random move at step={step_num}")
            break
        if not ww.agent_alive:
            print("Agent died on random move. Stopping.")
            break
        
        # 3) If still alive, do the (ii) BEST MOVE
        #    => choose neighbor with the lowest pit probability
        pit_prob = pit_model.update_and_infer_pit_probabilities(observations)
        possible_moves = ww.get_possible_moves()
        if not possible_moves:
            print("No moves available for best move; stopping.")
            break
        
        best_move = None
        best_p = 1.0
        for (nx, ny) in possible_moves:
            p = pit_prob[ny, nx]
            if p < best_p:
                best_p = p
                best_move = (nx, ny)
        
        last_safe_position = ww.agent_pos
        ww.move_agent(*best_move)
        ww.reset_agent_if_dead(last_safe_position)
        
        # Visualize after the best move
        fig = plt.figure(figsize=(10,4))
        
        # Left subplot: world state
        plt.subplot(1,2,1)
        plot_world_state(ww, step_num, title_suffix="(After Best Move)")
        
        # Right subplot: pit probability
        plt.subplot(1,2,2)
        plot_pit_probability_heatmap(pit_prob, step_num, title_suffix="(After Best Move)")
        
        plt.tight_layout()
        plt.savefig(f"step_{step_num}_best.png")
        plt.show()
        plt.close(fig)
        
        step_num += 1
        
        # Check if agent reached gold or is dead
        if ww.reached_gold():
            print(f"Reached gold after best move at step={step_num}")
            break
        if not ww.agent_alive:
            print("Agent died on best move. Stopping.")
            break
    
    # End of loop
    if ww.reached_gold():
        print(f"Agent got the GOLD in {step_num} steps!")
    elif step_num >= max_steps:
        print(f"Hit {max_steps} steps without reaching gold; stopping.")
    else:
        print("Stopped without reaching gold (agent died or no moves).")

if __name__ == "__main__":
    # Prompt the user for an integer N >= 4
    user_input = input("Enter board size (>=4): ")
    try:
        board_size = int(user_input)
        if board_size < 4:
            print("Using N=4 by default (must be >=4).")
            board_size = 4
    except ValueError:
        print("Invalid input, using N=4.")
        board_size = 4
    
    main(N=board_size, random_seed=42)

