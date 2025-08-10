import numpy as np
import matplotlib.pyplot as plt
import heapq
import io
import sys
import time 

# Grid size - configurable for different test cases
N = 8  # Default size - can be changed for larger tests
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]  # 4-way movement directions: 
                                    # up, down, left, right

def raw_map(prob_block=0.0, start=None, goal=None, grid_size=None):
    """
    Generate a random N x N grid for pathfinding with obstacles.

    Each cell is independently set to either open (0) or blocked (1), based on 
    prob_block.
    Ensures start and goal positions are open, and each has at least one open 
    neighbor (to avoid dead-ends).

    Parameters:
        prob_block (float): Probability that each cell (except start/goal) is 
                            an obstacle (blocked).
        start (tuple): (row, col) index for start position (guaranteed open).
        goal (tuple): (row, col) index for goal position (guaranteed open).
        grid_size (int): Size of the grid (if None, uses global N).

    Returns:
        grid (np.ndarray): 2D array of shape (size, size), with 0=open and 1=obstacle.
    """
    if start is None or goal is None:
        raise ValueError("start and goal must be specified")
    
    # Use provided grid_size or default N
    size = grid_size if grid_size is not None else N
    
    # Validate start and goal positions
    if not (0 <= start[0] < size and 0 <= start[1] < size):
        raise ValueError(f"start position {start} is out of bounds for {size}x{size} grid")
    if not (0 <= goal[0] < size and 0 <= goal[1] < size):
        raise ValueError(f"goal position {goal} is out of bounds for {size}x{size} grid")
    
    # randomly assign 0 (open) or 1 (blocked) for each cell
    grid = np.random.choice([0, 1], size=(size, size), p=[1-prob_block, prob_block])

    # guarantee that start and goal cells are open, never blocked
    # decrease the chance to have no completed path
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    # guarantee at least one open neighbor for start
    start_neighbors = []
    for dx, dy in DIRS:
        nx, ny = start[0]+dx, start[1]+dy
        if 0 <= nx < size and 0 <= ny < size:
            start_neighbors.append((nx, ny))
    np.random.shuffle(start_neighbors)  # randomize neighbor order
    neighbor_opened = any(grid[nx, ny] == 0 for nx, ny in start_neighbors)
     # if no open neighbor, forcibly open the first neighbor
    if not neighbor_opened and start_neighbors:
        nx, ny = start_neighbors[0]
        grid[nx, ny] = 0

    # guarantee at least one open neighbor for goal
    goal_neighbors = []
    for dx, dy in DIRS:
        nx, ny = goal[0]+dx, goal[1]+dy
        if 0 <= nx < size and 0 <= ny < size:
            goal_neighbors.append((nx, ny))
    np.random.shuffle(goal_neighbors)
    neighbor_opened = any(grid[nx, ny] == 0 for nx, ny in goal_neighbors)
    if not neighbor_opened and goal_neighbors:
        nx, ny = goal_neighbors[0]
        grid[nx, ny] = 0

    return grid


def build_path(came_from, start, goal):
    """
    Reconstructs the path from goal to start using the came_from map.

    Parameters:
        came_from (dict): Mapping of node -> parent node.
        start (tuple): Start node (row, col).
        goal (tuple): Goal node (row, col).

    Returns:
        path (list): List of nodes from start to goal, or None if unreachable.
    """
    path = [goal]
    current = goal
    while current != start:
        if current not in came_from:
            return None  # no path found
        current = came_from[current]
        path.append(current)
    return path[::-1]


def dijkstra(grid, start, goal, return_costs=False, return_expanded=False, 
             return_visited=False):
    """
    Performs Dijkstra's algorithm (uniform cost search) on a 2D grid with obstacles.
    
    This is like the "regular GPS" that searches everywhere equally until it finds 
    the destination. Unlike A*, it has no sense of direction - it explores like 
    ripples spreading in a pond.

    Parameters:
        grid: np.ndarray
              The 2D grid representing the environment 
              (0=open cell, 1=blocked cell).
        start: tuple
               (row, col) coordinates for the starting node.
        goal: tuple
              (row, col) coordinates for the goal node.
        return_costs: bool, optional
                      If True, returns a dictionary mapping expanded nodes to 
                      their (g, h=0, f=g) costs.
        return_expanded: bool, optional
                         If True, returns the total number of unique nodes 
                         expanded during the search.
        return_visited: bool, optional
                        If True, returns a list representing the exact order 
                        in which nodes were expanded.

    Returns:
        tuple (variable-length)
            The first returned value is always:
            - path: list of (row, col) tuples representing the optimal path 
                     from start to goal, or None if no path is found.

            Optional returned values based on flags:
            - costs: dict mapping (row, col) -> (g, h=0, f=g) for each expanded node
                     (if return_costs=True).
            - expanded: int, total number of unique nodes expanded 
                        (if return_expanded=True).
            - visit_order: list of (row, col) tuples showing the order of node 
                           expansions (if return_visited=True).

    Algorithm Notes:
        - Uses uniform cost search with no heuristic guidance
        - Explores nodes in order of actual distance from start
        - Guarantees optimal path but explores more nodes than A*
        - Like ripples in a pond - searches all directions equally

    Complexity:
        - Time: O(V log V + E log V) = O(N^2 log N) where V=N^2, E≤4V
        - Space: O(N^2)
    """
    # get grid dimensions
    grid_size = grid.shape[0]
    
    # priority queue: stores (f_cost, g_cost, node, path) tuples to explore
    # for dijkstra: f_cost = g_cost (since h=0)
    open_set = []
    
    # set of already explored nodes to avoid re-exploration
    closed_set = set()
    
    # tracking for analysis
    visit_order = []
    costs = {start: (0, 0, 0)}  # (g, h=0, f=g)
    expanded = 0

    # initialize with start node: (f=0, g=0, node=start, path=[start])
    heapq.heappush(open_set, (0, 0, start, [start]))

    while open_set:
        # get the node with lowest actual cost
        # tuple format: (f_cost, g_cost, node, path)
        current_f, current_g, current_node, current_path = heapq.heappop(open_set)
        
        # check if we reached the goal
        if current_node == goal:
            result = [current_path]
            if return_costs: 
                result.append(costs)
            if return_expanded: 
                result.append(expanded)
            if return_visited: 
                result.append(visit_order)
            return tuple(result)
        
        # skip if already explored
        if current_node in closed_set:
            continue
        
        # mark as explored
        closed_set.add(current_node)
        visit_order.append(current_node)
        expanded += 1
        
        # explore all neighbors
        for direction in DIRS:
            neighbor_row = current_node[0] + direction[0]
            neighbor_col = current_node[1] + direction[1]
            
            # skip if out of bounds or blocked
            if not (0 <= neighbor_row < grid_size and 
                    0 <= neighbor_col < grid_size and 
                    grid[neighbor_row, neighbor_col] == 0):
                continue
            
            next_node = (neighbor_row, neighbor_col)
            
            # skip if already explored
            if next_node in closed_set:
                continue
            
            # calculate cost to reach neighbor
            tentative_g = current_g + 1
            h = 0  # dijkstra: no heuristic
            f = tentative_g + h  # f = g when h = 0
            
            # only add/update if new or better cost
            if next_node not in costs or tentative_g < costs[next_node][0]:
                costs[next_node] = (tentative_g, h, f)
                # CONSISTENT tuple format: (f_cost, g_cost, node, path)
                heapq.heappush(open_set, (f, tentative_g, next_node, current_path + [next_node]))
    
    # no path found: return None and optionally other requested data
    result = [None]
    if return_costs:
        result.append(costs)
    if return_expanded:
        result.append(expanded)
    if return_visited:
        result.append(visit_order)
    return tuple(result)


def plot_map_with_costs(grid, costs, path=None, title="", start=None, goal=None,
                       filename=None, visited=None):
    """
    Visualizes the 2D grid:
    - Obstacles in black, open cells white
    - Visited/expanded nodes as yellow squares (no order numbers)
    - Solution path in red
    - Start (green star), Goal (blue star)
    - g/h/f values overlaid on expanded nodes

    Parameters:
        grid (np.ndarray): 2D grid (0=open, 1=blocked)
        costs (dict): (row, col) -> (g, h=0, f=g)
        path (list): Path from start to goal (optional)
        title (str): Plot title
        start (tuple): Start node (row, col)
        goal (tuple): Goal node (row, col)
        filename (str): If given, save figure to this file
        visited (list/set): Expanded nodes to highlight (optional)

    Returns:
        None. Shows and optionally saves the plot.
    """
    if start is None or goal is None:
        raise ValueError("start and goal must be specified")
    
    grid_size = grid.shape[0]
    plt.figure(figsize=(max(8, grid_size//2), max(8, grid_size//2)))
    plt.title(f"{title}\n(g: steps from start, h: always 0 for dijkstra, f: g+h)")
    plt.imshow(grid, cmap='gray_r')
    plt.xticks(np.arange(grid.shape[1]))
    plt.yticks(np.arange(grid.shape[0]))
    plt.grid(True, color='lightgray')

    # plot expanded/visited nodes as yellow squares
    if visited:
        vx, vy = zip(*visited)
        plt.scatter(vy, vx, c='orange', s=90, marker='s', alpha=0.5, 
                    label='Visited')

    # overlay g/h/f with labels for expanded nodes
    for (x, y), (g, h, f) in costs.items():
        txt = f"g={g}\nh={h}\nf={f}"
        plt.text(
            y, x, txt, ha='center', va='center', fontsize=8, color='black',
            bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', 
                      boxstyle='round,pad=0.18'))

    # draw the solution path
    if path:
        px, py = zip(*path)
        plt.plot(py, px, color='red', linewidth=3, marker='o', markersize=8, 
                 label='Path')

    # draw start and goal
    plt.scatter([start[1]], [start[0]], color='green', s=200, marker='*', 
                edgecolors='black', linewidths=1, zorder=2, label='Start')
    plt.scatter([goal[1]], [goal[0]], color='blue', s=200, marker='*', 
                edgecolors='black', linewidths=1, zorder=2, label='Goal')

    # legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(), by_label.keys(),
        loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()


def print_path_costs_and_heuristics(path, goal, costs, algo='dijkstra'):
    """
    Print the actual path cost (g), heuristic (h), and total cost (f) values
    for each node along the solution path for Dijkstra's algorithm.

    For each node in the path:
        - g: The actual cost from the start node to the current node along the path.
        - h: Always 0 for Dijkstra (no heuristic used)
        - f: Same as g (since h=0)

    Parameters:
        path: list of tuple
              Ordered list of (row, col) positions along the solution path, 
              from start to goal.
        goal: tuple
              The (row, col) coordinates of the goal node.
        costs: dict
               A dictionary mapping (row, col) positions to their cost values.
               For Dijkstra, costs[cell] should be a tuple (g, h=0, f=g).
        algo: str, optional
              The algorithm type: should be 'dijkstra'.

    Returns:
        None
    """
    print("Actual cost (g) and heuristic (h) values along the path:")
    for cell in path:
        g = costs[cell][0] if cell in costs else "NA"
        h = 0  # dijkstra never uses heuristics
        f = g  # f equals g when h=0
        print(f"  {cell}: g={g}, h={h}, f={f}")


def print_grid_stats(grid):
    """
    Prints statistics about the generated grid:
      - Total number of cells
      - Number and percentage of obstacles
      - Number and percentage of free cells

    Parameters:
        grid: np.ndarray
              2D grid of the map (0=open, 1=obstacle).

    Returns:
        None    
    """
    total = grid.size
    obstacles = np.sum(grid == 1)
    free = np.sum(grid == 0)
    percent_obstacles = 100.0 * obstacles / total
    percent_free = 100.0 * free / total
    print(f"Total cells: {total}")
    print(f"Obstacles: {obstacles} ({percent_obstacles:.1f}%)")
    print(f"Free cells: {free} ({percent_free:.1f}%)\n")


def create_map(prob_block=0.0, max_attempts=1000, start=None, goal=None, grid_size=None):
    """
    Generate a random map with a valid path from start to goal.

    - Tries up to max_attempts times to produce a map where a valid path exists.
    - Uses Dijkstra to check for solvability.
    - If no path is found after all attempts, raises an error.

    Parameters:
        prob_block: float
                    Probability that a given cell is an obstacle.
        start: tuple
               (row, col) position for the start node.
        goal: tuple
              (row, col) position for the goal node.
        max_attempts: int, optional
                      Maximum number of attempts to generate a solvable map.
        grid_size: int, optional
                   Size of the grid (if None, uses global N).

    Returns:
        grid: np.ndarray
              2D grid with 0=open, 1=obstacle.
        path: list of tuple
              The shortest path from start to goal, if found.
        costs: dict
               Expanded cells and their (g, h=0, f=g) values.

    Raises:
        RuntimeError
            If no solvable map is found after max_attempts.
    """
    if start is None or goal is None:
        raise ValueError("start and goal must be specified")
    for attempt in range(max_attempts):
        grid = raw_map(prob_block=prob_block, start=start, goal=goal, grid_size=grid_size)
        path, costs = dijkstra(grid, start, goal, return_costs=True, 
                               return_visited=False)
        if path:
            return grid, path, costs
    raise RuntimeError(
        f"Could not create a solvable map in {max_attempts} attempts. "
        "Try lowering obstacle rate or increasing attempts.")


def main():
    """
    For each test case (obstacle probability), generates a random map,
    runs Dijkstra algorithm, and prints/saves:
      - The found path
      - Number of expanded nodes and runtime
      - The full list of expanded (visited) nodes in order
      - Step-by-step g/h/f values along the path
    Visualizations show the path and expansion order.
    All output is saved to a .txt file for each case.
    """
    # np.random.seed(42) # For reproducibility: Remove the comment sign (#)
    # Note: Random seed not set to demonstrate algorithm behavior across different maps

    start = (0, 0)
    goal = (7, 7)
    probs = [0.0, 0.2, 0.5]
    titles = ["No Obstacles (0%)", "20% Obstacles", "50% Obstacles"]

    for i, (prob, title) in enumerate(zip(probs, titles)):
        output_lines = []
        print(f"\n=== Case: {title} ===")
        output_lines.append(f"\n=== Case: {title} ===")
        print(f"Start: {start}   Goal: {goal}")
        output_lines.append(f"Start: {start}   Goal: {goal}")

        try:
            # generate a random solvable map for this case
            grid, _, _ = create_map(prob_block=prob, start=start, goal=goal)

            # run dijkstra and time it
            t0 = time.time()
            path_dijkstra, costs_dijkstra, dijkstra_expanded, dijkstra_visit_order = dijkstra(
                grid, start, goal, return_costs=True, return_expanded=True, 
                return_visited=True)
            t1 = time.time()
            dijkstra_time = t1 - t0

            # print map statistics
            tempbuf = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = tempbuf
                print_grid_stats(grid)
            finally:
                sys.stdout = old_stdout
            gridstats = tempbuf.getvalue().rstrip()
            print(gridstats)
            output_lines.append(gridstats)
            print()
            output_lines.append("")

            # dijkstra path
            print("Dijkstra Path found:")
            output_lines.append("Dijkstra Path found:")
            print(path_dijkstra)
            output_lines.append(str(path_dijkstra))
            print()
            output_lines.append("")

            print(f"Dijkstra expanded {dijkstra_expanded} nodes, "
                  f"time: {dijkstra_time:.6f} sec")
            output_lines.append(
                f"Dijkstra expanded {dijkstra_expanded} nodes, "
                f"time: {dijkstra_time:.6f} sec")
            
            if path_dijkstra:
                print(f"Total nodes in found path: {len(path_dijkstra)}")
                output_lines.append(f"Total nodes in found path: {len(path_dijkstra)}")

            print()
            output_lines.append("")

            # print the full expansion order for dijkstra
            print("Dijkstra visited node order (all expanded nodes):")
            output_lines.append("Dijkstra visited node order (all expanded nodes):")
            for idx, cell in enumerate(dijkstra_visit_order):
                print(f"  {idx+1:3}: {cell}")
                output_lines.append(f"  {idx+1:3}: {cell}")
            print()
            output_lines.append("")

            if path_dijkstra:
                tempbuf = io.StringIO()
                old_stdout = sys.stdout
                try:
                    sys.stdout = tempbuf
                    print_path_costs_and_heuristics(path_dijkstra, goal, 
                                                    costs_dijkstra,
                                                    algo='dijkstra')
                finally:
                    sys.stdout = old_stdout
                pathcosts = tempbuf.getvalue().rstrip()
                print(pathcosts)
                output_lines.append(pathcosts)
                print()
                output_lines.append("")

            img_filename_dijkstra = f"dijkstra_case_{i+1}.png"
            plot_map_with_costs(grid, costs_dijkstra, path_dijkstra, 
                                title=title + " (Dijkstra)", start=start, goal=goal, 
                                filename=img_filename_dijkstra, 
                                visited=dijkstra_visit_order)

        except RuntimeError as e:
            errmsg = str(e)
            print(errmsg)
            output_lines.append(errmsg)
            print()
            output_lines.append("")

        # save all outputs for this test case to a text file
        text_filename = f"dijkstra_case_{i+1}.txt"
        with open(text_filename, "w", encoding="utf-8") as txtfile:
            for line in output_lines:
                txtfile.write(line + "\n")


if __name__ == "__main__":
    main()


'''

SPECIAL NOTES

Dijkstra's Algorithm "Regular GPS" Implementation

Key Characteristics:
- Uses uniform cost search (f = g, h = 0)
- Explores in "ripples" pattern from start
- Guarantees optimal path but explores more nodes than A*
- No heuristic guidance toward goal

Tuple Structure in Priority Queue:
- Format: (f_cost, g_cost, node, path)
- f_cost = g_cost for Dijkstra (since h = 0)
- Consistent ordering prevents runtime errors

Complexity Analysis:
- Time: O(V log V + E log V) = O(N² log N)
  where V = N² cells, E ≤ 4V edges
- Space: O(N²) for grid, queue, and path storage

Comparison with A*:
- A* uses f = g + h with Manhattan distance heuristic
- A* typically explores fewer nodes due to directional guidance
- Both guarantee optimal paths in grid environments

'''