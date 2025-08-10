import heapq
import numpy as np

def dijkstra(grid, start, goal, return_costs=False, return_expanded=False, return_visited=False):
    """
    Performs Dijkstra's algorithm (uniform cost search) on a 2D grid.
    
    This is the "regular GPS" that searches everywhere equally like ripples in a pond.
    It has no sense of direction - treats all paths as equally promising until it 
    systematically explores everywhere.
    
    Parameters:
        grid: np.ndarray
              2D grid where 0=open cell, 1=blocked cell
        start: tuple  
               (row, col) coordinates for starting position
        goal: tuple
              (row, col) coordinates for goal position  
        return_costs: bool, optional
                     If True, returns cost dictionary (g, h=0, f=g) for each expanded node
        return_expanded: bool, optional
                        If True, returns total number of nodes expanded
        return_visited: bool, optional
                       If True, returns list showing order of node expansion
    
    Returns:
        tuple: (path, [costs], [expanded_count], [visit_order])
               - path: list of (row, col) tuples from start to goal, or None
               - costs: dict mapping (row, col) -> (g, h=0, f=g) if requested
               - expanded_count: int, number of nodes expanded if requested  
               - visit_order: list of expansion order if requested
    
    Algorithm Notes:
        - Uses uniform cost search with no heuristic (h=0 always)
        - Explores nodes in order of actual distance from start
        - Guaranteed optimal path but explores more nodes than A*
        - Search pattern: expands in concentric circles like ripples in pond
    
    Complexity:
        - Time: O(V log V) where V = grid_size²
        - Space: O(V) for priority queue and tracking structures
    """
    if grid is None or start is None or goal is None:
        raise ValueError("grid, start, and goal must be specified")
    
    rows, cols = grid.shape
    
    # validate start and goal positions
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError(f"start position {start} is out of bounds")
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError(f"goal position {goal} is out of bounds")
    if grid[start[0], start[1]] == 1:
        raise ValueError(f"start position {start} is blocked")
    if grid[goal[0], goal[1]] == 1:
        raise ValueError(f"goal position {goal} is blocked")
    
    # movement directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # core dijkstra data structures
    open_set = []  # priority queue: (f_cost, g_cost, node)
    closed_set = set()  # already expanded nodes
    came_from = {}  # parent pointers for path reconstruction
    g_score = {start: 0}  # best known cost to reach each node
    
    # optional tracking for analysis
    costs = {start: (0, 0, 0)} if return_costs else {}  # (g, h=0, f=g)
    visit_order = [] if return_visited else []
    expanded_count = 0
    
    # initialize: add start to priority queue
    # tuple format: (f_cost, g_cost, node) for consistent ordering
    heapq.heappush(open_set, (0, 0, start))
    
    while open_set:
        # get node with lowest actual cost (no heuristic guidance)
        current_f, current_g, current_node = heapq.heappop(open_set)
        
        # skip if already processed (duplicate in queue)
        if current_node in closed_set:
            continue
            
        # mark as expanded
        closed_set.add(current_node)
        if return_visited:
            visit_order.append(current_node)
        if return_expanded:
            expanded_count += 1
            
        # check if goal reached
        if current_node == goal:
            # reconstruct path from goal to start using parent pointers
            path = _build_path(came_from, start, goal)
            
            # build return tuple based on requested information
            result = [path]
            if return_costs:
                result.append(costs)
            if return_expanded:
                result.append(expanded_count)
            if return_visited:
                result.append(visit_order)
            return tuple(result)
        
        # explore all neighbors in 4 directions
        for direction in directions:
            neighbor_row = current_node[0] + direction[0]
            neighbor_col = current_node[1] + direction[1]
            neighbor = (neighbor_row, neighbor_col)
            
            # skip if out of bounds
            if not (0 <= neighbor_row < rows and 0 <= neighbor_col < cols):
                continue
                
            # skip if blocked or already processed
            if grid[neighbor_row, neighbor_col] == 1 or neighbor in closed_set:
                continue
            
            # calculate cost to reach neighbor (each move costs 1)
            tentative_g = current_g + 1
            
            # skip if we already found a better path to this neighbor
            if neighbor in g_score and tentative_g >= g_score[neighbor]:
                continue
            
            # record this as the best path to neighbor
            came_from[neighbor] = current_node
            g_score[neighbor] = tentative_g
            
            # dijkstra: h=0 always, so f=g
            neighbor_h = 0
            neighbor_f = tentative_g + neighbor_h
            
            if return_costs:
                costs[neighbor] = (tentative_g, neighbor_h, neighbor_f)
            
            # add neighbor to priority queue for future exploration
            heapq.heappush(open_set, (neighbor_f, tentative_g, neighbor))
    
    # no path found - goal unreachable
    result = [None]
    if return_costs:
        result.append(costs)
    if return_expanded:
        result.append(expanded_count)
    if return_visited:
        result.append(visit_order)
    return tuple(result)


def _build_path(came_from, start, goal):
    """
    Reconstructs path from goal to start using parent pointers.
    
    Parameters:
        came_from: dict mapping node -> parent_node
        start: tuple, starting position
        goal: tuple, goal position
        
    Returns:
        list: path from start to goal, or None if no path exists
    """
    if goal not in came_from and goal != start:
        return None
        
    path = [goal]
    current = goal
    
    while current != start:
        if current not in came_from:
            return None  # broken path
        current = came_from[current]
        path.append(current)
    
    # reverse to get start->goal order
    return path[::-1]


def print_dijkstra_comparison():
    """
    Print a clear comparison between Dijkstra and A* for educational purposes.
    """
    print("=" * 70)
    print("DIJKSTRA vs A* - THE KEY DIFFERENCE")
    print("=" * 70)
    print()
    print("DIJKSTRA (Regular GPS):")
    print("  - Uses only actual cost: f(n) = g(n)")
    print("  - No heuristic guidance (h = 0 always)")  
    print("  - Explores like ripples in a pond")
    print("  - Systematic but slower")
    print("  - Guaranteed optimal path")
    print()
    print("A* (Smart GPS):")
    print("  - Uses actual cost + heuristic: f(n) = g(n) + h(n)")
    print("  - Manhattan distance guides toward goal")
    print("  - Explores in focused beam toward target")
    print("  - Faster with good heuristic")
    print("  - Guaranteed optimal path (with admissible heuristic)")
    print()
    print("FOR EMERGENCY VEHICLES:")
    print("  - Dijkstra: explores many unnecessary areas")
    print("  - A*: heads directly toward emergency")
    print("  - Time difference can save lives!")
    print("=" * 70)


# example usage and testing
if __name__ == "__main__":
    # create a simple test grid
    test_grid = np.array([
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0], 
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ])
    
    start_pos = (0, 0)
    goal_pos = (4, 4)
    
    print("Testing Dijkstra's Algorithm")
    print(f"Grid size: {test_grid.shape}")
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    print()
    
    # run dijkstra with full tracking
    result = dijkstra(test_grid, start_pos, goal_pos, 
                     return_costs=True, return_expanded=True, return_visited=True)
    
    path, costs, expanded, visit_order = result
    
    if path:
        print(f"✅ Path found: {path}")
        print(f"📊 Path length: {len(path)} steps")
        print(f"🔍 Nodes expanded: {expanded}")
        print(f"📈 Efficiency: {len(path)}/{expanded} = {len(path)/expanded:.2f}")
        print()
        print("Expansion order:", visit_order)
    else:
        print("❌ No path found!")
    
    print()
    print_dijkstra_comparison()


'''
 DIJKSTRA ALGORITHM SUMMARY

Purpose: Find shortest path in weighted graph with no heuristic guidance

Key Characteristics:
  - Uniform cost search (explores by actual distance only)
  - No direction sense toward goal
  - Guaranteed optimal for non-negative edge weights
  - Explores more nodes than A* but still mathematically optimal

Algorithm Steps:
  1. Initialize priority queue with start node (cost=0)
  2. While queue not empty:
     a. Extract node with lowest actual cost
     b. Mark as expanded
     c. If goal found, reconstruct path and return
     d. For each neighbor:
        - Calculate cost via current node
        - If better than known cost, update and add to queue
  3. Return None if no path exists

Time Complexity: O(V log V + E log V) = O(V log V) for grid graphs
Space Complexity: O(V) where V = number of grid cells

Comparison with A*:
  - Both guarantee optimal paths
  - Dijkstra explores more nodes (ripples outward)
  - A* explores fewer nodes (focused toward goal)
  - Choice depends on heuristic quality and performance requirements

Emergency Vehicle Application:
  - Dijkstra: reliable but explores entire areas
  - A*: faster route calculation = faster emergency response
  - Time savings can be critical in life-threatening situations

'''