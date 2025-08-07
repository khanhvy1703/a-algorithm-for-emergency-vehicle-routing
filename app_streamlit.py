"""
Emergency Vehicle Routing - Web Version
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import heapq
from enum import Enum
import time

# Set page config
st.set_page_config(
    page_title="Emergency Vehicle Routing",
    page_icon="🚑",
    layout="wide"
)

st.title("🚑 Emergency Vehicle Routing System")
st.markdown("Compare pathfinding algorithms for emergency vehicles")

# Sidebar controls
st.sidebar.header("Controls")

# Grid size
grid_size = st.sidebar.slider("Grid Size", 10, 50, 20)

# Algorithm selection
show_astar = st.sidebar.checkbox("A* Algorithm", value=True)
show_dijkstra = st.sidebar.checkbox("Dijkstra's Algorithm", value=True)
show_greedy = st.sidebar.checkbox("Greedy Best-First", value=True)

# Road types
class RoadType(Enum):
    NORMAL = 1.0
    HIGHWAY = 0.5
    TRAFFIC = 3.0
    BLOCKED = float('inf')

# Create grid
if 'grid' not in st.session_state:
    st.session_state.grid = np.ones((grid_size, grid_size))
    st.session_state.start = (0, 0)
    st.session_state.goal = (grid_size-1, grid_size-1)

# Grid editor
st.sidebar.subheader("Edit Map")
brush = st.sidebar.radio(
    "Select Road Type",
    ["Normal", "Highway", "Heavy Traffic", "Blocked"]
)

if st.sidebar.button("Clear Grid"):
    st.session_state.grid = np.ones((grid_size, grid_size))

if st.sidebar.button("Add Random Obstacles"):
    for _ in range(grid_size):
        x, y = np.random.randint(0, grid_size, 2)
        if (x, y) != st.session_state.start and (x, y) != st.session_state.goal:
            st.session_state.grid[y, x] = RoadType.BLOCKED.value

# Pathfinding algorithms
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, grid):
    x, y = pos
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
            if grid[ny, nx] != float('inf'):
                neighbors.append((nx, ny))
    return neighbors

def astar(grid, start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored = []
    
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        explored.append(current)
        
        if current == goal:
            break
            
        for next_pos in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + grid[next_pos[1], next_pos[0]]
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(goal, next_pos)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current
    
    # Reconstruct path
    path = []
    current = goal
    while current and current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path, explored

def dijkstra(grid, start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored = []
    
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        explored.append(current)
        
        if current == goal:
            break
            
        for next_pos in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + grid[next_pos[1], next_pos[0]]
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                heapq.heappush(frontier, (new_cost, next_pos))
                came_from[next_pos] = current
    
    # Reconstruct path
    path = []
    current = goal
    while current and current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path, explored

def greedy(grid, start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    explored = []
    
    while frontier:
        _, current = heapq.heappop(frontier)
        explored.append(current)
        
        if current == goal:
            break
            
        for next_pos in get_neighbors(current, grid):
            if next_pos not in came_from:
                priority = heuristic(goal, next_pos)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current
    
    # Reconstruct path
    path = []
    current = goal
    while current and current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path, explored

# Run algorithms
if st.button("🏃 Run Pathfinding", type="primary"):
    col1, col2, col3 = st.columns(3)
    
    results = {}
    
    if show_astar:
        start_time = time.time()
        path, explored = astar(st.session_state.grid, st.session_state.start, st.session_state.goal)
        elapsed = time.time() - start_time
        results['A*'] = {'path': path, 'explored': explored, 'time': elapsed}
    
    if show_dijkstra:
        start_time = time.time()
        path, explored = dijkstra(st.session_state.grid, st.session_state.start, st.session_state.goal)
        elapsed = time.time() - start_time
        results['Dijkstra'] = {'path': path, 'explored': explored, 'time': elapsed}
    
    if show_greedy:
        start_time = time.time()
        path, explored = greedy(st.session_state.grid, st.session_state.start, st.session_state.goal)
        elapsed = time.time() - start_time
        results['Greedy'] = {'path': path, 'explored': explored, 'time': elapsed}
    
    # Display results
    cols = st.columns(len(results))
    for i, (name, data) in enumerate(results.items()):
        with cols[i]:
            st.subheader(name)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(5, 5))
            
            # Draw grid
            display_grid = st.session_state.grid.copy()
            masked_grid = np.ma.masked_where(display_grid == float('inf'), display_grid)
            
            ax.imshow(masked_grid, cmap='RdYlGn_r', alpha=0.3)
            
            # Draw explored cells
            for x, y in data['explored']:
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                        linewidth=0, 
                                        edgecolor='none',
                                        facecolor='lightblue',
                                        alpha=0.3)
                ax.add_patch(rect)
            
            # Draw path
            if data['path']:
                path_array = np.array(data['path'])
                ax.plot(path_array[:, 0], path_array[:, 1], 
                       'b-', linewidth=3, label='Path')
            
            # Mark start and goal
            ax.plot(st.session_state.start[0], st.session_state.start[1], 
                   'go', markersize=15, label='Start')
            ax.plot(st.session_state.goal[0], st.session_state.goal[1], 
                   'ro', markersize=15, label='Goal')
            
            ax.set_xlim(-0.5, grid_size-0.5)
            ax.set_ylim(grid_size-0.5, -0.5)
            ax.grid(True, alpha=0.2)
            ax.set_aspect('equal')
            
            st.pyplot(fig)
            
            # Stats
            st.metric("Path Length", len(data['path']))
            st.metric("Cells Explored", len(data['explored']))
            st.metric("Time (ms)", f"{data['time']*1000:.2f}")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    1. **Adjust Grid Size**: Use the slider to change map size
    2. **Select Algorithms**: Choose which algorithms to compare
    3. **Edit Map**: Select a road type and click 'Add Random Obstacles'
    4. **Run Pathfinding**: Click the Run button to see results
    5. **Compare**: Look at path length, cells explored, and time
    
    **Road Types:**
    - 🟢 Normal: Standard roads
    - 🔵 Highway: Faster travel
    - 🔴 Traffic: Slower travel
    - ⬛ Blocked: Impassable
    """)

st.markdown("---")
st.caption("Emergency Vehicle Routing System - Algorithm Comparison Tool")