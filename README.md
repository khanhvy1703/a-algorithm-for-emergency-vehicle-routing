# Emergency Vehicle Routing System

## Project Overview

This project implements an interactive emergency vehicle routing system that compares three pathfinding algorithms in real-time. The system simulates various urban conditions that emergency vehicles encounter, from traffic congestion to road construction, allowing users to observe how different algorithms handle these challenges.

The motivation behind this project was to explore how classical pathfinding algorithms perform in practical scenarios where emergency response time is critical. By visualizing the exploration patterns and final paths of each algorithm, we can better understand their strengths and weaknesses.

## Features

### Core Functionality

- **Real-time Algorithm Comparison**: Watch A*, Dijkstra's, and Greedy Best-First Search solve the same routing problem simultaneously
- **Interactive Map Editor**: Click and drag to create custom road conditions and obstacles
- **Dynamic Start/Goal Placement**: Set emergency vehicle location and destination anywhere on the map
- **Step-by-Step Visualization**: Educational mode that shows exactly how each algorithm explores the grid
- **Individual Algorithm View**: Detailed side-by-side comparison with independent controls for each algorithm

### Road Types and Conditions

The system simulates 15 different road conditions:
- Normal roads (standard travel speed)
- Highways (faster routes)
- Residential areas (slower, more cautious travel)
- Emergency lanes (dedicated fast routes)
- Heavy traffic (significant delays)
- Accident scenes (major obstacles)
- Construction zones (partial blockages)
- School zones (reduced speed areas)
- Flooded roads (severely impacted travel)
- Blocked roads (completely impassable)
- Tunnels and bridges (slightly slower passages)
- Parking lots (complex navigation)
- One-way streets (directional constraints)
- Maze walls (for testing scenarios)

### Visualization Options

- **Path Display**: Shows the final route each algorithm found
- **Exploration Visualization**: Displays all cells examined during search
- **Animation Speed Control**: Adjust visualization speed from very slow to instant
- **Color-Coded Algorithms**: Blue for A*, Red for Dijkstra, Orange for Greedy

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. Clone or download the project folder

2. Navigate to the project directory:
```bash
cd emergency_routing_project
```

3. Create a virtual environment (recommended):
```bash
python3 -m venv venv
```

4. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
venv\Scripts\activate
```

5. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Quick Start

From the project directory, run:
```bash
python scripts/main.py
```

### Controls and Interface

#### Map Editing
- **Left Panel**: Select road types to place on the map
- Click and drag on the grid to draw roads
- Each road type has different movement costs

#### Algorithm Control
- **Run Button**: Start pathfinding for all algorithms
- **Clear Paths**: Remove current solution paths
- **Clear Grid**: Reset the entire map
- **Reset**: Clear everything and start fresh

#### Visualization Options
- **Show Exploration**: Toggle visibility of explored cells
- **Show Paths**: Toggle visibility of solution paths
- **Step Mode**: Enable step-by-step visualization
- **Individual View**: Open detailed algorithm comparison

#### Step Mode Controls (When Enabled)
- **Run**: Auto-play through algorithm steps
- **Pause/Resume**: Control automatic stepping
- **Previous/Next**: Manual step control
- **Reset**: Return to first step
- **Speed +/-**: Adjust animation speed

#### Start/Goal Placement
- **Place Start**: Click to activate, then click on grid to set vehicle position
- **Place Goal**: Click to activate, then click on grid to set destination

### Keyboard Shortcuts

- **Space**: Pause/Resume animation
- **+/-**: Increase/Decrease animation speed
- **Left/Right Arrow**: Previous/Next step (in step mode)
- **Home**: Reset to first step (in step mode)

## Algorithms Implemented

### A* (A-Star)
A* uses both the actual distance traveled and a heuristic estimate to the goal. This makes it efficient while guaranteeing the optimal path. In our implementation, we use Manhattan distance as the heuristic, which works well for grid-based movement.

### Dijkstra's Algorithm
Dijkstra's algorithm explores uniformly in all directions, guaranteeing the shortest path but potentially exploring more cells than necessary. It's essentially A* without the heuristic component.

### Greedy Best-First Search
This algorithm prioritizes cells that appear closest to the goal, making it fast but not always optimal. It can miss better paths by being too focused on the apparent direction to the goal.

## Project Structure

```
emergency_routing_project/
├── scripts/
│   └── main.py              # Main application (all code in one file)
├── docs/                    # Documentation
├── original_analysis/       # Original team A* implementation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Implementation Details

### Grid System
The city is represented as a 2D grid where each cell has a road type. The grid size is configurable, defaulting to 50x50 cells. Each cell's road type determines the cost of traveling through it.

### Cost Calculation
Movement cost between cells is calculated as:
- Base cost (1.0 for orthogonal movement)
- Multiplied by the road type's cost factor
- Special handling for blocked roads (infinite cost)

### Heuristic Function
For A* and Greedy algorithms, we use Manhattan distance (|x1-x2| + |y1-y2|) as the heuristic. This is admissible for grid-based movement with no diagonal travel.

### Priority Queue Implementation
All three algorithms use Python's heapq module for efficient priority queue operations. The priority determines which cell to explore next:
- A*: f(n) = g(n) + h(n) (cost so far + heuristic)
- Dijkstra: g(n) (cost so far only)
- Greedy: h(n) (heuristic only)

## Performance Considerations

- The grid size significantly impacts performance. Larger grids require more computation
- Step-by-step mode stores all exploration data, using more memory but enabling detailed visualization
- The visualization runs at 60 FPS for smooth animation

## Testing Scenarios

The project includes several pre-built scenarios accessible through the "Load Scenario" menu:
- Rush hour traffic patterns
- Accident response situations
- Flooded district navigation
- School zone routing
- Highway vs. local road choices

## Future Enhancements

Potential areas for expansion:
- Dynamic obstacles that move over time
- Multiple emergency vehicles with coordination
- Real traffic data integration
- 3D visualization options
- Machine learning for pattern recognition
- Historical path analysis

## Troubleshooting

### Common Issues

1. **Application won't start**: Ensure pygame is installed correctly:
```bash
pip install --upgrade pygame
```

2. **Slow performance**: Try reducing the grid size or disabling exploration visualization

3. **Algorithms not finding paths**: Check that start and goal positions are set and not blocked

## Dependencies

- pygame >= 2.0.0 (Graphics and user interface)
- numpy >= 1.19.0 (Efficient array operations)
- Python standard library modules (heapq, enum, dataclasses, etc.)

## Credits

Developed as a course project for demonstrating pathfinding algorithms in practical scenarios. The project emphasizes educational value through visualization and interactivity.

## License

This project is for educational purposes. Feel free to use and modify for learning and research.

---

For questions or suggestions, please refer to the documentation in the docs/ folder or examine the well-commented source code.