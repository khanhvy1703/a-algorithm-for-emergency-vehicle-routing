#!/usr/bin/env python3
"""
Emergency Vehicle Routing System

Compares three pathfinding algorithms (A*, Dijkstra, Greedy) for emergency vehicle
routing through different road conditions.
"""

import heapq
import time
import math
import json
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

import pygame
import numpy as np

pygame.init()
pygame.font.init()

# Configuration settings
@dataclass
class Config:
    """Window and grid configuration"""
    WINDOW_WIDTH = 1600      # Window width in pixels
    WINDOW_HEIGHT = 900      # Window height in pixels
    DEFAULT_GRID_SIZE = 50   # Grid size (50x50 cells)
    MIN_CELL_SIZE = 8        # Min cell size when zoomed
    MAX_CELL_SIZE = 40       # Max cell size when zoomed

class Colors:
    """Color definitions for the UI"""
    BACKGROUND = (255, 255, 255)  # White background
    GRID_LINE = (200, 200, 200)   # Gray grid lines
    TEXT = (0, 0, 0)              # Black text
    
    # Path colors for each algorithm
    PATH_ASTAR = (0, 100, 255)      # Blue
    PATH_DIJKSTRA = (255, 0, 0)     # Red
    PATH_GREEDY = (255, 165, 0)     # Orange
    
    # Exploration colors
    EXPLORED_ASTAR = (0, 0, 128, 100)
    EXPLORED_DIJKSTRA = (139, 0, 0, 100)
    EXPLORED_GREEDY = (0, 100, 0, 255)
    
    # Start and goal markers
    START = (0, 255, 0)  # Green
    GOAL = (255, 0, 0)   # Red
    
    # UI colors
    BUTTON_NORMAL = (240, 240, 240)
    BUTTON_HOVER = (220, 220, 250)
    BUTTON_ACTIVE = (180, 180, 220)
    BUTTON_TOGGLED = (100, 150, 250)
    BUTTON_TOGGLED_HOVER = (120, 170, 255)
    PANEL_BG = (255, 255, 255)
    SELECTED_BORDER = (0, 120, 255)
    BUTTON_BORDER = (100, 100, 100)
    BUTTON_DISABLED = (220, 220, 220)
    TEXT_DISABLED = (150, 150, 150)

class RoadType(Enum):
    """Different road types with their travel costs.
    Lower values = faster travel, higher values = slower/harder travel
    """
    NORMAL = 1.0           # Standard city road
    HIGHWAY = 0.8          # Faster travel on highways
    RESIDENTIAL = 1.5      # Slower in residential areas
    EMERGENCY_LANE = 0.5   # Dedicated emergency vehicle lanes
    HEAVY_TRAFFIC = 3.0    # Congested roads
    ACCIDENT = 5.0         # Major delays due to accidents
    CONSTRUCTION = 4.0     # Construction zones
    SCHOOL_ZONE = 2.0      # Reduced speed near schools
    FLOODED = 20.0         # Severely impacted roads
    BLOCKED = float('inf') # Completely impassable
    TUNNEL = 1.2           # Slightly slower in tunnels
    BRIDGE = 1.1           # Slightly slower on bridges
    PARKING_LOT = 2.5      # Parking areas
    ONE_WAY = 1.0          # One-way streets
    MAZE_WALL = 100.0      # Obstacles for maze scenarios

    @property
    def color(self):
        """Returns the RGB color associated with each road type.
        
        This method provides visual distinction between different road
        conditions, making it easy to identify obstacles and opportunities
        on the map at a glance.
        """
        color_map = {
            RoadType.NORMAL: (255, 255, 255),  # Pure white for normal roads
            RoadType.HIGHWAY: (90, 90, 90),  # Dark gray for highways
            RoadType.RESIDENTIAL: (255, 180, 120),  # Darker coral for residential
            RoadType.EMERGENCY_LANE: (0, 255, 100),  # Lime green for emergency
            RoadType.HEAVY_TRAFFIC: (255, 100, 0),  # Dark orange for traffic
            RoadType.ACCIDENT: (148, 0, 211),  # Dark violet for accidents
            RoadType.CONSTRUCTION: (139, 69, 19),  # Saddle brown for construction
            RoadType.SCHOOL_ZONE: (255, 20, 147),  # Deep pink for school zones
            RoadType.FLOODED: (0, 191, 255),  # Deep sky blue for flooded
            RoadType.BLOCKED: (0, 0, 0),  # Keep black for blocked
            RoadType.TUNNEL: (75, 0, 130),  # Indigo for tunnels
            RoadType.BRIDGE: (184, 134, 11),  # Dark goldenrod for bridges
            RoadType.PARKING_LOT: (192, 192, 192),  # Silver for parking
            RoadType.ONE_WAY: (245, 240, 250),  # Very light purple for one-way
            RoadType.MAZE_WALL: (64, 224, 208)  # Turquoise for maze walls - unique color
        }
        return color_map.get(self, (255, 255, 255))  # Default to white
    
    @property
    def icon(self):
        """Returns a single character icon representing the road type.
        
        These icons appear in the UI buttons and can be displayed on the
        grid cells for additional clarity.
        """
        icon_map = {
            RoadType.NORMAL: "",
            RoadType.HIGHWAY: "H",
            RoadType.RESIDENTIAL: "R",
            RoadType.EMERGENCY_LANE: "E",
            RoadType.HEAVY_TRAFFIC: "T",
            RoadType.ACCIDENT: "!",
            RoadType.CONSTRUCTION: "C",
            RoadType.SCHOOL_ZONE: "S",
            RoadType.FLOODED: "~",
            RoadType.BLOCKED: "X",
            RoadType.TUNNEL: "U",
            RoadType.BRIDGE: "B",
            RoadType.PARKING_LOT: "P",
            RoadType.ONE_WAY: "→",
            RoadType.MAZE_WALL: "#"
        }
        return icon_map.get(self, "")

class EducationalMode:
    """Handles step-by-step visualization of algorithm exploration.
    Records each step the algorithms take so we can replay them.
    """
    
    def __init__(self):
        self.enabled = False      # Whether step mode is active
        self.current_step = 0      # Current step being displayed
        self.max_steps = 0         # Total steps across all algorithms
        self.step_data = {         # Stores exploration data for each algorithm
            'astar': [],
            'dijkstra': [],
            'greedy': []
        }
        
    def reset(self):
        """Clears all recorded data for a fresh visualization.
        
        Called when starting a new pathfinding operation to ensure
        we don't mix data from different runs.
        """
        self.current_step = 0
        self.step_data = {
            'astar': [],
            'dijkstra': [],
            'greedy': []
        }
        self.max_steps = 0
        
    def record_step(self, algorithm: str, node: Tuple[int, int], 
                   came_from: Dict, current_path: List[Tuple[int, int]],
                   explored: Set[Tuple[int, int]], frontier: List):
        """Record a single exploration step"""
        step_info = {
            'node': node,
            'came_from': dict(came_from),  # Copy current state
            'path_so_far': list(current_path),
            'explored': set(explored),
            'frontier': [item[-1] for item in frontier]  # Extract nodes from heap
        }
        self.step_data[algorithm].append(step_info)
        
    def update_max_steps(self):
        """Update the maximum number of steps across all algorithms"""
        self.max_steps = max(
            len(self.step_data['astar']),
            len(self.step_data['dijkstra']),
            len(self.step_data['greedy'])
        )
        
    def get_current_state(self, algorithm: str):
        """Get the state at current step for given algorithm"""
        if self.current_step >= len(self.step_data[algorithm]):
            # Return final state if we've gone past the end
            if self.step_data[algorithm]:
                return self.step_data[algorithm][-1]
            return None
        return self.step_data[algorithm][self.current_step]
        
    def next_step(self):
        """Advance to next step"""
        if self.current_step < self.max_steps - 1:
            self.current_step += 1
            return True
        return False
        
    def prev_step(self):
        """Go back to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            return True
        return False

# ============================================================================
# UI COMPONENTS
# ============================================================================

class Button:
    """Interactive button class"""
    def __init__(self, x, y, width, height, text, font, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.action = action
        self.hovered = False
        self.active = False
        
    def draw(self, screen):
        """Draw the button"""
        if self.active:
            color = Colors.BUTTON_ACTIVE
            border_color = Colors.SELECTED_BORDER
            border_width = 3
            # Special colors for Place Start/Goal buttons
            if self.text == "Place Start":
                border_color = Colors.START
            elif self.text == "Place Goal":
                border_color = Colors.GOAL
        elif self.hovered:
            color = Colors.BUTTON_HOVER
            border_color = Colors.BUTTON_BORDER
            border_width = 2
        else:
            color = Colors.BUTTON_NORMAL
            border_color = Colors.BUTTON_BORDER
            border_width = 1
            
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, border_color, self.rect, border_width)
        
        # Draw text with better contrast
        text_color = Colors.TEXT if not self.active else (255, 255, 255)
        text_surface = self.font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        """Handle mouse events"""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.active and self.rect.collidepoint(event.pos) and self.action:
                self.action()
            self.active = False
        return False

class ToggleButton(Button):
    """Toggle button that maintains state"""
    def __init__(self, x, y, width, height, text, font, initial_state=False, callback=None):
        super().__init__(x, y, width, height, text, font)
        self.toggled = initial_state
        self.callback = callback
        
    def draw(self, screen):
        """Draw toggle button with clear state indication"""
        if self.toggled:
            color = Colors.BUTTON_TOGGLED_HOVER if self.hovered else Colors.BUTTON_TOGGLED
            border_color = Colors.SELECTED_BORDER
            border_width = 3
            text_color = (255, 255, 255)  # White text on blue background
        else:
            color = Colors.BUTTON_HOVER if self.hovered else Colors.BUTTON_NORMAL
            border_color = Colors.BUTTON_BORDER
            border_width = 2 if self.hovered else 1
            text_color = Colors.TEXT
            
        # Draw button background
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, border_color, self.rect, border_width)
        
        # Add visual indicator for toggled state
        if self.toggled:
            # Draw a small indicator dot
            indicator_rect = pygame.Rect(self.rect.left + 8, self.rect.centery - 4, 8, 8)
            pygame.draw.circle(screen, (255, 255, 255), indicator_rect.center, 4)
            
        # Draw text
        text_surface = self.font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        """Handle toggle"""
        if super().handle_event(event):
            self.toggled = not self.toggled
            if self.callback:
                self.callback(self.toggled)
            return True
        return False

class Direction(Enum):
    """Four cardinal directions for movement"""
    NORTH = (0, -1)  # Up
    EAST = (1, 0)    # Right
    SOUTH = (0, 1)   # Down
    WEST = (-1, 0)   # Left

class CityGrid:
    """Represents the city as a 2D grid of road cells.
    
    This class manages the city layout, including different road types,
    one-way street restrictions, and start/goal positions. It provides
    methods for pathfinding algorithms to query valid moves and costs.
    """
    
    def __init__(self, width: int = 50, height: int = 50):
        self.width = width
        self.height = height
        # Initialize all cells as normal roads
        self.grid = [[RoadType.NORMAL for _ in range(width)] for _ in range(height)]
        self.one_way_streets = {}  # Maps positions to allowed directions
        self.start = (1, 1)  # Default start near top-left
        self.goal = (width - 2, height - 2)  # Default goal near bottom-right
        
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find all valid neighboring cells from a given position.
        
        This method considers grid boundaries, blocked roads, and one-way
        street restrictions to return only accessible neighbors.
        """
        x, y = pos
        neighbors = []
        
        for direction in Direction:
            dx, dy = direction.value
            nx, ny = x + dx, y + dy
            
            if not self.is_valid_position(nx, ny):
                continue
                
            if self.grid[ny][nx] == RoadType.BLOCKED:
                continue
            
            # Check one-way restrictions
            if pos in self.one_way_streets:
                allowed_dir = self.one_way_streets[pos]
                if (dx, dy) != allowed_dir.value:
                    continue
                    
            if (nx, ny) in self.one_way_streets:
                allowed_dir = self.one_way_streets[(nx, ny)]
                if (-dx, -dy) != allowed_dir.value:
                    continue
            
            neighbors.append((nx, ny))
            
        return neighbors
    
    def get_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        return self.grid[to_pos[1]][to_pos[0]].value
    
    def set_road_type(self, x: int, y: int, road_type: RoadType):
        if self.is_valid_position(x, y):
            self.grid[y][x] = road_type
            
    def clear_grid(self):
        self.grid = [[RoadType.NORMAL for _ in range(self.width)] 
                     for _ in range(self.height)]
        self.one_way_streets.clear()

class PathfindingAlgorithm:
    """Base class for pathfinding algorithms.
    Each algorithm (A*, Dijkstra, Greedy) inherits from this.
    """
    
    def __init__(self, grid: CityGrid, name: str):
        self.grid = grid
        self.name = name
        self.path = []               # Final path from start to goal
        self.explored = []           # List of explored nodes in order
        self.explored_set = set()   # Set for O(1) lookup of explored nodes
        self.nodes_explored = 0     # Counter for performance metrics
        self.search_time = 0.0      # Time taken to find path
        self.path_cost = 0.0        # Total cost of the found path
        self.exploration_order = {}  # Maps nodes to exploration order
        
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        raise NotImplementedError()
        
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path
        
    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        if not path or len(path) < 2:
            return 0.0
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self.grid.get_cost(path[i], path[i + 1])
        return total_cost

class AStarAlgorithm(PathfindingAlgorithm):
    """A* Search Algorithm"""
    
    def __init__(self, grid: CityGrid):
        super().__init__(grid, "A* Search")
        self.educational_mode = None  # Will be set if needed
        
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        start_time = time.time()
        
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        self.explored = []
        self.explored_set = set()
        self.nodes_explored = 0
        exploration_step = 0
        
        # Record initial state for step 0
        if hasattr(self, 'educational_mode') and self.educational_mode and self.educational_mode.enabled:
            self.educational_mode.record_step(
                'astar', start, came_from, [start],
                set(), open_set
            )
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current in self.explored_set:
                continue
                
            self.explored.append(current)
            self.explored_set.add(current)
            self.exploration_order[current] = exploration_step
            exploration_step += 1
            self.nodes_explored += 1
            
            # Record step for educational mode (A*)
            if hasattr(self, 'educational_mode') and self.educational_mode and self.educational_mode.enabled:
                partial_path = []
                if current in came_from:
                    partial_path = self.reconstruct_path(came_from, current)
                self.educational_mode.record_step(
                    'astar', current, came_from, partial_path,
                    self.explored_set, open_set
                )
            
            if current == goal:
                self.path = self.reconstruct_path(came_from, current)
                self.path_cost = self.calculate_path_cost(self.path)
                self.search_time = time.time() - start_time
                return self.path
                
            for neighbor in self.grid.get_neighbors(current):
                tentative_g = g_score[current] + self.grid.get_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
                    
        self.search_time = time.time() - start_time
        return []

class DijkstraAlgorithm(PathfindingAlgorithm):
    """Dijkstra's Algorithm"""
    
    def __init__(self, grid: CityGrid):
        super().__init__(grid, "Dijkstra's Algorithm")
        self.educational_mode = None  # Will be set if needed
        
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        start_time = time.time()
        
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))
        
        came_from = {}
        cost = {start: 0}
        
        self.explored = []
        self.explored_set = set()
        self.nodes_explored = 0
        exploration_step = 0
        
        # Record initial state for step 0
        if hasattr(self, 'educational_mode') and self.educational_mode and self.educational_mode.enabled:
            self.educational_mode.record_step(
                'dijkstra', start, came_from, [start],
                set(), open_set
            )
        
        while open_set:
            current_cost, _, current = heapq.heappop(open_set)
            
            if current in self.explored_set:
                continue
                
            self.explored.append(current)
            self.explored_set.add(current)
            self.exploration_order[current] = exploration_step
            exploration_step += 1
            self.nodes_explored += 1
            
            # Record step for educational mode (Dijkstra)
            if hasattr(self, 'educational_mode') and self.educational_mode and self.educational_mode.enabled:
                partial_path = []
                if current in came_from:
                    partial_path = self.reconstruct_path(came_from, current)
                self.educational_mode.record_step(
                    'dijkstra', current, came_from, partial_path,
                    self.explored_set, open_set
                )
            
            if current == goal:
                self.path = self.reconstruct_path(came_from, current)
                self.path_cost = self.calculate_path_cost(self.path)
                self.search_time = time.time() - start_time
                return self.path
                
            for neighbor in self.grid.get_neighbors(current):
                new_cost = cost[current] + self.grid.get_cost(current, neighbor)
                
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (new_cost, counter, neighbor))
                    
        self.search_time = time.time() - start_time
        return []

class GreedyBestFirstSearch(PathfindingAlgorithm):
    """Greedy Best-First Search"""
    
    def __init__(self, grid: CityGrid):
        super().__init__(grid, "Greedy Best-First")
        self.educational_mode = None  # Will be set if needed
        
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        start_time = time.time()
        
        open_set = []
        counter = 0
        heapq.heappush(open_set, (self.heuristic(start, goal), counter, start))
        
        came_from = {}
        
        self.explored = []
        self.explored_set = set()
        self.nodes_explored = 0
        exploration_step = 0
        
        # Record initial state for step 0
        if hasattr(self, 'educational_mode') and self.educational_mode and self.educational_mode.enabled:
            self.educational_mode.record_step(
                'greedy', start, came_from, [start],
                set(), open_set
            )
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current in self.explored_set:
                continue
                
            self.explored.append(current)
            self.explored_set.add(current)
            self.exploration_order[current] = exploration_step
            exploration_step += 1
            self.nodes_explored += 1
            
            # Record step for educational mode (Greedy)
            if hasattr(self, 'educational_mode') and self.educational_mode and self.educational_mode.enabled:
                partial_path = []
                if current == start:
                    partial_path = [start]
                elif current in came_from:
                    partial_path = self.reconstruct_path(came_from, current)
                self.educational_mode.record_step(
                    'greedy', current, came_from, partial_path,
                    self.explored_set, open_set
                )
            
            if current == goal:
                self.path = self.reconstruct_path(came_from, current)
                self.path_cost = self.calculate_path_cost(self.path)
                self.search_time = time.time() - start_time
                return self.path
                
            for neighbor in self.grid.get_neighbors(current):
                if neighbor not in self.explored_set:
                    came_from[neighbor] = current
                    h = self.heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (h, counter, neighbor))
                    
        self.search_time = time.time() - start_time
        return []

# ============================================================================
# SCENARIO MANAGER (All previous scenarios)
# ============================================================================

class ScenarioManager:
    """Manages all predefined scenarios"""
    
    def __init__(self):
        self.scenarios = [
            # Simple Scenarios
            ("Basic Obstacles", self.basic_obstacles),
            ("Traffic Zones", self.traffic_zones),
            ("Emergency Lane Test", self.emergency_lane_test),
            ("Construction Zone", self.construction_zone),
            
            # Realistic Scenarios
            ("Downtown Rush Hour", self.downtown_rush_hour),
            ("Flooded District", self.flooded_district),
            ("School Zones", self.school_zones),
            ("Highway vs Local", self.highway_vs_local),
            ("Multi-Accident", self.multi_accident),
            ("Tunnel Network", self.tunnel_network),
            ("Bridge Congestion", self.bridge_congestion),
            
            # Extreme Scenarios
            ("Mega City Maze", self.mega_city_maze),
            ("Spiral of Doom", self.spiral_of_doom),
            ("The Gauntlet", self.the_gauntlet),
            ("Island Hopping", self.island_hopping),
            ("One-Way Nightmare", self.one_way_nightmare),
            
            # Educational Scenarios
            ("Algorithm Demo", self.algorithm_demo),
            ("Greedy Trap", self.greedy_trap),
        ]
        
    def basic_obstacles(self, grid: CityGrid):
        """Simple obstacles"""
        grid.clear_grid()
        for i in range(5, 15):
            grid.set_road_type(i, grid.height // 2, RoadType.BLOCKED)
        for i in range(5, 15):
            grid.set_road_type(grid.width // 2, i, RoadType.BLOCKED)
            
    def traffic_zones(self, grid: CityGrid):
        """Different traffic zones"""
        grid.clear_grid()
        for x in range(grid.width // 3, 2 * grid.width // 3):
            for y in range(grid.height // 3, 2 * grid.height // 3):
                grid.set_road_type(x, y, RoadType.HEAVY_TRAFFIC)
                
    def emergency_lane_test(self, grid: CityGrid):
        """Emergency lane through traffic"""
        grid.clear_grid()
        for x in range(grid.width):
            for y in range(grid.height):
                grid.set_road_type(x, y, RoadType.HEAVY_TRAFFIC)
        for x in range(grid.width):
            grid.set_road_type(x, grid.height // 2, RoadType.EMERGENCY_LANE)
            
    def construction_zone(self, grid: CityGrid):
        """Construction with detours"""
        grid.clear_grid()
        for x in range(grid.width // 3, 2 * grid.width // 3):
            for y in range(grid.height // 2 - 2, grid.height // 2 + 3):
                grid.set_road_type(x, y, RoadType.CONSTRUCTION)
                
    def downtown_rush_hour(self, grid: CityGrid):
        """Downtown traffic patterns"""
        grid.clear_grid()
        # Highways
        for i in range(grid.height):
            grid.set_road_type(grid.width // 4, i, RoadType.HIGHWAY)
            grid.set_road_type(3 * grid.width // 4, i, RoadType.HIGHWAY)
        for i in range(grid.width):
            grid.set_road_type(i, grid.height // 4, RoadType.HIGHWAY)
            grid.set_road_type(i, 3 * grid.height // 4, RoadType.HIGHWAY)
        # Downtown core
        for x in range(grid.width // 3, 2 * grid.width // 3):
            for y in range(grid.height // 3, 2 * grid.height // 3):
                if grid.grid[y][x] != RoadType.HIGHWAY:
                    grid.set_road_type(x, y, RoadType.HEAVY_TRAFFIC)
                    
    def flooded_district(self, grid: CityGrid):
        """Flooding scenario"""
        grid.clear_grid()
        center_x, center_y = grid.width // 2, grid.height // 2
        for x in range(grid.width):
            for y in range(grid.height):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 15:
                    grid.set_road_type(x, y, RoadType.FLOODED)
                elif dist < 20:
                    grid.set_road_type(x, y, RoadType.HEAVY_TRAFFIC)
                    
    def school_zones(self, grid: CityGrid):
        """Morning school zones"""
        grid.clear_grid()
        spacing = 15
        for x in range(spacing, grid.width, spacing):
            for y in range(spacing, grid.height, spacing):
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        if grid.is_valid_position(x + dx, y + dy):
                            grid.set_road_type(x + dx, y + dy, RoadType.SCHOOL_ZONE)
                            
    def highway_vs_local(self, grid: CityGrid):
        """Highway around vs local through"""
        grid.clear_grid()
        # Residential center
        for x in range(grid.width):
            for y in range(grid.height // 2 - 5, grid.height // 2 + 5):
                grid.set_road_type(x, y, RoadType.RESIDENTIAL)
        # Highway ring
        for i in range(grid.width):
            grid.set_road_type(i, 5, RoadType.HIGHWAY)
            grid.set_road_type(i, grid.height - 6, RoadType.HIGHWAY)
        for i in range(5, grid.height - 5):
            grid.set_road_type(5, i, RoadType.HIGHWAY)
            grid.set_road_type(grid.width - 6, i, RoadType.HIGHWAY)
            
    def multi_accident(self, grid: CityGrid):
        """Multiple accidents"""
        grid.clear_grid()
        accident_points = [
            (grid.width // 3, grid.height // 3),
            (2 * grid.width // 3, grid.height // 3),
            (grid.width // 2, grid.height // 2)
        ]
        for x, y in accident_points:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if grid.is_valid_position(x + dx, y + dy):
                        grid.set_road_type(x + dx, y + dy, RoadType.ACCIDENT)
                        
    def tunnel_network(self, grid: CityGrid):
        """Underground tunnels"""
        grid.clear_grid()
        # Surface is slow
        for x in range(grid.width):
            for y in range(grid.height):
                grid.set_road_type(x, y, RoadType.HEAVY_TRAFFIC)
        # Tunnel paths
        tunnel_points = [(10, 10), (grid.width - 10, 10), 
                        (10, grid.height - 10), (grid.width - 10, grid.height - 10)]
        for i in range(len(tunnel_points)):
            for j in range(i + 1, len(tunnel_points)):
                x1, y1 = tunnel_points[i]
                x2, y2 = tunnel_points[j]
                # L-shaped tunnel
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    grid.set_road_type(x, y1, RoadType.TUNNEL)
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    grid.set_road_type(x2, y, RoadType.TUNNEL)
                    
    def bridge_congestion(self, grid: CityGrid):
        """Limited bridges"""
        grid.clear_grid()
        # River
        river_y = grid.height // 2
        for x in range(grid.width):
            for dy in range(-3, 4):
                if grid.is_valid_position(x, river_y + dy):
                    grid.set_road_type(x, river_y + dy, RoadType.BLOCKED)
        # Bridges
        bridge_positions = [grid.width // 4, grid.width // 2, 3 * grid.width // 4]
        for bridge_x in bridge_positions:
            for dy in range(-3, 4):
                if grid.is_valid_position(bridge_x, river_y + dy):
                    if bridge_x == grid.width // 2:
                        grid.set_road_type(bridge_x, river_y + dy, RoadType.HEAVY_TRAFFIC)
                    else:
                        grid.set_road_type(bridge_x, river_y + dy, RoadType.BRIDGE)
                        
    def mega_city_maze(self, grid: CityGrid):
        """Complex maze"""
        grid.clear_grid()
        # Maze walls
        for x in range(0, grid.width, 4):
            for y in range(grid.height):
                if y % 8 != 4:
                    grid.set_road_type(x, y, RoadType.MAZE_WALL)
        for y in range(0, grid.height, 4):
            for x in range(grid.width):
                if x % 8 != 4:
                    grid.set_road_type(x, y, RoadType.MAZE_WALL)
        # Highway ring
        for i in range(grid.width):
            grid.set_road_type(i, 0, RoadType.HIGHWAY)
            grid.set_road_type(i, grid.height - 1, RoadType.HIGHWAY)
        for i in range(grid.height):
            grid.set_road_type(0, i, RoadType.HIGHWAY)
            grid.set_road_type(grid.width - 1, i, RoadType.HIGHWAY)
            
    def spiral_of_doom(self, grid: CityGrid):
        """Spiral pattern"""
        grid.clear_grid()
        center_x, center_y = grid.width // 2, grid.height // 2
        for x in range(grid.width):
            for y in range(grid.height):
                dist = abs(x - center_x) + abs(y - center_y)
                if dist < 5:
                    road_type = RoadType.NORMAL
                elif dist < 10:
                    road_type = RoadType.HEAVY_TRAFFIC
                elif dist < 15:
                    road_type = RoadType.ACCIDENT
                else:
                    road_type = RoadType.MAZE_WALL
                grid.set_road_type(x, y, road_type)
                
    def the_gauntlet(self, grid: CityGrid):
        """Progressive difficulty"""
        grid.clear_grid()
        band_width = grid.width // 6
        difficulties = [RoadType.NORMAL, RoadType.HEAVY_TRAFFIC, 
                       RoadType.ACCIDENT, RoadType.CONSTRUCTION,
                       RoadType.FLOODED, RoadType.MAZE_WALL]
        for i, difficulty in enumerate(difficulties):
            start_x = i * band_width
            end_x = min((i + 1) * band_width, grid.width)
            for x in range(start_x, end_x):
                for y in range(grid.height):
                    grid.set_road_type(x, y, difficulty)
        # Hidden emergency lane
        for x in range(grid.width):
            grid.set_road_type(x, grid.height // 3, RoadType.EMERGENCY_LANE)
            
    def island_hopping(self, grid: CityGrid):
        """Islands with bridges"""
        grid.clear_grid()
        # Water everywhere
        for x in range(grid.width):
            for y in range(grid.height):
                grid.set_road_type(x, y, RoadType.FLOODED)
        # Islands
        island_centers = [(10, 10), (grid.width - 10, 10),
                         (10, grid.height - 10), (grid.width - 10, grid.height - 10),
                         (grid.width // 2, grid.height // 2)]
        for cx, cy in island_centers:
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx*dx + dy*dy <= 25 and grid.is_valid_position(cx + dx, cy + dy):
                        grid.set_road_type(cx + dx, cy + dy, RoadType.NORMAL)
        # Connect some islands
        for i in range(len(island_centers) - 1):
            x1, y1 = island_centers[i]
            x2, y2 = island_centers[i + 1]
            steps = max(abs(x2 - x1), abs(y2 - y1))
            for j in range(steps + 1):
                t = j / steps if steps > 0 else 0
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if grid.is_valid_position(x, y):
                    grid.set_road_type(x, y, RoadType.BRIDGE)
                    
    def one_way_nightmare(self, grid: CityGrid):
        """Complex one-way streets"""
        grid.clear_grid()
        # Grid of one-way streets
        for x in range(0, grid.width, 4):
            for y in range(grid.height):
                if x < grid.width:
                    direction = Direction.SOUTH if x % 8 == 0 else Direction.NORTH
                    grid.one_way_streets[(x, y)] = direction
        for y in range(0, grid.height, 4):
            for x in range(grid.width):
                if y < grid.height and x % 4 != 0:
                    direction = Direction.EAST if y % 8 == 0 else Direction.WEST
                    grid.one_way_streets[(x, y)] = direction
                    
    def algorithm_demo(self, grid: CityGrid):
        """Clear algorithm differences"""
        grid.clear_grid()
        grid.start = (5, 5)
        grid.goal = (grid.width - 5, grid.height - 5)
        # Some obstacles
        for i in range(20, 30):
            grid.set_road_type(i, grid.height // 2, RoadType.BLOCKED)
            grid.set_road_type(grid.width // 2, i, RoadType.BLOCKED)
            
    def greedy_trap(self, grid: CityGrid):
        """Trap for greedy algorithm"""
        grid.clear_grid()
        # U-shaped wall
        center_x = grid.width // 2
        for y in range(10, grid.height - 10):
            grid.set_road_type(center_x, y, RoadType.MAZE_WALL)
        for x in range(center_x - 10, center_x + 11):
            grid.set_road_type(x, 10, RoadType.MAZE_WALL)
        grid.start = (center_x - 5, 15)
        grid.goal = (center_x + 5, 15)

# ============================================================================
# MAIN INTERACTIVE APPLICATION
# ============================================================================

class InteractiveRoutingSystem:
    """Main application with mouse interaction and map editing"""
    
    def __init__(self):
        # Make window resizable
        self.clock = pygame.time.Clock()
        # Store window dimensions as instance variables
        self.window_width = Config.WINDOW_WIDTH
        self.window_height = Config.WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), 
                                             pygame.RESIZABLE)
        pygame.display.set_caption("Emergency Vehicle Routing - Interactive Edition")
        
        # Create initial fonts (will be updated based on window size)
        self.update_fonts()
        
        # Grid and algorithms
        self.grid = CityGrid(50, 50)
        self.algorithms = {}
        self.scenario_manager = ScenarioManager()
        
        # Educational mode for step-by-step
        self.educational_mode = EducationalMode()
        
        # UI state
        self.selected_road_type = RoadType.NORMAL
        self.placing_start = False
        self.placing_goal = False
        self.show_exploration = True
        self.show_paths = True
        self.animation_speed = 10  # Start with slower default speed
        self.individual_view_mode = False  # New feature: individual algorithm view
        self.animation_frame = 0
        self.animating = False
        self.paused = False
        self.auto_stepping = False
        self.auto_step_timer = 0
        
        # Individual view step tracking for each algorithm
        self.individual_steps = {
            'astar': 0,
            'dijkstra': 0,
            'greedy': 0
        }
        self.individual_animating = {
            'astar': False,
            'dijkstra': False,
            'greedy': False
        }
        self.individual_paused = {
            'astar': False,
            'dijkstra': False,
            'greedy': False
        }
        self.individual_speeds = {
            'astar': 10,
            'dijkstra': 10,
            'greedy': 10
        }
        self.individual_animation_frames = {
            'astar': 0,
            'dijkstra': 0,
            'greedy': 0
        }
        
        # Mouse state
        self.mouse_drawing = False
        self.last_mouse_grid_pos = None
        
        # Layout
        self.setup_layout()
        
        # Create UI elements
        self.create_ui_elements()
        
        # Run initial search
        self.run_algorithms()
        
    def update_fonts(self):
        """Update font sizes based on window size"""
        # Scale fonts proportionally to window size
        base_size = min(self.window_width, self.window_height) / 50  # Base scaling factor
        
        self.font_large = pygame.font.Font(None, int(base_size * 2.2))  # ~36 at 1600x900
        self.font_medium = pygame.font.Font(None, int(base_size * 1.5))  # ~24 at 1600x900
        self.font_small = pygame.font.Font(None, int(base_size * 1.1))  # ~18 at 1600x900
        self.font_tiny = pygame.font.Font(None, int(base_size * 0.85))  # ~14 at 1600x900
        
    def setup_layout(self):
        """Calculate layout dimensions - fully responsive to window size"""
        # Update fonts for new window size
        self.update_fonts()
        
        # Fully proportional sizing - no caps, pure percentages
        self.left_panel_width = int(self.window_width * 0.16)  # 16% of screen width
        self.right_panel_width = int(self.window_width * 0.22)  # 22% of screen width
        
        # Proportional bar heights  
        self.top_bar_height = int(self.window_height * 0.08)  # 8% of height
        self.bottom_bar_height = int(self.window_height * 0.10)  # 10% of height
        
        self.grid_area = pygame.Rect(
            self.left_panel_width,
            self.top_bar_height,
            self.window_width - self.left_panel_width - self.right_panel_width,
            self.window_height - self.top_bar_height - self.bottom_bar_height
        )
        
        # Calculate cell size - pure dynamic scaling
        self.cell_size = min(
            self.grid_area.width // self.grid.width,
            self.grid_area.height // self.grid.height
        )
        # No limits at all - pure scaling
        
        # Center grid
        self.grid_offset_x = self.grid_area.x + (self.grid_area.width - self.cell_size * self.grid.width) // 2
        self.grid_offset_y = self.grid_area.y + (self.grid_area.height - self.cell_size * self.grid.height) // 2
        
    def calculate_grid_dimensions(self):
        """Calculate grid dimensions based on window size"""
        if not self.grid:
            return
            
        # Calculate cell size - pure dynamic scaling
        self.cell_size = min(
            self.grid_area.width // self.grid.width,
            self.grid_area.height // self.grid.height
        )
        
        # Center grid
        self.grid_offset_x = self.grid_area.x + (self.grid_area.width - self.cell_size * self.grid.width) // 2
        self.grid_offset_y = self.grid_area.y + (self.grid_area.height - self.cell_size * self.grid.height) // 2
    
    def create_ui_elements(self):
        """Create all UI elements - fully responsive"""
        self.buttons = []
        self.toggle_buttons = []
        
        # Proportional button sizes
        button_height = int(self.window_height * 0.035)  # 3.5% of height (~30px at 900px)
        button_spacing = int(self.window_width * 0.005)  # 0.5% of width (~8px at 1600px)
        
        # Available width for top bar buttons
        available_width = self.window_width - self.left_panel_width - self.right_panel_width
        start_x = self.left_panel_width + 20
        
        # Single row layout - all buttons on same line
        button_y = (self.top_bar_height - button_height) // 2  # Center vertically
        
        # Calculate button widths to fit all in one row - pure percentage
        num_buttons = 9  # Total buttons we need to fit (including Step Mode)
        total_spacing = button_spacing * (num_buttons - 1)
        button_width = (available_width - 40 - total_spacing) // num_buttons
        # No minimum - pure scaling
        
        # Navigation buttons
        current_x = start_x
        self.prev_scenario_btn = Button(current_x, button_y, button_width, button_height, 
                                       "< Prev", self.font_small, 
                                       self.prev_scenario)
        current_x += button_width + button_spacing
        
        self.next_scenario_btn = Button(current_x, button_y, button_width, button_height,
                                       "Next >", self.font_small,
                                       self.next_scenario)
        current_x += button_width + button_spacing
        
        self.custom_map_btn = Button(current_x, button_y, button_width, button_height,
                                    "Clear", self.font_small,
                                    self.clear_map)
        current_x += button_width + button_spacing
        
        # Control buttons
        self.run_btn = Button(current_x, button_y, button_width, button_height,
                             "Run", self.font_small,
                             self.run_animation)
        current_x += button_width + button_spacing
        
        self.pause_btn = Button(current_x, button_y, button_width, button_height,
                               "Pause", self.font_small,
                               self.toggle_pause)
        current_x += button_width + button_spacing
        
        self.restart_btn = Button(current_x, button_y, button_width, button_height,
                               "Restart", self.font_small,
                               self.restart_animation)
        current_x += button_width + button_spacing
        
        # Toggle buttons
        self.show_exploration_btn = ToggleButton(current_x, button_y, button_width, button_height,
                                                "Explore", self.font_small, True)
        current_x += button_width + button_spacing
        
        self.show_paths_btn = ToggleButton(current_x, button_y, button_width, button_height,
                                          "Paths", self.font_small, True)
        
        # Step mode button - add as 9th button
        current_x += button_width + button_spacing
        self.step_mode_btn = ToggleButton(current_x, button_y, button_width, button_height,
                                         "Step Mode", self.font_small, False,
                                         callback=self.on_step_mode_toggle)
        
        # Step control buttons - place them in the bottom bar when step mode is on
        # Position them in the lower portion of the bottom bar to avoid text
        bottom_bar_y = self.window_height - int(self.bottom_bar_height * 0.35)  # Lower in the bar
        bottom_center_x = self.left_panel_width + (self.window_width - self.left_panel_width - self.right_panel_width) // 2
        
        step_button_width = int(self.window_width * 0.065)  # ~100px at 1600px
        step_button_height = int(self.window_height * 0.04)  # ~35px at 900px
        step_button_spacing = int(self.window_width * 0.006)  # ~10px at 1600px
        
        # Center the step control buttons in bottom bar
        total_width = (step_button_width * 3) + (step_button_spacing * 2)
        step_start_x = bottom_center_x - total_width // 2
        
        self.prev_step_btn = Button(step_start_x, bottom_bar_y, step_button_width, step_button_height,
                                   "< Prev Step", self.font_small, self.prev_step)
        self.next_step_btn = Button(step_start_x + step_button_width + step_button_spacing, bottom_bar_y,
                                   step_button_width, step_button_height,
                                   "Next Step >", self.font_small, self.next_step)
        self.reset_steps_btn = Button(step_start_x + (step_button_width + step_button_spacing) * 2, bottom_bar_y,
                                     step_button_width, step_button_height,
                                     "Reset Steps", self.font_small, self.reset_steps)
        
        self.step_buttons = [self.prev_step_btn, self.next_step_btn, self.reset_steps_btn]
        
        # Add all buttons to lists
        self.buttons.extend([
            self.prev_scenario_btn, self.next_scenario_btn, self.custom_map_btn,
            self.run_btn, self.pause_btn, self.restart_btn
        ])
        self.toggle_buttons.extend([
            self.show_exploration_btn, self.show_paths_btn, self.step_mode_btn
        ])
        
        # Left panel - road type buttons with responsive sizing
        self.road_type_buttons = []
        button_x = 10
        button_y = int(self.window_height * 0.09)  # ~80px at 900px
        button_width = self.left_panel_width - 20  # Dynamic width
        button_height = int(self.window_height * 0.031)  # ~28px at 900px
        button_spacing = int(self.window_height * 0.007)  # ~6px at 900px
        
        for i, road_type in enumerate(RoadType):
            y = button_y + i * (button_height + button_spacing)
            # Shorten the text to fit with color square
            btn_text = f"{road_type.name[:12]}"  # Truncate long names
            btn = Button(button_x, y, button_width, button_height,
                        btn_text,
                        self.font_tiny,
                        lambda rt=road_type: self.select_road_type(rt))
            self.road_type_buttons.append((road_type, btn))
            
        # Special placement buttons - positioned after road type buttons
        num_road_types = len(list(RoadType))
        special_y = button_y + num_road_types * (button_height + button_spacing) + 20
        self.place_start_btn = Button(button_x, special_y,
                                     button_width, button_height,
                                     "Place Start", self.font_small,
                                     self.toggle_place_start)
        self.place_goal_btn = Button(button_x, special_y + button_height + button_spacing,
                                    button_width, button_height,
                                    "Place Goal", self.font_small,
                                    self.toggle_place_goal)
        
        self.buttons.extend([self.place_start_btn, self.place_goal_btn])
        
        # Scenario tracking
        self.current_scenario = 0
        
    def select_road_type(self, road_type: RoadType):
        """Select a road type for placement"""
        self.selected_road_type = road_type
        self.placing_start = False
        self.placing_goal = False
        
    def toggle_place_start(self):
        """Toggle start placement mode"""
        self.placing_start = not self.placing_start
        self.placing_goal = False
        # Update button visual state
        self.place_start_btn.active = self.placing_start
        self.place_goal_btn.active = False
        
    def toggle_place_goal(self):
        """Toggle goal placement mode"""
        self.placing_goal = not self.placing_goal
        self.placing_start = False
        # Update button visual state
        self.place_goal_btn.active = self.placing_goal
        self.place_start_btn.active = False
        
    def prev_scenario(self):
        """Load previous scenario"""
        self.current_scenario = (self.current_scenario - 1) % len(self.scenario_manager.scenarios)
        self.load_scenario(self.current_scenario)
        
    def next_scenario(self):
        """Load next scenario"""
        self.current_scenario = (self.current_scenario + 1) % len(self.scenario_manager.scenarios)
        self.load_scenario(self.current_scenario)
        
    def load_scenario(self, index: int):
        """Load a specific scenario"""
        name, func = self.scenario_manager.scenarios[index]
        func(self.grid)
        self.run_algorithms()
        
    def clear_map(self):
        """Clear the map for custom editing"""
        self.grid.clear_grid()
        self.run_algorithms()
        
    def run_animation(self):
        """Run or continue the animation"""
        # Make sure we have algorithms to animate
        if not self.algorithms or not self.grid or not self.grid.start or not self.grid.goal:
            # Run algorithms first if needed
            self.run_algorithms()
        
        if self.educational_mode.enabled:
            # In step mode, check if start/goal points have changed
            need_rerun = False
            if hasattr(self, 'last_run_start') and hasattr(self, 'last_run_goal'):
                if self.grid.start != self.last_run_start or self.grid.goal != self.last_run_goal:
                    need_rerun = True
            else:
                need_rerun = True
            
            if need_rerun:
                # Points changed or first run - re-run algorithms with new points
                self.last_run_start = self.grid.start
                self.last_run_goal = self.grid.goal
                
                # Reset and re-run with educational mode
                self.educational_mode.reset()
                self.run_algorithms()
                self.educational_mode.update_max_steps()
                self.educational_mode.current_step = 0
            
            # Start/resume auto-stepping (Run button should always work)
            self.auto_stepping = True
            self.paused = False
        else:
            # In normal mode, continue animation from current frame
            self.animating = True
            self.paused = False
    
    def restart_animation(self):
        """Restart animation from beginning"""
        if self.educational_mode.enabled:
            # In step mode, completely restart with clean grid
            self.auto_stepping = False
            self.paused = False
            
            # Reset educational mode and re-run algorithms
            if self.grid and self.grid.start and self.grid.goal:
                self.educational_mode.enabled = True
                self.educational_mode.reset()
                
                # Run algorithms to collect fresh step data
                self.algorithms = {
                    'astar': AStarAlgorithm(self.grid),
                    'dijkstra': DijkstraAlgorithm(self.grid),
                    'greedy': GreedyBestFirstSearch(self.grid)
                }
                
                # Set educational mode on algorithms
                for name, algo in self.algorithms.items():
                    algo.educational_mode = self.educational_mode
                    algo.algorithm_name = name
                
                # Run searches
                for name, algo in self.algorithms.items():
                    algo.search(self.grid.start, self.grid.goal)
                
                # Update max steps and go to step 0 (clean grid)
                self.educational_mode.update_max_steps()
                self.educational_mode.current_step = 0
        else:
            # In normal mode, reset animation frame
            self.animation_frame = 0
            self.animating = True
            self.paused = False
            
            # Make sure we have algorithms to animate
            if not self.algorithms or not self.grid or not self.grid.start or not self.grid.goal:
                self.run_algorithms()
        
    def toggle_pause(self):
        """Toggle animation pause/resume"""
        if self.educational_mode.enabled:
            # In step mode, toggle auto-stepping
            if self.auto_stepping:
                # Currently running - pause it
                self.auto_stepping = False
                self.paused = True
            elif self.paused:
                # Currently paused - resume it
                self.auto_stepping = True
                self.paused = False
            # If not auto_stepping and not paused (like after restart), do nothing
            # The Run button should be used to start from stopped state
        else:
            # In normal mode, toggle pause
            if self.animating:
                self.paused = not self.paused
            else:
                # If not animating, start the animation
                self.run_animation()
        
        
    def on_step_mode_toggle(self, enabled):
        """Handle Step Mode toggle"""
        if enabled:
            # Entering step mode - immediately run algorithms if we have start/goal
            if self.grid and self.grid.start and self.grid.goal:
                # Turn off Explore and Paths buttons during step mode
                self.show_exploration_btn.toggled = False
                self.show_paths_btn.toggled = False
                
                # Reset animation states when entering step mode
                self.animating = False
                self.paused = False
                self.auto_stepping = False
                
                self.educational_mode.enabled = True
                self.educational_mode.reset()
                
                # Run algorithms to collect step data
                self.algorithms = {
                    'astar': AStarAlgorithm(self.grid),
                    'dijkstra': DijkstraAlgorithm(self.grid),
                    'greedy': GreedyBestFirstSearch(self.grid)
                }
                
                # Set educational mode on algorithms
                for name, algo in self.algorithms.items():
                    algo.educational_mode = self.educational_mode
                    algo.algorithm_name = name
                
                # Run searches
                for name, algo in self.algorithms.items():
                    algo.search(self.grid.start, self.grid.goal)
                
                # Update max steps and reset to beginning
                self.educational_mode.update_max_steps()
                self.educational_mode.current_step = 0
        else:
            # Exiting step mode - restore normal view with exploration on
            self.educational_mode.enabled = False
            self.auto_stepping = False  # Stop auto-stepping
            self.paused = False  # Reset pause state
            self.show_exploration_btn.toggled = True  # Turn exploration back on
            self.show_paths_btn.toggled = True  # Turn paths back on
            if self.grid and self.grid.start and self.grid.goal:
                # Run algorithms normally to restore exploration data
                self.run_algorithms()
    
    def prev_step(self):
        """Go to previous step in educational mode"""
        if self.educational_mode.enabled:
            self.educational_mode.prev_step()
            
    def next_step(self):
        """Go to next step in educational mode"""
        if self.educational_mode.enabled:
            self.educational_mode.next_step()
            
    def reset_steps(self):
        """Reset to first step"""
        if self.educational_mode.enabled:
            self.educational_mode.current_step = 0
        
    def run_algorithms(self):
        """Run all pathfinding algorithms"""
        if not self.grid:
            return
            
        # Enable educational mode if Step Mode button is toggled OR if it's already enabled
        # (educational mode might be enabled from Individual View for A*)
        if not self.educational_mode.enabled:
            # Only check button if educational mode is not already enabled
            if hasattr(self, 'step_mode_btn') and self.step_mode_btn.toggled:
                self.educational_mode.enabled = True
                self.educational_mode.reset()
            else:
                self.educational_mode.enabled = False
            
        self.algorithms = {
            'astar': AStarAlgorithm(self.grid),
            'dijkstra': DijkstraAlgorithm(self.grid),
            'greedy': GreedyBestFirstSearch(self.grid)
        }
        
        # Set educational mode on algorithms if enabled
        if self.educational_mode.enabled:
            for name, algo in self.algorithms.items():
                algo.educational_mode = self.educational_mode
        
        # Run searches
        for name, algo in self.algorithms.items():
            algo.search(self.grid.start, self.grid.goal)
        
        # Update max steps after recording
        if self.educational_mode.enabled:
            self.educational_mode.update_max_steps()
            self.educational_mode.current_step = 0
            self.animating = False  # No animation in step mode
        else:
            # Normal animation mode
            self.animation_frame = 0
            self.animating = True
        
    def grid_pos_from_mouse(self, mouse_pos):
        """Convert mouse position to grid coordinates"""
        x = (mouse_pos[0] - self.grid_offset_x) // self.cell_size
        y = (mouse_pos[1] - self.grid_offset_y) // self.cell_size
        
        if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
            return (x, y)
        return None
        
    def handle_mouse_click(self, pos):
        """Handle mouse click on grid"""
        grid_pos = self.grid_pos_from_mouse(pos)
        if not grid_pos:
            return
            
        x, y = grid_pos
        
        if self.placing_start:
            self.grid.start = (x, y)
            self.placing_start = False  # Exit placement mode after placing
            self.place_start_btn.active = False
            # Reset animation state when start point changes
            if self.educational_mode.enabled:
                # Check if it was running before
                was_running = self.auto_stepping
                # Reset everything
                self.paused = False
                self.auto_stepping = False
                self.educational_mode.current_step = 0
                self.auto_step_timer = 0
                # Force reset educational mode to clear old step data
                self.educational_mode.reset()
                # Re-run algorithms to capture new step data
                self.run_algorithms()
                self.educational_mode.update_max_steps()
                # If it was running, start it again automatically
                if was_running:
                    self.auto_stepping = True
            else:
                # Reset normal animation state
                self.animation_frame = 0
                self.animation_counter = 0
                self.run_algorithms()
        elif self.placing_goal:
            self.grid.goal = (x, y)
            self.placing_goal = False  # Exit placement mode after placing
            self.place_goal_btn.active = False
            # Reset animation state when goal point changes
            if self.educational_mode.enabled:
                # Check if it was running before
                was_running = self.auto_stepping
                # Reset everything
                self.paused = False
                self.auto_stepping = False
                self.educational_mode.current_step = 0
                self.auto_step_timer = 0
                # Force reset educational mode to clear old step data
                self.educational_mode.reset()
                # Re-run algorithms to capture new step data
                self.run_algorithms()
                self.educational_mode.update_max_steps()
                # If it was running, start it again automatically
                if was_running:
                    self.auto_stepping = True
            else:
                # Reset normal animation state
                self.animation_frame = 0
                self.animation_counter = 0
                self.run_algorithms()
        else:
            # Place selected road type
            self.grid.set_road_type(x, y, self.selected_road_type)
            self.last_mouse_grid_pos = grid_pos
            
    def handle_mouse_drag(self, pos):
        """Handle mouse drag for continuous drawing"""
        if not self.mouse_drawing:
            return
            
        grid_pos = self.grid_pos_from_mouse(pos)
        if not grid_pos:
            return
            
        # Only update if moved to new cell
        if grid_pos != self.last_mouse_grid_pos:
            x, y = grid_pos
            self.grid.set_road_type(x, y, self.selected_road_type)
            self.last_mouse_grid_pos = grid_pos
            
    def draw(self):
        """Main draw method"""
        if self.individual_view_mode:
            # Individual view - fill with clean background
            self.screen.fill(Colors.BACKGROUND)
            # Draw only the individual view (no panels)
            self.draw_individual_view()
        else:
            # Normal view - fill with panel background
            self.screen.fill(Colors.PANEL_BG)
            # Draw all panels and grid
            self.draw_left_panel()
            self.draw_top_bar()
            self.draw_right_panel()
            self.draw_bottom_bar()
            # Draw grid
            self.draw_grid()
        
        # Draw UI elements only if not in individual view mode
        if not self.individual_view_mode:
            # Update Pause button text based on state
            if self.educational_mode.enabled:
                # In step mode
                if self.paused and not self.auto_stepping:
                    self.pause_btn.text = "Resume"
                else:
                    self.pause_btn.text = "Pause"
            else:
                # In normal mode
                if self.animating and self.paused:
                    self.pause_btn.text = "Resume"
                else:
                    self.pause_btn.text = "Pause"
            
            for btn in self.buttons:
                btn.draw(self.screen)
            # Draw toggle buttons (but disable Explore/Paths during step mode)
            for btn in self.toggle_buttons:
                if hasattr(self, 'step_mode_btn') and self.step_mode_btn.toggled:
                    # During step mode, gray out Explore and Paths buttons
                    if btn in [self.show_exploration_btn, self.show_paths_btn]:
                        # Draw disabled state
                        disabled_color = (200, 200, 200)
                        pygame.draw.rect(self.screen, disabled_color, btn.rect)
                        pygame.draw.rect(self.screen, (150, 150, 150), btn.rect, 1)
                        
                        # Draw text in gray
                        text_surface = self.font_small.render(btn.text, True, (150, 150, 150))
                        text_rect = text_surface.get_rect(center=btn.rect.center)
                        self.screen.blit(text_surface, text_rect)
                    else:
                        btn.draw(self.screen)
                else:
                    btn.draw(self.screen)
                
            # Draw step control buttons if in step mode
            if hasattr(self, 'step_mode_btn') and self.step_mode_btn.toggled:
                for btn in self.step_buttons:
                    btn.draw(self.screen)
                    
            for road_type, btn in self.road_type_buttons:
                btn.draw(self.screen)
                
                # Draw color preview square INSIDE the button on the right side
                square_size = btn.rect.height - 8
                color_rect = pygame.Rect(btn.rect.right - square_size - 4, 
                                        btn.rect.top + 4, 
                                        square_size, square_size)
                pygame.draw.rect(self.screen, road_type.color, color_rect)
                pygame.draw.rect(self.screen, (0, 0, 0), color_rect, 1)  # Black border
                
                # Show cost multiplier - position it properly within button bounds
                cost_text = self.font_tiny.render(f"({road_type.value}x)", True, Colors.TEXT)
                # Scale the position based on button height
                cost_y = btn.rect.bottom - int(btn.rect.height * 0.4)  # 40% from bottom
                self.screen.blit(cost_text, (btn.rect.left + 5, cost_y))
                
                # Highlight selected with better visual feedback
                if road_type == self.selected_road_type:
                    # Draw a filled background for selected button
                    selected_rect = btn.rect.inflate(4, 4)
                    pygame.draw.rect(self.screen, Colors.BUTTON_TOGGLED, selected_rect)
                    pygame.draw.rect(self.screen, Colors.SELECTED_BORDER, selected_rect, 3)
                    
                    # Redraw button content on top with white text
                    pygame.draw.rect(self.screen, road_type.color, color_rect)
                    pygame.draw.rect(self.screen, (255, 255, 255), color_rect, 2)
                    
                    # White text for selected button
                    text_surface = self.font_small.render(btn.text, True, (255, 255, 255))
                    text_rect = text_surface.get_rect(center=btn.rect.center)
                    text_rect.x = btn.rect.left + 5
                    self.screen.blit(text_surface, text_rect)
                    
                    cost_text = self.font_tiny.render(f"({road_type.value}x)", True, (255, 255, 255))
                    # Scale the position based on button height
                    cost_y = btn.rect.bottom - int(btn.rect.height * 0.4)  # 40% from bottom
                    self.screen.blit(cost_text, (btn.rect.left + 5, cost_y))
                
    def draw_left_panel(self):
        """Draw left panel with road types"""
        panel_rect = pygame.Rect(0, 0, self.left_panel_width, self.window_height)
        pygame.draw.rect(self.screen, Colors.PANEL_BG, panel_rect)
        pygame.draw.rect(self.screen, Colors.TEXT, panel_rect, 1)  # Thin border
        
        # Title
        title = self.font_medium.render("Road Types", True, Colors.TEXT)
        self.screen.blit(title, (10, 10))
        
        # Instructions
        inst1 = self.font_tiny.render("Click to select", True, Colors.TEXT)
        inst2 = self.font_tiny.render("Then click/drag on map", True, Colors.TEXT)
        self.screen.blit(inst1, (10, 40))
        self.screen.blit(inst2, (10, 55))
        
        # Draw placement mode indicator at bottom of left panel
        mode_text = ""
        mode_color = Colors.TEXT
        if self.placing_start:
            mode_text = "🟢 PLACING START"
            mode_color = Colors.START
        elif self.placing_goal:
            mode_text = "🔴 PLACING GOAL"
            mode_color = Colors.GOAL
        elif self.selected_road_type:
            road_name = self.selected_road_type.name.replace('_', ' ').title()
            mode_text = f"✏️ {road_name}"
            mode_color = Colors.TEXT
            
        if mode_text:
            # Draw compact indicator at bottom of left panel
            mode_surface = self.font_small.render(mode_text, True, mode_color)
            mode_width = self.left_panel_width - 20
            mode_height = int(self.window_height * 0.033)  # Scale with window
            
            # Position at very bottom of left panel
            mode_y = self.window_height - mode_height - 10
            mode_bg_rect = pygame.Rect(10, mode_y, mode_width, mode_height)
            pygame.draw.rect(self.screen, (250, 250, 250), mode_bg_rect)
            pygame.draw.rect(self.screen, mode_color, mode_bg_rect, 2)
            
            # Draw text centered in the box
            text_rect = mode_surface.get_rect(center=mode_bg_rect.center)
            self.screen.blit(mode_surface, text_rect)
        
    def draw_top_bar(self):
        """Draw top control bar"""
        bar_rect = pygame.Rect(self.left_panel_width, 0, 
                              self.window_width - self.left_panel_width - self.right_panel_width,
                              self.top_bar_height)
        pygame.draw.rect(self.screen, Colors.PANEL_BG, bar_rect)
        pygame.draw.rect(self.screen, Colors.TEXT, bar_rect, 1)  # Thin border
        
        # Current scenario - place at top, but check for space
        if self.current_scenario < len(self.scenario_manager.scenarios):
            name, _ = self.scenario_manager.scenarios[self.current_scenario]
            scenario_text = self.font_small.render(f"Scenario: {name}", True, Colors.TEXT)
            # Place at top left of bar, small text
            text_y = 5
            text_x = self.left_panel_width + 10
            self.screen.blit(scenario_text, (text_x, text_y))
            
    def draw_right_panel(self):
        """Draw right panel with metrics"""
        panel_x = self.window_width - self.right_panel_width
        panel_rect = pygame.Rect(panel_x, 0, self.right_panel_width, self.window_height)
        pygame.draw.rect(self.screen, Colors.PANEL_BG, panel_rect)
        pygame.draw.rect(self.screen, Colors.TEXT, panel_rect, 1)  # Thin border
        
        # Title
        title = self.font_medium.render("Algorithm Results", True, Colors.TEXT)
        self.screen.blit(title, (panel_x + 10, 10))
        
        # Results for each algorithm
        y_offset = int(self.window_height * 0.055)  # Scale initial offset
        
        for algo_name, algo in self.algorithms.items():
            # Algorithm name with color
            color = {'astar': Colors.PATH_ASTAR, 
                    'dijkstra': Colors.PATH_DIJKSTRA,
                    'greedy': Colors.PATH_GREEDY}[algo_name]
            
            name_text = self.font_medium.render(algo.name, True, color)
            self.screen.blit(name_text, (panel_x + 10, y_offset))
            y_offset += int(self.font_medium.get_height() * 1.5)  # Scale spacing with font
            
            # Metrics
            if algo.path:
                metrics = [
                    f"Path cost: {algo.path_cost:.1f} minutes",
                    f"Nodes explored: {algo.nodes_explored:,}",
                    f"Grid coverage: {algo.nodes_explored / (self.grid.width * self.grid.height) * 100:.1f}%",
                    f"Search time: {algo.search_time * 1000:.1f}ms"
                ]
            else:
                metrics = ["No path found!"]
                
            for metric in metrics:
                text = self.font_small.render(metric, True, Colors.TEXT)
                self.screen.blit(text, (panel_x + 20, y_offset))
                # Scale line spacing with font size
                y_offset += int(self.font_small.get_height() * 1.3)  # 30% spacing between lines
                
            y_offset += int(self.font_small.get_height() * 0.8)  # Extra space before next section
            
        # Animation info
        y_offset += int(self.font_medium.get_height())  # Scale spacing
        anim_text = self.font_medium.render("Animation", True, Colors.TEXT)
        self.screen.blit(anim_text, (panel_x + 10, y_offset))
        y_offset += int(self.font_medium.get_height() * 1.5)  # Proportional spacing
        
        if self.animating:
            status = "Paused" if self.paused else "Playing"
        else:
            status = "Complete"
            
        status_text = self.font_small.render(f"Status: {status}", True, Colors.TEXT)
        self.screen.blit(status_text, (panel_x + 20, y_offset))
        y_offset += int(self.font_small.get_height() * 1.3)  # Scale line spacing
        
        # Display speed in a more readable format
        if self.animation_speed < 1:
            speed_display = f"Speed: {self.animation_speed:.1f} (very slow)"
        elif self.animation_speed <= 5:
            speed_display = f"Speed: {self.animation_speed} (slow)"
        elif self.animation_speed <= 20:
            speed_display = f"Speed: {self.animation_speed} (medium)"
        else:
            speed_display = f"Speed: {self.animation_speed} (fast)"
        speed_text = self.font_small.render(speed_display, True, Colors.TEXT)
        self.screen.blit(speed_text, (panel_x + 20, y_offset))
        
        # Add Individual View button at bottom of right panel
        button_width = int(self.right_panel_width * 0.8)
        button_height = int(self.window_height * 0.04)
        button_x = panel_x + (self.right_panel_width - button_width) // 2
        button_y = self.window_height - button_height - 20
        
        # Draw button
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        button_color = Colors.BUTTON_TOGGLED if self.individual_view_mode else Colors.BUTTON_NORMAL
        border_color = Colors.SELECTED_BORDER if self.individual_view_mode else Colors.BUTTON_BORDER
        
        pygame.draw.rect(self.screen, button_color, button_rect)
        pygame.draw.rect(self.screen, border_color, button_rect, 2)
        
        # Button text
        button_text = "Exit Individual View" if self.individual_view_mode else "Individual View"
        text_color = (255, 255, 255) if self.individual_view_mode else Colors.TEXT
        text_surface = self.font_small.render(button_text, True, text_color)
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        # Store button rect for click detection
        self.individual_view_btn_rect = button_rect
        
    def draw_bottom_bar(self):
        """Draw bottom instruction bar"""
        bar_y = self.window_height - self.bottom_bar_height
        bar_rect = pygame.Rect(self.left_panel_width, bar_y,
                              self.window_width - self.left_panel_width - self.right_panel_width,
                              self.bottom_bar_height)
        pygame.draw.rect(self.screen, Colors.PANEL_BG, bar_rect)
        pygame.draw.rect(self.screen, Colors.TEXT, bar_rect, 1)  # Thin border
        
        # Check if in step mode
        if hasattr(self, 'step_mode_btn') and self.step_mode_btn.toggled and self.educational_mode.enabled:
            # Draw step information - move to top of bottom bar
            step_info_y = bar_y + 10
            step_text = f"Step {self.educational_mode.current_step + 1} / {self.educational_mode.max_steps}"
            step_surface = self.font_medium.render(step_text, True, Colors.TEXT)
            step_x = self.left_panel_width + 50
            self.screen.blit(step_surface, (step_x, step_info_y))
            
            # Draw explanation text - below step info
            if self.educational_mode.current_step < 5:
                explanation = "Algorithms start at the green circle and explore toward the red goal"
            elif self.educational_mode.current_step < self.educational_mode.max_steps // 2:
                explanation = "Watch how each algorithm explores differently"
            else:
                explanation = "A* heads toward goal, Dijkstra spreads evenly, Greedy follows heuristic"
            
            exp_surface = self.font_small.render(explanation, True, Colors.TEXT)
            exp_x = self.left_panel_width + (self.window_width - self.left_panel_width - self.right_panel_width) // 2
            exp_x -= exp_surface.get_width() // 2
            self.screen.blit(exp_surface, (exp_x, bar_y + 35))
        else:
            # Normal instructions
            instructions = [
                "Mouse: Click to place roads, drag to draw continuously",
                "Keyboard: +/- for animation speed, SPACE to pause, 0 to reset speed",
                "Speed range: 0.5 (very slow) to 200 (very fast)"
            ]
            
            y = bar_y + int(self.window_height * 0.012)  # Scale initial offset
            for inst in instructions:
                text = self.font_small.render(inst, True, Colors.TEXT)
                self.screen.blit(text, (self.left_panel_width + 10, y))
                y += int(self.font_small.get_height() * 1.4)  # Scale line spacing with font
            
    def draw_pattern(self, rect, road_type):
        """Draw patterns on road types for better visibility"""
        import math
        
        if road_type == RoadType.HIGHWAY:
            # Dashed lines pattern
            for x in range(rect.left, rect.right, 8):
                pygame.draw.line(self.screen, (255, 255, 255), 
                               (x, rect.centery), (min(x + 4, rect.right), rect.centery), 1)
                               
        elif road_type == RoadType.RESIDENTIAL:
            # Dot pattern - more visible
            for y in range(rect.top + 3, rect.bottom, 6):
                for x in range(rect.left + 3, rect.right, 6):
                    pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 2)
                    
        elif road_type == RoadType.EMERGENCY_LANE:
            # Plus/Medical cross pattern
            cross_width = 3
            pygame.draw.rect(self.screen, (255, 255, 255),
                           (rect.left + rect.width//4, rect.centery - cross_width//2,
                            rect.width//2, cross_width))
            pygame.draw.rect(self.screen, (255, 255, 255),
                           (rect.centerx - cross_width//2, rect.top + rect.height//4,
                            cross_width, rect.height//2))
                            
        elif road_type == RoadType.HEAVY_TRAFFIC:
            # Horizontal stripes - clear pattern
            for y in range(rect.top, rect.bottom, 6):
                pygame.draw.rect(self.screen, (180, 50, 0), 
                               (rect.left, y, rect.width, 2))
                               
        elif road_type == RoadType.ACCIDENT:
            # X cross pattern - extra visible
            # Draw thicker X with yellow for warning
            pygame.draw.line(self.screen, (255, 255, 0), 
                           rect.topleft, rect.bottomright, 3)
            pygame.draw.line(self.screen, (255, 255, 0), 
                           rect.topright, rect.bottomleft, 3)
            # Add center highlight for small cells
            if rect.width > 10:
                pygame.draw.circle(self.screen, (255, 255, 0), rect.center, 2)
                           
        elif road_type == RoadType.CONSTRUCTION:
            # Diagonal warning stripes - yellow and black
            for i in range(-rect.height, rect.width + rect.height, 8):
                points = [(rect.left + i, rect.bottom),
                         (rect.left + i + 4, rect.bottom),
                         (rect.left + i + rect.height + 4, rect.top),
                         (rect.left + i + rect.height, rect.top)]
                clipped = []
                for px, py in points:
                    px = max(rect.left, min(rect.right, px))
                    py = max(rect.top, min(rect.bottom, py))
                    clipped.append((px, py))
                if len(clipped) >= 3:
                    pygame.draw.polygon(self.screen, (255, 255, 0), clipped)
                    
        elif road_type == RoadType.SCHOOL_ZONE:
            # Checkerboard pattern with white
            square_size = max(rect.width // 3, 4)
            for row in range(3):
                for col in range(3):
                    if (row + col) % 2 == 0:
                        pygame.draw.rect(self.screen, (255, 255, 255),
                                       (rect.left + col * square_size,
                                        rect.top + row * square_size,
                                        square_size, square_size))
                                        
        elif road_type == RoadType.FLOODED:
            # Wave pattern - darker blue waves
            for y in range(rect.top, rect.bottom, 4):
                points = []
                for x in range(rect.left, rect.right + 1):
                    wave_y = y + int(2 * math.sin((x - rect.left) * 0.5))
                    if rect.top <= wave_y <= rect.bottom:
                        points.append((x, wave_y))
                if len(points) > 1:
                    pygame.draw.lines(self.screen, (0, 100, 180), False, points, 2)
                    
        elif road_type == RoadType.TUNNEL:
            # Vertical stripes - lighter purple for visibility
            for x in range(rect.left, rect.right, 6):
                pygame.draw.rect(self.screen, (150, 100, 200), 
                               (x, rect.top, 2, rect.height))
                               
        elif road_type == RoadType.BRIDGE:
            # Grid/plank pattern - darker lines for visibility
            for x in range(rect.left, rect.right, 6):
                pygame.draw.line(self.screen, (100, 70, 0), 
                               (x, rect.top), (x, rect.bottom), 1)
            for y in range(rect.top, rect.bottom, 4):
                pygame.draw.line(self.screen, (100, 70, 0), 
                               (rect.left, y), (rect.right, y), 1)
                               
        elif road_type == RoadType.PARKING_LOT:
            # Grid pattern - parking spaces
            for x in range(rect.left, rect.right, 8):
                pygame.draw.line(self.screen, (120, 120, 120), 
                               (x, rect.top), (x, rect.bottom), 1)
            for y in range(rect.top, rect.bottom, 8):
                pygame.draw.line(self.screen, (120, 120, 120), 
                               (rect.left, y), (rect.right, y), 1)
                               
        elif road_type == RoadType.ONE_WAY:
            # Arrow pattern
            if rect.width > 10 and rect.height > 10:
                arrow_points = [(rect.centerx - 5, rect.centery - 3),
                              (rect.centerx + 3, rect.centery),
                              (rect.centerx - 5, rect.centery + 3)]
                pygame.draw.lines(self.screen, (255, 255, 255), False, arrow_points, 2)
                
        elif road_type == RoadType.MAZE_WALL:
            # Brick pattern - darker lines for visibility
            brick_height = 4
            brick_width = 8
            for row in range(0, rect.height, brick_height):
                offset = (brick_width // 2) if (row // brick_height) % 2 else 0
                for col in range(-offset, rect.width, brick_width):
                    brick_rect = pygame.Rect(rect.left + col, rect.top + row, 
                                           brick_width - 1, brick_height - 1)
                    if rect.left <= brick_rect.left < rect.right:
                        pygame.draw.rect(self.screen, (40, 140, 120), brick_rect, 1)
    
    def draw_grid(self):
        """Draw the city grid"""
        # First fill the entire grid area with background
        grid_area_rect = pygame.Rect(
            self.left_panel_width,
            self.top_bar_height,
            self.window_width - self.left_panel_width - self.right_panel_width,
            self.window_height - self.top_bar_height - self.bottom_bar_height
        )
        pygame.draw.rect(self.screen, Colors.PANEL_BG, grid_area_rect)
        
        # Draw cells
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Draw road type
                road_type = self.grid.grid[y][x]
                pygame.draw.rect(self.screen, road_type.color, rect)
                
                # Draw icon if cell is large enough
                if self.cell_size > 20 and road_type.icon:
                    icon_text = self.font_tiny.render(road_type.icon, True, Colors.TEXT)
                    icon_rect = icon_text.get_rect(center=rect.center)
                    self.screen.blit(icon_text, icon_rect)
                    
                # Draw grid lines if cells are large enough
                if self.cell_size > 10:
                    pygame.draw.rect(self.screen, Colors.GRID_LINE, rect, 1)
                    
        # Draw exploration based on mode
        if hasattr(self, 'step_mode_btn') and self.step_mode_btn.toggled and self.educational_mode.enabled:
            # Educational step-by-step mode
            self.draw_educational_exploration()
            self.draw_educational_paths()
        else:
            # Normal animated mode
            if self.show_exploration_btn.toggled:
                self.draw_exploration()
                
            # Draw paths
            if self.show_paths_btn.toggled:
                self.draw_paths()
            
        # Draw start and goal
        self.draw_start_goal()
        
    def draw_exploration(self):
        """Draw algorithm exploration"""
        for algo_name, algo in self.algorithms.items():
            if algo_name == 'astar':
                color = Colors.EXPLORED_ASTAR
            elif algo_name == 'dijkstra':
                color = Colors.EXPLORED_DIJKSTRA
            else:
                color = Colors.EXPLORED_GREEDY
                
            # Draw explored nodes up to current frame
            for node in algo.explored[:self.animation_frame]:
                x, y = node
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Create transparent surface
                s = pygame.Surface((self.cell_size, self.cell_size))
                s.set_alpha(50)
                s.fill(color[:3])
                self.screen.blit(s, rect)
                
    def draw_paths(self):
        """Draw the found paths"""
        # Draw paths with offsets to avoid overlap
        offsets = {'astar': -2, 'dijkstra': 0, 'greedy': 2}
        colors = {'astar': Colors.PATH_ASTAR, 
                 'dijkstra': Colors.PATH_DIJKSTRA,
                 'greedy': Colors.PATH_GREEDY}
        
        for algo_name, algo in self.algorithms.items():
            if algo.path and len(algo.path) > 1:
                points = []
                offset = offsets[algo_name]
                
                for x, y in algo.path:
                    center_x = self.grid_offset_x + x * self.cell_size + self.cell_size // 2 + offset
                    center_y = self.grid_offset_y + y * self.cell_size + self.cell_size // 2
                    points.append((center_x, center_y))
                    
                pygame.draw.lines(self.screen, colors[algo_name], False, points, 3)
                
    def draw_educational_exploration(self):
        """Draw exploration for educational mode"""
        colors = {
            'astar': Colors.EXPLORED_ASTAR,
            'dijkstra': Colors.EXPLORED_DIJKSTRA,
            'greedy': Colors.EXPLORED_GREEDY
        }
        
        # Only draw if we've started stepping (not at step 0)
        if self.educational_mode.current_step == 0:
            return  # Start with blank grid
            
        for algo_name in ['astar', 'dijkstra', 'greedy']:
            state = self.educational_mode.get_current_state(algo_name)
            if state:
                # Draw explored nodes
                for node in state['explored']:
                    x, y = node
                    rect = pygame.Rect(
                        self.grid_offset_x + x * self.cell_size,
                        self.grid_offset_y + y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    
                    s = pygame.Surface((self.cell_size, self.cell_size))
                    s.set_alpha(50)
                    s.fill(colors[algo_name][:3])
                    self.screen.blit(s, rect)
                    
                # Don't draw frontier nodes - only show actually explored nodes
                # Frontier nodes are just candidates, not actually explored
                pass
                    
    def draw_educational_paths(self):
        """Draw paths for educational mode - only when goal is reached"""
        colors = {
            'astar': Colors.PATH_ASTAR,
            'dijkstra': Colors.PATH_DIJKSTRA,
            'greedy': Colors.PATH_GREEDY
        }
        offsets = {'astar': -2, 'dijkstra': 0, 'greedy': 2}
        
        # Only draw if we've started stepping
        if self.educational_mode.current_step == 0:
            return
            
        for algo_name, algo in self.algorithms.items():
            state = self.educational_mode.get_current_state(algo_name)
            
            # Only draw path if the algorithm has found the goal
            if state and algo.path and len(algo.path) > 1:
                # Check if we've reached the step where the goal is found
                if self.grid.goal in state['explored']:
                    points = []
                    offset = offsets[algo_name]
                    
                    for x, y in algo.path:
                        center_x = self.grid_offset_x + x * self.cell_size + self.cell_size // 2 + offset
                        center_y = self.grid_offset_y + y * self.cell_size + self.cell_size // 2
                        points.append((center_x, center_y))
                        
                    pygame.draw.lines(self.screen, colors[algo_name], False, points, 3)
                    
            # Draw current node being explored with a special marker
            if state and state['node']:
                x, y = state['node']
                center_x = self.grid_offset_x + x * self.cell_size + self.cell_size // 2
                center_y = self.grid_offset_y + y * self.cell_size + self.cell_size // 2
                pygame.draw.circle(self.screen, colors[algo_name], (center_x, center_y), 5)
                pygame.draw.circle(self.screen, Colors.TEXT, (center_x, center_y), 5, 2)
    
    def draw_individual_view(self):
        """Draw side-by-side individual algorithm views with individual controls"""
        # Minimal header with just back button
        title_height = int(self.window_height * 0.05)
        
        # Back button in top left corner
        exit_btn_width = int(self.window_width * 0.06)
        exit_btn_height = int(title_height * 0.8)
        exit_btn_x = 10
        exit_btn_y = (title_height - exit_btn_height) // 2
        
        exit_rect = pygame.Rect(exit_btn_x, exit_btn_y, exit_btn_width, exit_btn_height)
        pygame.draw.rect(self.screen, Colors.BUTTON_NORMAL, exit_rect)
        pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, exit_rect, 2)
        
        exit_text = self.font_small.render("Back", True, Colors.TEXT)
        exit_text_rect = exit_text.get_rect(center=exit_rect.center)
        self.screen.blit(exit_text, exit_text_rect)
        
        # Store for click detection
        self.individual_exit_btn_rect = exit_rect
        
        # Store button rects for each algorithm
        self.individual_control_buttons = {
            'astar': {},
            'dijkstra': {},
            'greedy': {}
        }
        
        # Calculate panel dimensions (3 panels side by side)
        panel_width = self.window_width // 3
        panel_height = self.window_height - title_height
        panel_y = title_height
        
        # Draw three algorithm panels
        algorithms = [
            ('astar', Colors.PATH_ASTAR, 0),
            ('dijkstra', Colors.PATH_DIJKSTRA, panel_width),
            ('greedy', Colors.PATH_GREEDY, panel_width * 2)
        ]
        
        for algo_name, color, panel_x in algorithms:
            # Panel background
            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            pygame.draw.rect(self.screen, Colors.PANEL_BG, panel_rect)
            
            # Set clipping to panel area to prevent text overflow
            self.screen.set_clip(panel_rect)
            
            if algo_name in self.algorithms:
                algo = self.algorithms[algo_name]
                
                # Algorithm title - ensure it fits
                title_text = self.font_medium.render(algo.name, True, color)
                # Check if title fits, if not, use smaller font
                if title_text.get_width() > panel_width - 20:
                    title_text = self.font_small.render(algo.name, True, color)
                self.screen.blit(title_text, (panel_x + 10, panel_y + 10))
                
                # Grid area
                grid_margin = 20
                grid_top_offset = 50
                grid_rect = pygame.Rect(
                    panel_x + grid_margin,
                    panel_y + grid_top_offset,
                    panel_width - (grid_margin * 2),
                    panel_height - grid_top_offset - 120
                )
                
                # Fill grid background first (like main grid)
                pygame.draw.rect(self.screen, Colors.BACKGROUND, grid_rect)
                
                # Calculate cell size for individual grids
                cell_size = min(
                    grid_rect.width // self.grid.width,
                    grid_rect.height // self.grid.height
                )
                
                # Center the grid
                grid_offset_x = grid_rect.x + (grid_rect.width - cell_size * self.grid.width) // 2
                grid_offset_y = grid_rect.y + (grid_rect.height - cell_size * self.grid.height) // 2
                
                # Draw actual grid area background
                actual_grid_rect = pygame.Rect(
                    grid_offset_x,
                    grid_offset_y,
                    cell_size * self.grid.width,
                    cell_size * self.grid.height
                )
                pygame.draw.rect(self.screen, Colors.BACKGROUND, actual_grid_rect)
                
                # Draw grid cells (exactly like main grid)
                for y in range(self.grid.height):
                    for x in range(self.grid.width):
                        rect = pygame.Rect(
                            grid_offset_x + x * cell_size,
                            grid_offset_y + y * cell_size,
                            cell_size,
                            cell_size
                        )
                        
                        # Draw road type color
                        road_type = self.grid.grid[y][x]
                        pygame.draw.rect(self.screen, road_type.color, rect)
                        
                        # Always draw grid lines for structure (like main grid)
                        pygame.draw.rect(self.screen, Colors.GRID_LINE, rect, 1)
                        
                        # Draw road type icon if cell is large enough
                        if cell_size > 20 and road_type.icon:
                            icon_text = self.font_tiny.render(road_type.icon, True, Colors.TEXT)
                            icon_rect = icon_text.get_rect(center=rect.center)
                            self.screen.blit(icon_text, icon_rect)
                
                # Draw exploration for this algorithm
                # For any algorithm, use step data if in step mode, or animation frame if animating
                algo_step_mode = getattr(self, f'{algo_name}_step_mode', False)
                if algo_step_mode:
                    # Use step data for current algorithm
                    if algo_name == 'astar':
                        exp_color = Colors.EXPLORED_ASTAR
                    elif algo_name == 'dijkstra':
                        exp_color = Colors.EXPLORED_DIJKSTRA
                    else:  # greedy
                        exp_color = Colors.EXPLORED_GREEDY
                    
                    if hasattr(self, 'educational_mode') and self.educational_mode.step_data.get(algo_name):
                        # Get exploration up to current step
                        algo_current_step = getattr(self, f'{algo_name}_current_step', 0)
                        if algo_current_step > 0 and algo_current_step <= len(self.educational_mode.step_data[algo_name]):
                            step_info = self.educational_mode.step_data[algo_name][algo_current_step - 1]
                            explored_nodes = step_info.get('explored', set())
                            
                            for node in explored_nodes:
                                x, y = node
                                rect = pygame.Rect(
                                    grid_offset_x + x * cell_size,
                                    grid_offset_y + y * cell_size,
                                    cell_size,
                                    cell_size
                                )
                                # Use proper transparency
                                s = pygame.Surface((cell_size, cell_size))
                                s.set_alpha(50)
                                s.fill(exp_color[:3] if len(exp_color) >= 3 else exp_color)
                                self.screen.blit(s, rect)
                            
                            # Draw current node being explored with highlight
                            if step_info.get('node'):
                                x, y = step_info['node']
                                center_x = grid_offset_x + x * cell_size + cell_size // 2
                                center_y = grid_offset_y + y * cell_size + cell_size // 2
                                pygame.draw.circle(self.screen, Colors.PATH_ASTAR, (center_x, center_y), 5)
                                pygame.draw.circle(self.screen, Colors.TEXT, (center_x, center_y), 5, 2)
                
                elif algo.explored:
                    # Normal exploration display - for A* check if animating
                    if algo_name == 'astar':
                        exp_color = Colors.EXPLORED_ASTAR
                        # If animating, show exploration based on animation frame (like step mode)
                        if hasattr(self, 'astar_animating') and (self.astar_animating or self.astar_animation_frame > 0):
                            # Use step data for animation (exactly like step mode)
                            if hasattr(self, 'educational_mode') and self.educational_mode.step_data.get('astar'):
                                step_data = self.educational_mode.step_data['astar']
                                if self.astar_animation_frame > 0 and self.astar_animation_frame <= len(step_data):
                                    step_index = self.astar_animation_frame - 1
                                    step_info = step_data[step_index]
                                    
                                    # Draw explored nodes
                                    for node in step_info.get('explored', set()):
                                        x, y = node
                                        rect = pygame.Rect(
                                            grid_offset_x + x * cell_size,
                                            grid_offset_y + y * cell_size,
                                            cell_size,
                                            cell_size
                                        )
                                        s = pygame.Surface((cell_size, cell_size))
                                        s.set_alpha(50)
                                        s.fill(exp_color[:3])
                                        self.screen.blit(s, rect)
                                    
                                    # Draw current node highlight (like step mode)
                                    if step_info.get('node'):
                                        x, y = step_info['node']
                                        center_x = grid_offset_x + x * cell_size + cell_size // 2
                                        center_y = grid_offset_y + y * cell_size + cell_size // 2
                                        pygame.draw.circle(self.screen, Colors.PATH_ASTAR, (center_x, center_y), 5)
                                        pygame.draw.circle(self.screen, Colors.TEXT, (center_x, center_y), 5, 2)
                        else:
                            # Not animating - show full exploration
                            for node in algo.explored:
                                x, y = node
                                rect = pygame.Rect(
                                    grid_offset_x + x * cell_size,
                                    grid_offset_y + y * cell_size,
                                    cell_size,
                                    cell_size
                                )
                                s = pygame.Surface((cell_size, cell_size))
                                s.set_alpha(50)
                                s.fill(exp_color[:3])
                                self.screen.blit(s, rect)
                    else:
                        # Other algorithms - check if they are animating too
                        if algo_name == 'dijkstra':
                            exp_color = Colors.EXPLORED_DIJKSTRA
                            algo_animating = getattr(self, 'dijkstra_animating', False)
                            algo_animation_frame = getattr(self, 'dijkstra_animation_frame', 0)
                        else:  # greedy
                            exp_color = Colors.EXPLORED_GREEDY
                            algo_animating = getattr(self, 'greedy_animating', False)
                            algo_animation_frame = getattr(self, 'greedy_animation_frame', 0)
                        
                        # If animating, show step-by-step like A*
                        if algo_animating or algo_animation_frame > 0:
                            if hasattr(self, 'educational_mode') and self.educational_mode.step_data.get(algo_name):
                                step_data = self.educational_mode.step_data[algo_name]
                                if algo_animation_frame > 0 and algo_animation_frame <= len(step_data):
                                    step_index = algo_animation_frame - 1
                                    step_info = step_data[step_index]
                                    
                                    # Draw explored nodes up to current step
                                    for node in step_info.get('explored', set()):
                                        x, y = node
                                        rect = pygame.Rect(
                                            grid_offset_x + x * cell_size,
                                            grid_offset_y + y * cell_size,
                                            cell_size,
                                            cell_size
                                        )
                                        s = pygame.Surface((cell_size, cell_size))
                                        s.set_alpha(50)
                                        s.fill(exp_color[:3])
                                        self.screen.blit(s, rect)
                                    
                                    # Draw current exploration node (leading circle)
                                    if step_info.get('node'):
                                        x, y = step_info['node']
                                        center_x = grid_offset_x + x * cell_size + cell_size // 2
                                        center_y = grid_offset_y + y * cell_size + cell_size // 2
                                        
                                        # Use algorithm-specific colors
                                        if algo_name == 'dijkstra':
                                            circle_color = Colors.PATH_DIJKSTRA
                                        else:
                                            circle_color = Colors.PATH_GREEDY
                                        
                                        pygame.draw.circle(self.screen, circle_color, (center_x, center_y), 5)
                                        pygame.draw.circle(self.screen, Colors.TEXT, (center_x, center_y), 5, 2)
                        else:
                            # Not animating - show full exploration
                            for node in algo.explored:
                                x, y = node
                                rect = pygame.Rect(
                                    grid_offset_x + x * cell_size,
                                    grid_offset_y + y * cell_size,
                                    cell_size,
                                    cell_size
                                )
                                s = pygame.Surface((cell_size, cell_size))
                                s.set_alpha(50)
                                s.fill(exp_color[:3] if len(exp_color) >= 3 else exp_color)
                                self.screen.blit(s, rect)
                
                # Draw path for this algorithm
                # For any algorithm in step mode, show partial path if available
                if algo_step_mode:
                    if hasattr(self, 'educational_mode') and self.educational_mode.step_data.get(algo_name):
                        if algo_current_step > 0 and algo_current_step <= len(self.educational_mode.step_data[algo_name]):
                            step_info = self.educational_mode.step_data[algo_name][algo_current_step - 1]
                            partial_path = step_info.get('path_so_far', [])
                            
                            if partial_path and len(partial_path) > 1:
                                points = []
                                for x, y in partial_path:
                                    center_x = grid_offset_x + x * cell_size + cell_size // 2
                                    center_y = grid_offset_y + y * cell_size + cell_size // 2
                                    points.append((center_x, center_y))
                                
                                if len(points) > 1:
                                    pygame.draw.lines(self.screen, color, False, points, 3)
                else:
                    # Normal path display - but during animation, use step data for progressive path
                    algo_animating = getattr(self, f'{algo_name}_animating', False)
                    algo_animation_frame = getattr(self, f'{algo_name}_animation_frame', 0)
                    if algo_animating or algo_animation_frame > 0:
                        # During animation, show the partial path from step data
                        if hasattr(self, 'educational_mode') and self.educational_mode.step_data.get(algo_name):
                            step_data = self.educational_mode.step_data[algo_name]
                            if step_data and algo_animation_frame > 0:
                                # Map animation frame to step index
                                # Each step corresponds to one explored node
                                step_index = min(algo_animation_frame - 1, len(step_data) - 1)
                                if step_index >= 0:
                                    step_info = step_data[step_index]
                                    partial_path = step_info.get('path_so_far', [])
                                    
                                    # Draw the partial path
                                    if partial_path and len(partial_path) > 1:
                                        points = []
                                        for x, y in partial_path:
                                            center_x = grid_offset_x + x * cell_size + cell_size // 2
                                            center_y = grid_offset_y + y * cell_size + cell_size // 2
                                            points.append((center_x, center_y))
                                        
                                        if len(points) > 1:
                                            pygame.draw.lines(self.screen, color, False, points, 3)
                    else:
                        # Normal path display for other algorithms or when not animating
                        if algo.path and len(algo.path) > 1:
                            points = []
                            for x, y in algo.path:
                                center_x = grid_offset_x + x * cell_size + cell_size // 2
                                center_y = grid_offset_y + y * cell_size + cell_size // 2
                                points.append((center_x, center_y))
                            
                            if len(points) > 1:
                                pygame.draw.lines(self.screen, color, False, points, 3)
                
                # Draw start and goal (matching main grid style)
                if self.grid.start:
                    start_x, start_y = self.grid.start
                    start_center = (
                        grid_offset_x + start_x * cell_size + cell_size // 2,
                        grid_offset_y + start_y * cell_size + cell_size // 2
                    )
                    radius = max(cell_size // 3, 3)
                    pygame.draw.circle(self.screen, Colors.START, start_center, radius)
                    pygame.draw.circle(self.screen, Colors.TEXT, start_center, radius, 2)
                    
                if self.grid.goal:
                    goal_x, goal_y = self.grid.goal
                    goal_center = (
                        grid_offset_x + goal_x * cell_size + cell_size // 2,
                        grid_offset_y + goal_y * cell_size + cell_size // 2
                    )
                    radius = max(cell_size // 3, 3)
                    pygame.draw.circle(self.screen, Colors.GOAL, goal_center, radius)
                    pygame.draw.circle(self.screen, Colors.TEXT, goal_center, radius, 2)
                
                # Add step controls for all algorithms 
                if algo_name in ['astar', 'dijkstra', 'greedy']:
                    # Control area at bottom of panel
                    control_y = panel_y + panel_height - 120
                    
                    # Initialize animation states if not exists - for each algorithm
                    for alg in ['astar', 'dijkstra', 'greedy']:
                        if not hasattr(self, f'{alg}_step_mode'):
                            setattr(self, f'{alg}_step_mode', False)
                            setattr(self, f'{alg}_current_step', 0)
                        if not hasattr(self, f'{alg}_animating'):
                            setattr(self, f'{alg}_animating', False)
                            setattr(self, f'{alg}_paused', False)
                            setattr(self, f'{alg}_animation_frame', 0)
                            setattr(self, f'{alg}_animation_speed', 10)
                    
                    # Step Mode toggle button for current algorithm
                    btn_width = 80
                    btn_height = 25
                    btn_x = panel_x + (panel_width - btn_width) // 2
                    btn_y = control_y
                    
                    # Draw Step Mode button
                    step_btn_rect = pygame.Rect(btn_x, btn_y, btn_width, btn_height)
                    step_mode_active = getattr(self, f'{algo_name}_step_mode', False)
                    btn_color = Colors.BUTTON_TOGGLED if step_mode_active else Colors.BUTTON_NORMAL
                    pygame.draw.rect(self.screen, btn_color, step_btn_rect)
                    pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, step_btn_rect, 2)
                    
                    step_text = self.font_tiny.render("Step Mode", True, Colors.TEXT)
                    step_text_rect = step_text.get_rect(center=step_btn_rect.center)
                    self.screen.blit(step_text, step_text_rect)
                    
                    # Store button for click detection
                    self.individual_control_buttons[algo_name]['step_mode'] = step_btn_rect
                    
                    # If step mode is active, show step controls
                    if step_mode_active:
                        # Prev and Next buttons
                        control_btn_width = 50
                        control_btn_height = 20
                        control_y2 = control_y + 30
                        
                        # Prev button
                        prev_btn_x = panel_x + panel_width // 2 - control_btn_width - 5
                        prev_btn_rect = pygame.Rect(prev_btn_x, control_y2, control_btn_width, control_btn_height)
                        
                        # Check if we can go prev
                        current_step = getattr(self, f'{algo_name}_current_step', 0)
                        can_prev = current_step > 0
                        prev_color = Colors.BUTTON_NORMAL if can_prev else Colors.BUTTON_DISABLED
                        
                        pygame.draw.rect(self.screen, prev_color, prev_btn_rect)
                        pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, prev_btn_rect, 1)
                        
                        prev_text = self.font_tiny.render("< Prev", True, Colors.TEXT if can_prev else Colors.TEXT_DISABLED)
                        prev_text_rect = prev_text.get_rect(center=prev_btn_rect.center)
                        self.screen.blit(prev_text, prev_text_rect)
                        
                        self.individual_control_buttons[algo_name]['prev'] = prev_btn_rect if can_prev else None
                        
                        # Next button
                        next_btn_x = panel_x + panel_width // 2 + 5
                        next_btn_rect = pygame.Rect(next_btn_x, control_y2, control_btn_width, control_btn_height)
                        
                        # Check if we can go next
                        max_steps = len(self.educational_mode.step_data.get(algo_name, [])) if hasattr(self, 'educational_mode') else 0
                        can_next = current_step < max_steps
                        next_color = Colors.BUTTON_NORMAL if can_next else Colors.BUTTON_DISABLED
                        
                        pygame.draw.rect(self.screen, next_color, next_btn_rect)
                        pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, next_btn_rect, 1)
                        
                        next_text = self.font_tiny.render("Next >", True, Colors.TEXT if can_next else Colors.TEXT_DISABLED)
                        next_text_rect = next_text.get_rect(center=next_btn_rect.center)
                        self.screen.blit(next_text, next_text_rect)
                        
                        self.individual_control_buttons[algo_name]['next'] = next_btn_rect if can_next else None
                        
                        # Step counter
                        step_text = f"Step {current_step}/{max_steps}"
                        counter_text = self.font_tiny.render(step_text, True, Colors.TEXT)
                        counter_x = panel_x + (panel_width - counter_text.get_width()) // 2
                        self.screen.blit(counter_text, (counter_x, control_y2 + 25))
                        
                        # Metrics
                        metrics_y = control_y2 + 45
                    else:
                        # Normal mode - show animation controls
                        control_y2 = control_y + 30
                        
                        # Animation control buttons (Run/Pause/Restart)
                        btn_spacing = 5
                        anim_btn_width = 45
                        anim_btn_height = 20
                        
                        # Calculate starting x to center the button group
                        total_width = (anim_btn_width * 3) + (btn_spacing * 2)
                        start_x = panel_x + (panel_width - total_width) // 2
                        
                        # Run button
                        run_btn_x = start_x
                        run_btn_rect = pygame.Rect(run_btn_x, control_y2, anim_btn_width, anim_btn_height)
                        algo_animating = getattr(self, f'{algo_name}_animating', False)
                        algo_paused = getattr(self, f'{algo_name}_paused', False)
                        run_color = Colors.BUTTON_TOGGLED if algo_animating and not algo_paused else Colors.BUTTON_NORMAL
                        pygame.draw.rect(self.screen, run_color, run_btn_rect)
                        pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, run_btn_rect, 1)
                        
                        run_text = self.font_tiny.render("Run", True, Colors.TEXT)
                        run_text_rect = run_text.get_rect(center=run_btn_rect.center)
                        self.screen.blit(run_text, run_text_rect)
                        self.individual_control_buttons[algo_name]['run'] = run_btn_rect
                        
                        # Pause button
                        pause_btn_x = start_x + anim_btn_width + btn_spacing
                        pause_btn_rect = pygame.Rect(pause_btn_x, control_y2, anim_btn_width, anim_btn_height)
                        pause_color = Colors.BUTTON_TOGGLED if algo_paused else Colors.BUTTON_NORMAL
                        pygame.draw.rect(self.screen, pause_color, pause_btn_rect)
                        pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, pause_btn_rect, 1)
                        
                        pause_text = self.font_tiny.render("Pause", True, Colors.TEXT)
                        pause_text_rect = pause_text.get_rect(center=pause_btn_rect.center)
                        self.screen.blit(pause_text, pause_text_rect)
                        self.individual_control_buttons[algo_name]['pause'] = pause_btn_rect
                        
                        # Restart button
                        restart_btn_x = start_x + (anim_btn_width + btn_spacing) * 2
                        restart_btn_rect = pygame.Rect(restart_btn_x, control_y2, anim_btn_width, anim_btn_height)
                        pygame.draw.rect(self.screen, Colors.BUTTON_NORMAL, restart_btn_rect)
                        pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, restart_btn_rect, 1)
                        
                        restart_text = self.font_tiny.render("Restart", True, Colors.TEXT)
                        restart_text_rect = restart_text.get_rect(center=restart_btn_rect.center)
                        self.screen.blit(restart_text, restart_text_rect)
                        self.individual_control_buttons[algo_name]['restart'] = restart_btn_rect
                        
                        # Speed control
                        speed_y = control_y2 + 25
                        algo_speed = getattr(self, f'{algo_name}_animation_speed', 10)
                        speed_text = f"Speed: {algo_speed}"
                        speed_surface = self.font_tiny.render(speed_text, True, Colors.TEXT)
                        speed_x = panel_x + (panel_width - speed_surface.get_width()) // 2
                        self.screen.blit(speed_surface, (speed_x, speed_y))
                        
                        # Speed adjustment buttons
                        speed_btn_width = 20
                        speed_btn_height = 15
                        
                        # Decrease speed button
                        dec_speed_x = speed_x - 25
                        dec_speed_rect = pygame.Rect(dec_speed_x, speed_y, speed_btn_width, speed_btn_height)
                        pygame.draw.rect(self.screen, Colors.BUTTON_NORMAL, dec_speed_rect)
                        pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, dec_speed_rect, 1)
                        dec_text = self.font_tiny.render("-", True, Colors.TEXT)
                        dec_text_rect = dec_text.get_rect(center=dec_speed_rect.center)
                        self.screen.blit(dec_text, dec_text_rect)
                        self.individual_control_buttons[algo_name]['dec_speed'] = dec_speed_rect
                        
                        # Increase speed button
                        inc_speed_x = speed_x + speed_surface.get_width() + 5
                        inc_speed_rect = pygame.Rect(inc_speed_x, speed_y, speed_btn_width, speed_btn_height)
                        pygame.draw.rect(self.screen, Colors.BUTTON_NORMAL, inc_speed_rect)
                        pygame.draw.rect(self.screen, Colors.BUTTON_BORDER, inc_speed_rect, 1)
                        inc_text = self.font_tiny.render("+", True, Colors.TEXT)
                        inc_text_rect = inc_text.get_rect(center=inc_speed_rect.center)
                        self.screen.blit(inc_text, inc_text_rect)
                        self.individual_control_buttons[algo_name]['inc_speed'] = inc_speed_rect
                        
                        # Animation progress display
                        algo_animation_frame = getattr(self, f'{algo_name}_animation_frame', 0)
                        if algo_animating or algo_animation_frame > 0:
                            max_steps = len(self.educational_mode.step_data.get(algo_name, [])) if hasattr(self, 'educational_mode') else 0
                            progress_text = f"Step: {algo_animation_frame}/{max_steps}"
                            progress_surface = self.font_tiny.render(progress_text, True, Colors.TEXT)
                            progress_x = panel_x + (panel_width - progress_surface.get_width()) // 2
                            self.screen.blit(progress_surface, (progress_x, speed_y + 18))
                            metrics_y = speed_y + 35
                        else:
                            metrics_y = speed_y + 20
                    
                    # Always show metrics
                    metrics = [
                        f"Path: {algo.path_cost:.1f}min" if algo.path else "No path",
                        f"Nodes: {algo.nodes_explored}"
                    ]
                    
                    for i, metric in enumerate(metrics):
                        text = self.font_tiny.render(metric, True, Colors.TEXT)
                        # Ensure text fits within panel width
                        if text.get_width() > panel_width - 20:
                            # Truncate or use ellipsis if text is too long
                            metric = metric[:20] + "..." if len(metric) > 20 else metric
                            text = self.font_tiny.render(metric, True, Colors.TEXT)
                        self.screen.blit(text, (panel_x + 10, metrics_y + i * 15))
                    
                else:
                    # For other algorithms (Dijkstra and Greedy), just show metrics
                    control_y = panel_y + panel_height - 100
                    metrics = [
                        f"Path: {algo.path_cost:.1f}min" if algo.path else "No path",
                        f"Nodes: {algo.nodes_explored}"
                    ]
                    
                    for i, metric in enumerate(metrics):
                        text = self.font_tiny.render(metric, True, Colors.TEXT)
                        # Ensure text fits within panel width
                        if text.get_width() > panel_width - 20:
                            metric = metric[:20] + "..." if len(metric) > 20 else metric
                            text = self.font_tiny.render(metric, True, Colors.TEXT)
                        self.screen.blit(text, (panel_x + 10, control_y + i * 15))
        
        # Reset clipping after drawing all panels
        self.screen.set_clip(None)
    
    def draw_start_goal(self):
        """Draw start and goal positions"""
        # Start (green circle)
        start_x, start_y = self.grid.start
        start_rect = pygame.Rect(
            self.grid_offset_x + start_x * self.cell_size + 2,
            self.grid_offset_y + start_y * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.ellipse(self.screen, Colors.START, start_rect)
        pygame.draw.ellipse(self.screen, Colors.TEXT, start_rect, 2)
        
        # Goal (red circle)
        goal_x, goal_y = self.grid.goal
        goal_rect = pygame.Rect(
            self.grid_offset_x + goal_x * self.cell_size + 2,
            self.grid_offset_y + goal_y * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.ellipse(self.screen, Colors.GOAL, goal_rect)
        pygame.draw.ellipse(self.screen, Colors.TEXT, goal_rect, 2)
        
    def update(self):
        """Update animation state"""
        # Handle A* animation in Individual View (when not in step mode)
        if self.individual_view_mode and hasattr(self, 'astar_animating'):
            if self.astar_animating and not self.astar_paused and not self.astar_step_mode:
                # Auto-advance frames like step mode Run button
                if not hasattr(self, 'astar_animation_timer'):
                    self.astar_animation_timer = 0
                
                self.astar_animation_timer += 1
                
                # Speed control - higher speed = faster frames (same as dijkstra/greedy)
                frame_delay = max(1, 60 // self.astar_animation_speed)
                
                if self.astar_animation_timer >= frame_delay:
                    self.astar_animation_timer = 0
                    
                    if hasattr(self, 'educational_mode') and self.educational_mode.step_data.get('astar'):
                        max_frames = len(self.educational_mode.step_data['astar'])
                        if self.astar_animation_frame < max_frames:
                            self.astar_animation_frame += 1
                            if self.astar_animation_frame >= max_frames:
                                self.astar_animating = False
                                self.astar_paused = True
        
        # Handle Dijkstra animation in Individual View (when not in step mode)
        if self.individual_view_mode and hasattr(self, 'dijkstra_animating'):
            if self.dijkstra_animating and not self.dijkstra_paused and not self.dijkstra_step_mode:
                # Auto-advance frames like step mode Run button
                if not hasattr(self, 'dijkstra_animation_timer'):
                    self.dijkstra_animation_timer = 0
                
                self.dijkstra_animation_timer += 1
                
                # Speed control - higher speed = faster frames
                frame_delay = max(1, 60 // self.dijkstra_animation_speed)
                
                if self.dijkstra_animation_timer >= frame_delay:
                    self.dijkstra_animation_timer = 0
                    
                    if hasattr(self, 'educational_mode') and self.educational_mode.step_data.get('dijkstra'):
                        max_frames = len(self.educational_mode.step_data['dijkstra'])
                        if self.dijkstra_animation_frame < max_frames:
                            self.dijkstra_animation_frame += 1
                            if self.dijkstra_animation_frame >= max_frames:
                                self.dijkstra_animating = False
                                self.dijkstra_paused = True
        
        # Handle Greedy animation in Individual View (when not in step mode)
        if self.individual_view_mode and hasattr(self, 'greedy_animating'):
            if self.greedy_animating and not self.greedy_paused and not self.greedy_step_mode:
                # Auto-advance frames like step mode Run button
                if not hasattr(self, 'greedy_animation_timer'):
                    self.greedy_animation_timer = 0
                
                self.greedy_animation_timer += 1
                
                # Speed control - higher speed = faster frames
                frame_delay = max(1, 60 // self.greedy_animation_speed)
                
                if self.greedy_animation_timer >= frame_delay:
                    self.greedy_animation_timer = 0
                    
                    if hasattr(self, 'educational_mode') and self.educational_mode.step_data.get('greedy'):
                        max_frames = len(self.educational_mode.step_data['greedy'])
                        if self.greedy_animation_frame < max_frames:
                            self.greedy_animation_frame += 1
                            if self.greedy_animation_frame >= max_frames:
                                self.greedy_animating = False
                                self.greedy_paused = True
        
        # Handle individual view animations (old code for step mode)
        if self.individual_view_mode and self.educational_mode.enabled:
            for algo_name in ['astar', 'dijkstra', 'greedy']:
                if self.individual_animating[algo_name] and not self.individual_paused[algo_name]:
                    # Auto-advance steps for this algorithm
                    if not hasattr(self, 'individual_timers'):
                        self.individual_timers = {'astar': 0, 'dijkstra': 0, 'greedy': 0}
                    
                    self.individual_timers[algo_name] += self.individual_speeds[algo_name] * 0.5
                    if self.individual_timers[algo_name] >= 10:
                        self.individual_timers[algo_name] = 0
                        if self.individual_steps[algo_name] < self.educational_mode.max_steps - 1:
                            self.individual_steps[algo_name] += 1
                        else:
                            # Reached the end
                            self.individual_animating[algo_name] = False
                            self.individual_paused[algo_name] = False
        
        # Handle auto-stepping in educational mode (normal view)
        elif self.educational_mode.enabled and self.auto_stepping and not self.paused:
            # Auto-advance steps with timing based on animation speed
            self.auto_step_timer += self.animation_speed * 0.5  # Scale with animation speed
            if self.auto_step_timer >= 10:  # Threshold for stepping
                self.auto_step_timer = 0
                if self.educational_mode.current_step < self.educational_mode.max_steps - 1:
                    self.educational_mode.next_step()
                else:
                    # Reached the end, stop auto-stepping
                    self.auto_stepping = False
                    self.paused = True
        
        # Handle normal animation
        elif self.animating and not self.paused:
            max_exploration = max(len(algo.explored) for algo in self.algorithms.values())
            # Handle fractional speeds for very slow animation
            if self.animation_speed < 1:
                # For speeds less than 1, only advance occasionally
                self.animation_counter = getattr(self, 'animation_counter', 0) + self.animation_speed
                if self.animation_counter >= 1:
                    self.animation_frame = min(self.animation_frame + 1, max_exploration)
                    self.animation_counter -= 1
            else:
                self.animation_frame = min(self.animation_frame + int(self.animation_speed), max_exploration)
            
            if self.animation_frame >= max_exploration:
                self.animating = False
                
        # Update toggle button states
        self.show_exploration = self.show_exploration_btn.toggled
        self.show_paths = self.show_paths_btn.toggled
        
    def handle_event(self, event):
        """Handle events"""
        # Handle button events
        for btn in self.buttons:
            btn.handle_event(event)
        # Handle toggle buttons (but not Explore/Paths during step mode)
        for btn in self.toggle_buttons:
            if hasattr(self, 'step_mode_btn') and self.step_mode_btn.toggled:
                # Don't handle events for Explore/Paths during step mode
                if btn not in [self.show_exploration_btn, self.show_paths_btn]:
                    btn.handle_event(event)
            else:
                btn.handle_event(event)
            
        # Handle step control buttons if in step mode
        if hasattr(self, 'step_mode_btn') and self.step_mode_btn.toggled:
            for btn in self.step_buttons:
                btn.handle_event(event)
                
        for road_type, btn in self.road_type_buttons:
            btn.handle_event(event)
            
        # Handle keyboard
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.toggle_pause()
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                # Increase speed with better granularity
                if self.animation_speed < 5:
                    self.animation_speed += 1
                elif self.animation_speed < 20:
                    self.animation_speed += 5
                else:
                    self.animation_speed = min(self.animation_speed + 10, 200)
            elif event.key == pygame.K_MINUS:
                # Decrease speed with better granularity for slow speeds
                if self.animation_speed > 20:
                    self.animation_speed -= 10
                elif self.animation_speed > 5:
                    self.animation_speed -= 5
                elif self.animation_speed > 1:
                    self.animation_speed -= 1
                else:
                    self.animation_speed = 0.5  # Allow fractional speeds for very slow
            elif event.key == pygame.K_0:
                # Reset to default speed
                self.animation_speed = 10
            # Step mode keyboard shortcuts
            elif hasattr(self, 'step_mode_btn') and self.step_mode_btn.toggled:
                if event.key == pygame.K_LEFT:
                    self.prev_step()
                elif event.key == pygame.K_RIGHT:
                    self.next_step()
                elif event.key == pygame.K_HOME:
                    self.reset_steps()
                
        # Handle mouse on grid
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                # Check if clicking on grid
                if self.grid_area.collidepoint(event.pos):
                    self.mouse_drawing = True
                    self.handle_mouse_click(event.pos)
                    
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.mouse_drawing = False
                
                # Check if Individual View button was clicked
                if hasattr(self, 'individual_view_btn_rect'):
                    if self.individual_view_btn_rect.collidepoint(event.pos):
                        self.individual_view_mode = not self.individual_view_mode
                        
                        # If entering individual view
                        if self.individual_view_mode:
                            # Handle transition from main step mode if needed
                            if self.educational_mode.enabled:
                                # Exit main step mode
                                self.educational_mode.enabled = False
                            
                            # Capture step data for A* (only once)
                            self.educational_mode.enabled = True
                            self.educational_mode.reset()
                            self.run_algorithms()
                            self.educational_mode.update_max_steps()
                            self.educational_mode.enabled = False
                            
                            # All step modes should always start fresh (disabled) when entering Individual View
                            for alg in ['astar', 'dijkstra', 'greedy']:
                                if hasattr(self, f'{alg}_step_mode'):
                                    setattr(self, f'{alg}_step_mode', False)
                                    setattr(self, f'{alg}_current_step', 0)
                                
                                # Reset animation states
                                setattr(self, f'{alg}_animating', False)
                                setattr(self, f'{alg}_paused', False)
                                setattr(self, f'{alg}_animation_frame', 0)
                                setattr(self, f'{alg}_animation_speed', 10)
                        else:
                            # Leaving Individual View - disable all step modes
                            for alg in ['astar', 'dijkstra', 'greedy']:
                                if hasattr(self, f'{alg}_step_mode'):
                                    setattr(self, f'{alg}_step_mode', False)
                                    setattr(self, f'{alg}_current_step', 0)
                        
                        return  # Don't run algorithms again
                        
                # Check if exit button in individual view was clicked
                if self.individual_view_mode:
                    # Handle Individual View buttons
                    if hasattr(self, 'individual_exit_btn_rect') and self.individual_exit_btn_rect.collidepoint(event.pos):
                        self.individual_view_mode = False
                        # Turn off all step modes when leaving Individual View
                        for alg in ['astar', 'dijkstra', 'greedy']:
                            if hasattr(self, f'{alg}_step_mode'):
                                setattr(self, f'{alg}_step_mode', False)
                                setattr(self, f'{alg}_current_step', 0)
                        # Don't re-run algorithms - just exit the view
                        return
                    
                    # Handle individual control buttons for each algorithm
                    if hasattr(self, 'individual_control_buttons'):
                        for algo_name, buttons in self.individual_control_buttons.items():
                            # A* specific step mode controls
                            if algo_name == 'astar':
                                # Step Mode toggle button
                                if 'step_mode' in buttons and buttons['step_mode'].collidepoint(event.pos):
                                    # Check if points have changed (for refresh when already in step mode)
                                    points_changed = False
                                    if hasattr(self, 'last_step_start') and hasattr(self, 'last_step_goal'):
                                        points_changed = (self.grid.start != self.last_step_start or 
                                                        self.grid.goal != self.last_step_goal)
                                    
                                    # Toggle or refresh step mode
                                    if not self.astar_step_mode or points_changed:
                                        self.astar_step_mode = True
                                        
                                        # Check if we have start and goal
                                        if not self.grid or not self.grid.start or not self.grid.goal:
                                            self.astar_step_mode = False
                                            return
                                        
                                        # Store current points
                                        self.last_step_start = self.grid.start
                                        self.last_step_goal = self.grid.goal
                                        
                                        # Always re-run A* with educational mode to capture fresh step data
                                        self.educational_mode.enabled = True
                                        self.educational_mode.reset()
                                        
                                        # Run algorithms to capture step data
                                        self.run_algorithms()
                                        
                                        # Update max steps after running
                                        self.educational_mode.update_max_steps()
                                        
                                        # Turn off educational mode for display (we just need the data)
                                        self.educational_mode.enabled = False
                                        
                                        # Reset step counter to start
                                        self.astar_current_step = 0
                                    else:
                                        # Turning off step mode
                                        self.astar_step_mode = False
                                    return
                                
                                # Prev button (only if step mode is active)
                                if 'prev' in buttons and buttons['prev'] and buttons['prev'].collidepoint(event.pos):
                                    self.astar_current_step = max(0, self.astar_current_step - 1)
                                    return
                                
                                # Next button (only if step mode is active)
                                if 'next' in buttons and buttons['next'] and buttons['next'].collidepoint(event.pos):
                                    max_steps = len(self.educational_mode.step_data.get('astar', []))
                                    self.astar_current_step = min(max_steps, self.astar_current_step + 1)
                                    return
                                
                                # Animation controls (work in normal mode)
                                if 'run' in buttons and buttons['run'] and buttons['run'].collidepoint(event.pos):
                                    # Ensure we have step data for animation
                                    if not self.educational_mode.step_data.get('astar'):
                                        # Need to capture step data first
                                        self.educational_mode.enabled = True
                                        self.educational_mode.reset()
                                        self.run_algorithms()
                                        self.educational_mode.update_max_steps()
                                        self.educational_mode.enabled = False
                                    
                                    self.astar_animating = True
                                    self.astar_paused = False
                                    return
                                
                                if 'pause' in buttons and buttons['pause'] and buttons['pause'].collidepoint(event.pos):
                                    self.astar_paused = not self.astar_paused
                                    return
                                
                                if 'restart' in buttons and buttons['restart'] and buttons['restart'].collidepoint(event.pos):
                                    self.astar_animation_frame = 0
                                    self.astar_animating = False
                                    self.astar_paused = False
                                    return
                                
                                # Speed controls
                                if 'dec_speed' in buttons and buttons['dec_speed'] and buttons['dec_speed'].collidepoint(event.pos):
                                    if self.astar_animation_speed <= 10:
                                        # Fine control for speeds 1-10
                                        self.astar_animation_speed = max(1, self.astar_animation_speed - 1)
                                    else:
                                        # Bigger jumps for speeds above 10
                                        self.astar_animation_speed = max(10, self.astar_animation_speed - 5)
                                    return
                                
                                if 'inc_speed' in buttons and buttons['inc_speed'] and buttons['inc_speed'].collidepoint(event.pos):
                                    if self.astar_animation_speed < 10:
                                        # Fine control for speeds 1-10
                                        self.astar_animation_speed = min(10, self.astar_animation_speed + 1)
                                    else:
                                        # Bigger jumps for speeds above 10
                                        self.astar_animation_speed = min(100, self.astar_animation_speed + 5)
                                    return
                            
                            # Dijkstra specific step mode controls
                            elif algo_name == 'dijkstra':
                                # Step Mode toggle button
                                if 'step_mode' in buttons and buttons['step_mode'].collidepoint(event.pos):
                                    # Check if points have changed (for refresh when already in step mode)
                                    points_changed = False
                                    if hasattr(self, 'last_dijkstra_step_start') and hasattr(self, 'last_dijkstra_step_goal'):
                                        points_changed = (self.grid.start != self.last_dijkstra_step_start or 
                                                        self.grid.goal != self.last_dijkstra_step_goal)
                                    
                                    # Toggle or refresh step mode
                                    if not self.dijkstra_step_mode or points_changed:
                                        self.dijkstra_step_mode = True
                                        
                                        # Check if we have start and goal
                                        if not self.grid or not self.grid.start or not self.grid.goal:
                                            self.dijkstra_step_mode = False
                                            return
                                        
                                        # Store current points
                                        self.last_dijkstra_step_start = self.grid.start
                                        self.last_dijkstra_step_goal = self.grid.goal
                                        
                                        # Always re-run algorithms with educational mode to capture fresh step data
                                        self.educational_mode.enabled = True
                                        self.educational_mode.reset()
                                        
                                        # Run algorithms to capture step data
                                        self.run_algorithms()
                                        
                                        # Update max steps after running
                                        self.educational_mode.update_max_steps()
                                        
                                        # Turn off educational mode for display (we just need the data)
                                        self.educational_mode.enabled = False
                                        
                                        # Reset step counter to start
                                        self.dijkstra_current_step = 0
                                    else:
                                        # Turning off step mode
                                        self.dijkstra_step_mode = False
                                    return
                                
                                # Prev button (only if step mode is active)
                                if 'prev' in buttons and buttons['prev'] and buttons['prev'].collidepoint(event.pos):
                                    self.dijkstra_current_step = max(0, self.dijkstra_current_step - 1)
                                    return
                                
                                # Next button (only if step mode is active)
                                if 'next' in buttons and buttons['next'] and buttons['next'].collidepoint(event.pos):
                                    max_steps = len(self.educational_mode.step_data.get('dijkstra', []))
                                    self.dijkstra_current_step = min(max_steps, self.dijkstra_current_step + 1)
                                    return
                                
                                # Animation controls (work in normal mode)
                                if 'run' in buttons and buttons['run'] and buttons['run'].collidepoint(event.pos):
                                    # Ensure we have step data for animation
                                    if not self.educational_mode.step_data.get('dijkstra'):
                                        # Need to capture step data first
                                        self.educational_mode.enabled = True
                                        self.educational_mode.reset()
                                        self.run_algorithms()
                                        self.educational_mode.update_max_steps()
                                        self.educational_mode.enabled = False
                                    
                                    self.dijkstra_animating = True
                                    self.dijkstra_paused = False
                                    return
                                
                                if 'pause' in buttons and buttons['pause'] and buttons['pause'].collidepoint(event.pos):
                                    self.dijkstra_paused = not self.dijkstra_paused
                                    return
                                
                                if 'restart' in buttons and buttons['restart'] and buttons['restart'].collidepoint(event.pos):
                                    self.dijkstra_animation_frame = 0
                                    self.dijkstra_animating = False
                                    self.dijkstra_paused = False
                                    return
                                
                                # Speed controls
                                if 'dec_speed' in buttons and buttons['dec_speed'] and buttons['dec_speed'].collidepoint(event.pos):
                                    if self.dijkstra_animation_speed <= 10:
                                        # Fine control for speeds 1-10
                                        self.dijkstra_animation_speed = max(1, self.dijkstra_animation_speed - 1)
                                    else:
                                        # Bigger jumps for speeds above 10
                                        self.dijkstra_animation_speed = max(10, self.dijkstra_animation_speed - 5)
                                    return
                                
                                if 'inc_speed' in buttons and buttons['inc_speed'] and buttons['inc_speed'].collidepoint(event.pos):
                                    if self.dijkstra_animation_speed < 10:
                                        # Fine control for speeds 1-10
                                        self.dijkstra_animation_speed = min(10, self.dijkstra_animation_speed + 1)
                                    else:
                                        # Bigger jumps for speeds above 10
                                        self.dijkstra_animation_speed = min(100, self.dijkstra_animation_speed + 5)
                                    return
                            
                            # Greedy specific step mode controls
                            elif algo_name == 'greedy':
                                # Step Mode toggle button
                                if 'step_mode' in buttons and buttons['step_mode'].collidepoint(event.pos):
                                    # Check if points have changed (for refresh when already in step mode)
                                    points_changed = False
                                    if hasattr(self, 'last_greedy_step_start') and hasattr(self, 'last_greedy_step_goal'):
                                        points_changed = (self.grid.start != self.last_greedy_step_start or 
                                                        self.grid.goal != self.last_greedy_step_goal)
                                    
                                    # Toggle or refresh step mode
                                    if not self.greedy_step_mode or points_changed:
                                        self.greedy_step_mode = True
                                        
                                        # Check if we have start and goal
                                        if not self.grid or not self.grid.start or not self.grid.goal:
                                            self.greedy_step_mode = False
                                            return
                                        
                                        # Store current points
                                        self.last_greedy_step_start = self.grid.start
                                        self.last_greedy_step_goal = self.grid.goal
                                        
                                        # Always re-run algorithms with educational mode to capture fresh step data
                                        self.educational_mode.enabled = True
                                        self.educational_mode.reset()
                                        
                                        # Run algorithms to capture step data
                                        self.run_algorithms()
                                        
                                        # Update max steps after running
                                        self.educational_mode.update_max_steps()
                                        
                                        # Turn off educational mode for display (we just need the data)
                                        self.educational_mode.enabled = False
                                        
                                        # Reset step counter to start
                                        self.greedy_current_step = 0
                                    else:
                                        # Turning off step mode
                                        self.greedy_step_mode = False
                                    return
                                
                                # Prev button (only if step mode is active)
                                if 'prev' in buttons and buttons['prev'] and buttons['prev'].collidepoint(event.pos):
                                    self.greedy_current_step = max(0, self.greedy_current_step - 1)
                                    return
                                
                                # Next button (only if step mode is active)
                                if 'next' in buttons and buttons['next'] and buttons['next'].collidepoint(event.pos):
                                    max_steps = len(self.educational_mode.step_data.get('greedy', []))
                                    self.greedy_current_step = min(max_steps, self.greedy_current_step + 1)
                                    return
                                
                                # Animation controls (work in normal mode)
                                if 'run' in buttons and buttons['run'] and buttons['run'].collidepoint(event.pos):
                                    # Ensure we have step data for animation
                                    if not self.educational_mode.step_data.get('greedy'):
                                        # Need to capture step data first
                                        self.educational_mode.enabled = True
                                        self.educational_mode.reset()
                                        self.run_algorithms()
                                        self.educational_mode.update_max_steps()
                                        self.educational_mode.enabled = False
                                    
                                    self.greedy_animating = True
                                    self.greedy_paused = False
                                    return
                                
                                if 'pause' in buttons and buttons['pause'] and buttons['pause'].collidepoint(event.pos):
                                    self.greedy_paused = not self.greedy_paused
                                    return
                                
                                if 'restart' in buttons and buttons['restart'] and buttons['restart'].collidepoint(event.pos):
                                    self.greedy_animation_frame = 0
                                    self.greedy_animating = False
                                    self.greedy_paused = False
                                    return
                                
                                # Speed controls
                                if 'dec_speed' in buttons and buttons['dec_speed'] and buttons['dec_speed'].collidepoint(event.pos):
                                    if self.greedy_animation_speed <= 10:
                                        # Fine control for speeds 1-10
                                        self.greedy_animation_speed = max(1, self.greedy_animation_speed - 1)
                                    else:
                                        # Bigger jumps for speeds above 10
                                        self.greedy_animation_speed = max(10, self.greedy_animation_speed - 5)
                                    return
                                
                                if 'inc_speed' in buttons and buttons['inc_speed'] and buttons['inc_speed'].collidepoint(event.pos):
                                    if self.greedy_animation_speed < 10:
                                        # Fine control for speeds 1-10
                                        self.greedy_animation_speed = min(10, self.greedy_animation_speed + 1)
                                    else:
                                        # Bigger jumps for speeds above 10
                                        self.greedy_animation_speed = min(100, self.greedy_animation_speed + 5)
                                    return
                            
                            # Animation controls
                            if 'run' in buttons and buttons['run'].collidepoint(event.pos):
                                self.individual_animating[algo_name] = True
                                self.individual_paused[algo_name] = False
                                return
                            if 'pause' in buttons and buttons['pause'].collidepoint(event.pos):
                                self.individual_paused[algo_name] = not self.individual_paused[algo_name]
                                return
                            if 'reset' in buttons and buttons['reset'].collidepoint(event.pos):
                                # Reset to beginning
                                self.individual_steps[algo_name] = 0
                                self.individual_animation_frames[algo_name] = 0
                                self.individual_animating[algo_name] = False
                                self.individual_paused[algo_name] = False
                                # Also reset timer if it exists
                                if hasattr(self, 'individual_timers'):
                                    self.individual_timers[algo_name] = 0
                                return
                            if 'speed' in buttons and buttons['speed'].collidepoint(event.pos):
                                # Cycle through speeds
                                current = self.individual_speeds[algo_name]
                                if current < 5:
                                    self.individual_speeds[algo_name] = 5
                                elif current < 10:
                                    self.individual_speeds[algo_name] = 10
                                elif current < 20:
                                    self.individual_speeds[algo_name] = 20
                                elif current < 50:
                                    self.individual_speeds[algo_name] = 50
                                else:
                                    self.individual_speeds[algo_name] = 1
                                return
                
                # Run algorithms after drawing
                if self.last_mouse_grid_pos is not None:
                    self.run_algorithms()
                    self.last_mouse_grid_pos = None
                    
        elif event.type == pygame.MOUSEMOTION:
            if self.mouse_drawing:
                self.handle_mouse_drag(event.pos)
                
    def run(self):
        """Main application loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.window_width = max(1200, event.w)  # Minimum for usability
                    self.window_height = max(700, event.h)
                    self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
                    # Recalculate everything
                    self.setup_layout()
                    self.create_ui_elements()  # Recreate UI with new sizes
                    self.setup_layout()
                    self.create_ui_elements()
                    self.calculate_grid_dimensions()
                    # Keep fonts at fixed sizes
                    # Don't recalculate fonts on resize
                else:
                    self.handle_event(event)
                    
            # Update
            self.update()
            
            # Draw
            self.draw()
            pygame.display.flip()
            
            # Control frame rate
            clock.tick(60)
            
        pygame.quit()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 70)
    print("Emergency Vehicle Routing System - Interactive Edition")
    print("=" * 70)
    print("\nStarting interactive system with mouse controls...")
    print("\nFeatures:")
    print("- Click and drag to draw roads")
    print("- Select road types from left panel")
    print("- Place custom start/goal positions")
    print("- Watch algorithms compete in real-time")
    print("- NO 'optimal' judgments - see for yourself!")
    
    app = InteractiveRoutingSystem()
    app.run()
    
    print("\nThank you for using the Interactive Routing System!")
    print("=" * 70)

if __name__ == "__main__":
    main()