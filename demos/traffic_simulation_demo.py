#!/usr/bin/env python3
"""
Phase 1: Real-Time Traffic Simulation Demo
Shows vehicles moving on roads in the emergency routing system
"""

import pygame
import random
import math
from enum import Enum

pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
RED = (255, 100, 100)
BLUE = (100, 150, 255)
GREEN = (100, 255, 100)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)
PURPLE = (200, 100, 255)

class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

class Vehicle:
    """Represents a moving vehicle in traffic"""
    def __init__(self, x, y, direction, speed=1.0, color=None):
        self.x = x  # Grid position
        self.y = y
        self.pixel_x = x * 30 + 15  # Actual pixel position (center of cell)
        self.pixel_y = y * 30 + 15
        self.direction = direction
        self.speed = speed
        self.base_speed = speed
        self.color = color or random.choice([BLUE, GREEN, YELLOW, PURPLE, ORANGE])
        self.size = 8
        self.trail = []  # For smooth movement visualization
        self.target_x = self.pixel_x
        self.target_y = self.pixel_y
        self.moving = True
        
    def update(self, grid, vehicles):
        """Update vehicle position"""
        if not self.moving:
            return
            
        # Smooth pixel movement toward target
        dx = self.target_x - self.pixel_x
        dy = self.target_y - self.pixel_y
        
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            # Move toward target
            move_speed = self.speed * 2
            self.pixel_x += move_speed * (1 if dx > 0 else -1 if dx < 0 else 0)
            self.pixel_y += move_speed * (1 if dy > 0 else -1 if dy < 0 else 0)
        else:
            # Reached target, pick next cell
            self.pixel_x = self.target_x
            self.pixel_y = self.target_y
            
            # Update grid position
            self.x = int(self.pixel_x // 30)
            self.y = int(self.pixel_y // 30)
            
            # Find next valid move
            next_pos = self.get_next_position(grid, vehicles)
            if next_pos:
                self.x, self.y = next_pos
                self.target_x = self.x * 30 + 15
                self.target_y = self.y * 30 + 15
            else:
                # Can't move, try to change direction
                self.change_direction(grid)
    
    def get_next_position(self, grid, vehicles):
        """Get next valid position based on direction"""
        dx, dy = self.direction.value
        next_x = self.x + dx
        next_y = self.y + dy
        
        # Check bounds
        if not (0 <= next_x < len(grid[0]) and 0 <= next_y < len(grid)):
            return None
            
        # Check if road exists
        if not grid[next_y][next_x]:
            return None
            
        # Check for collision with other vehicles
        for other in vehicles:
            if other != self and other.x == next_x and other.y == next_y:
                # Check if we're too close
                dist = math.sqrt((other.pixel_x - self.pixel_x)**2 + 
                               (other.pixel_y - self.pixel_y)**2)
                if dist < 25:  # Minimum safe distance
                    return None
        
        return (next_x, next_y)
    
    def change_direction(self, grid):
        """Change direction at intersection or dead end"""
        possible_directions = []
        
        for direction in Direction:
            dx, dy = direction.value
            next_x = self.x + dx
            next_y = self.y + dy
            
            if (0 <= next_x < len(grid[0]) and 
                0 <= next_y < len(grid) and 
                grid[next_y][next_x]):
                possible_directions.append(direction)
        
        # Prefer not going backwards
        current_opposite = self.get_opposite_direction()
        non_backward = [d for d in possible_directions if d != current_opposite]
        
        if non_backward:
            self.direction = random.choice(non_backward)
        elif possible_directions:
            self.direction = random.choice(possible_directions)
    
    def get_opposite_direction(self):
        """Get opposite of current direction"""
        opposites = {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST
        }
        return opposites[self.direction]
    
    def draw(self, screen):
        """Draw the vehicle"""
        # Draw vehicle as a small rectangle oriented in direction
        if self.direction in [Direction.NORTH, Direction.SOUTH]:
            width, height = self.size, self.size * 1.5
        else:
            width, height = self.size * 1.5, self.size
            
        rect = pygame.Rect(self.pixel_x - width//2, 
                          self.pixel_y - height//2,
                          width, height)
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)
        
        # Draw direction indicator (small triangle)
        dx, dy = self.direction.value
        front_x = self.pixel_x + dx * 8
        front_y = self.pixel_y + dy * 8
        pygame.draw.circle(screen, WHITE, (int(front_x), int(front_y)), 2)

class TrafficSimulation:
    def __init__(self):
        self.width = 1200
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Phase 1: Basic Traffic Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Grid setup (40x20)
        self.grid_width = 40
        self.grid_height = 20
        self.cell_size = 30
        self.grid = [[False for _ in range(self.grid_width)] 
                     for _ in range(self.grid_height)]
        
        # Create sample road network
        self.create_road_network()
        
        # Vehicles
        self.vehicles = []
        self.max_vehicles = 20
        self.spawn_timer = 0
        self.spawn_delay = 30  # Frames between spawns
        
        # Control flags
        self.paused = False
        self.show_grid = True
        
    def create_road_network(self):
        """Create a sample road network"""
        # Main horizontal roads
        for y in [5, 10, 15]:
            for x in range(self.grid_width):
                self.grid[y][x] = True
        
        # Main vertical roads
        for x in [8, 16, 24, 32]:
            for y in range(self.grid_height):
                self.grid[y][x] = True
        
        # Some diagonal connections
        for i in range(5):
            if i + 11 < self.grid_height and i + 20 < self.grid_width:
                self.grid[i + 11][i + 20] = True
                self.grid[i + 10][i + 20] = True
                self.grid[i + 11][i + 19] = True
    
    def spawn_vehicle(self):
        """Spawn a new vehicle at a random road position"""
        if len(self.vehicles) >= self.max_vehicles:
            return
        
        # Find all valid spawn points (road cells)
        spawn_points = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y][x]:
                    # Check if no vehicle nearby
                    occupied = False
                    for vehicle in self.vehicles:
                        if abs(vehicle.x - x) <= 1 and abs(vehicle.y - y) <= 1:
                            occupied = True
                            break
                    if not occupied:
                        spawn_points.append((x, y))
        
        if spawn_points:
            x, y = random.choice(spawn_points)
            direction = random.choice(list(Direction))
            speed = random.uniform(0.5, 1.5)
            vehicle = Vehicle(x, y, direction, speed)
            self.vehicles.append(vehicle)
    
    def update(self):
        """Update simulation state"""
        if not self.paused:
            # Update all vehicles
            for vehicle in self.vehicles:
                vehicle.update(self.grid, self.vehicles)
            
            # Remove vehicles that are stuck for too long
            self.vehicles = [v for v in self.vehicles if v.moving or random.random() > 0.01]
            
            # Spawn new vehicles periodically
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_delay:
                self.spawn_timer = 0
                self.spawn_vehicle()
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(WHITE)
        
        # Draw grid/roads
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)
                if self.grid[y][x]:
                    # Road cell
                    pygame.draw.rect(self.screen, GRAY, rect)
                    if self.show_grid:
                        pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
                else:
                    # Empty cell
                    if self.show_grid:
                        pygame.draw.rect(self.screen, LIGHT_GRAY, rect, 1)
        
        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(self.screen)
        
        # Draw UI
        self.draw_ui()
    
    def draw_ui(self):
        """Draw user interface elements"""
        # Title
        title = self.font.render("Phase 1: Basic Traffic Simulation", True, BLACK)
        self.screen.blit(title, (10, 10))
        
        # Stats
        stats = [
            f"Vehicles: {len(self.vehicles)}/{self.max_vehicles}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}",
            f"Grid: {'ON' if self.show_grid else 'OFF'}"
        ]
        
        y = 40
        for stat in stats:
            text = self.small_font.render(stat, True, BLACK)
            self.screen.blit(text, (10, y))
            y += 20
        
        # Controls
        controls = [
            "SPACE: Pause/Resume",
            "G: Toggle Grid",
            "R: Reset",
            "+/-: Add/Remove Vehicles",
            "ESC: Exit"
        ]
        
        y = self.height - 100
        for control in controls:
            text = self.small_font.render(control, True, BLACK)
            self.screen.blit(text, (10, y))
            y += 20
        
        # Feature highlights
        features = [
            "✓ Vehicles move along roads",
            "✓ Collision avoidance",
            "✓ Direction changes at intersections",
            "✓ Variable speeds",
            "✓ Smooth animation"
        ]
        
        y = 40
        for feature in features:
            text = self.small_font.render(feature, True, GREEN if "✓" in feature else BLACK)
            self.screen.blit(text, (self.width - 250, y))
            y += 20
    
    def handle_event(self, event):
        """Handle user input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_g:
                self.show_grid = not self.show_grid
            elif event.key == pygame.K_r:
                self.vehicles.clear()
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.max_vehicles = min(50, self.max_vehicles + 5)
            elif event.key == pygame.K_MINUS:
                self.max_vehicles = max(5, self.max_vehicles - 5)
                # Remove excess vehicles
                while len(self.vehicles) > self.max_vehicles:
                    self.vehicles.pop()
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    self.handle_event(event)
            
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        pygame.quit()

if __name__ == "__main__":
    sim = TrafficSimulation()
    sim.run()