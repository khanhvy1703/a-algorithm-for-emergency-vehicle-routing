#!/usr/bin/env python3
"""
REALISTIC Traffic Simulation with proper lanes, dividers, and traffic flow
"""

import pygame
import random
import math
from enum import Enum

pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ROAD_GRAY = (60, 60, 60)
LANE_MARKING = (255, 255, 200)
DIVIDER_YELLOW = (255, 200, 0)
RED = (220, 50, 50)
BLUE = (70, 130, 255)
GREEN = (50, 200, 50)
SILVER = (192, 192, 192)
DARK_BLUE = (20, 50, 120)
DARK_GREEN = (20, 80, 20)
ORANGE = (255, 140, 0)
PURPLE = (150, 50, 200)

# Car colors for variety
CAR_COLORS = [
    (200, 50, 50),    # Red
    (50, 100, 200),   # Blue
    (180, 180, 180),  # Silver
    (50, 50, 50),     # Black
    (255, 255, 255),  # White
    (50, 150, 50),    # Green
    (150, 50, 150),   # Purple
    (200, 150, 50),   # Gold
]

class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

class LaneType(Enum):
    NORTHBOUND_LEFT = "NL"
    NORTHBOUND_RIGHT = "NR"
    SOUTHBOUND_LEFT = "SL"
    SOUTHBOUND_RIGHT = "SR"
    EASTBOUND_TOP = "ET"
    EASTBOUND_BOTTOM = "EB"
    WESTBOUND_TOP = "WT"
    WESTBOUND_BOTTOM = "WB"

class Vehicle:
    """Realistic vehicle with proper lane following"""
    def __init__(self, x, y, lane_type, direction, speed=1.0):
        self.x = x  # Actual pixel position
        self.y = y
        self.lane_type = lane_type
        self.direction = direction
        self.speed = speed * random.uniform(0.8, 1.2)  # Slight speed variation
        self.color = random.choice(CAR_COLORS)
        self.length = 20
        self.width = 12
        self.safe_distance = 30  # Minimum distance to car in front
        self.brake_lights = False
        self.turn_signal = None  # 'left', 'right', or None
        
    def update(self, vehicles, roads):
        """Update vehicle position with proper lane following"""
        # Check for vehicle in front
        front_vehicle = self.get_vehicle_in_front(vehicles)
        
        if front_vehicle:
            distance = self.calculate_distance(front_vehicle)
            if distance < self.safe_distance + 10:
                # Too close, brake
                self.brake_lights = True
                if distance < self.safe_distance:
                    return  # Stop completely
                else:
                    # Slow down
                    actual_speed = self.speed * 0.3
            else:
                self.brake_lights = False
                actual_speed = self.speed
        else:
            self.brake_lights = False
            actual_speed = self.speed
        
        # Move based on direction
        dx, dy = self.direction.value
        self.x += dx * actual_speed
        self.y += dy * actual_speed
        
    def get_vehicle_in_front(self, vehicles):
        """Find vehicle directly in front in same lane"""
        closest = None
        min_distance = float('inf')
        
        for other in vehicles:
            if other == self or other.lane_type != self.lane_type:
                continue
                
            # Check if it's in front based on direction
            dx, dy = self.direction.value
            if dx > 0:  # Going East
                if other.x > self.x and abs(other.y - self.y) < 10:
                    dist = other.x - self.x
                    if dist < min_distance:
                        min_distance = dist
                        closest = other
            elif dx < 0:  # Going West
                if other.x < self.x and abs(other.y - self.y) < 10:
                    dist = self.x - other.x
                    if dist < min_distance:
                        min_distance = dist
                        closest = other
            elif dy > 0:  # Going South
                if other.y > self.y and abs(other.x - self.x) < 10:
                    dist = other.y - self.y
                    if dist < min_distance:
                        min_distance = dist
                        closest = other
            elif dy < 0:  # Going North
                if other.y < self.y and abs(other.x - self.x) < 10:
                    dist = self.y - other.y
                    if dist < min_distance:
                        min_distance = dist
                        closest = other
        
        return closest
    
    def calculate_distance(self, other):
        """Calculate distance to another vehicle"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def draw(self, screen):
        """Draw realistic vehicle"""
        # Create rotated rectangle for car body
        if self.direction in [Direction.NORTH, Direction.SOUTH]:
            width, height = self.width, self.length
        else:
            width, height = self.length, self.width
        
        # Car body
        car_rect = pygame.Rect(self.x - width//2, self.y - height//2, width, height)
        pygame.draw.rect(screen, self.color, car_rect)
        pygame.draw.rect(screen, BLACK, car_rect, 1)
        
        # Windows (darker rectangles)
        window_color = (max(0, self.color[0] - 50), 
                       max(0, self.color[1] - 50),
                       max(0, self.color[2] - 50))
        
        if self.direction in [Direction.NORTH, Direction.SOUTH]:
            # Front and back windows
            window_rect = pygame.Rect(self.x - width//4, self.y - height//3, width//2, height//6)
            pygame.draw.rect(screen, window_color, window_rect)
            window_rect = pygame.Rect(self.x - width//4, self.y + height//6, width//2, height//6)
            pygame.draw.rect(screen, window_color, window_rect)
        else:
            # Side windows
            window_rect = pygame.Rect(self.x - width//3, self.y - height//4, width//6, height//2)
            pygame.draw.rect(screen, window_color, window_rect)
            window_rect = pygame.Rect(self.x + width//6, self.y - height//4, width//6, height//2)
            pygame.draw.rect(screen, window_color, window_rect)
        
        # Brake lights
        if self.brake_lights:
            if self.direction == Direction.NORTH:
                pygame.draw.circle(screen, RED, (int(self.x - 4), int(self.y + height//2)), 2)
                pygame.draw.circle(screen, RED, (int(self.x + 4), int(self.y + height//2)), 2)
            elif self.direction == Direction.SOUTH:
                pygame.draw.circle(screen, RED, (int(self.x - 4), int(self.y - height//2)), 2)
                pygame.draw.circle(screen, RED, (int(self.x + 4), int(self.y - height//2)), 2)
            elif self.direction == Direction.WEST:
                pygame.draw.circle(screen, RED, (int(self.x + width//2), int(self.y - 4)), 2)
                pygame.draw.circle(screen, RED, (int(self.x + width//2), int(self.y + 4)), 2)
            elif self.direction == Direction.EAST:
                pygame.draw.circle(screen, RED, (int(self.x - width//2), int(self.y - 4)), 2)
                pygame.draw.circle(screen, RED, (int(self.x - width//2), int(self.y + 4)), 2)

class Road:
    """Represents a road with multiple lanes"""
    def __init__(self, x, y, width, height, orientation):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.orientation = orientation  # 'horizontal' or 'vertical'
        self.lanes = []
        self.setup_lanes()
    
    def setup_lanes(self):
        """Setup lanes based on orientation"""
        if self.orientation == 'horizontal':
            # Top lanes go west, bottom lanes go east
            lane_height = self.height // 4
            self.lanes = [
                {'type': LaneType.WESTBOUND_TOP, 'y': self.y + lane_height * 0.5, 'direction': Direction.WEST},
                {'type': LaneType.WESTBOUND_BOTTOM, 'y': self.y + lane_height * 1.5, 'direction': Direction.WEST},
                {'type': LaneType.EASTBOUND_TOP, 'y': self.y + lane_height * 2.5, 'direction': Direction.EAST},
                {'type': LaneType.EASTBOUND_BOTTOM, 'y': self.y + lane_height * 3.5, 'direction': Direction.EAST},
            ]
        else:  # vertical
            # Left lanes go north, right lanes go south
            lane_width = self.width // 4
            self.lanes = [
                {'type': LaneType.NORTHBOUND_LEFT, 'x': self.x + lane_width * 0.5, 'direction': Direction.NORTH},
                {'type': LaneType.NORTHBOUND_RIGHT, 'x': self.x + lane_width * 1.5, 'direction': Direction.NORTH},
                {'type': LaneType.SOUTHBOUND_LEFT, 'x': self.x + lane_width * 2.5, 'direction': Direction.SOUTH},
                {'type': LaneType.SOUTHBOUND_RIGHT, 'x': self.x + lane_width * 3.5, 'direction': Direction.SOUTH},
            ]
    
    def draw(self, screen):
        """Draw road with lanes and markings"""
        # Draw road surface
        road_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, ROAD_GRAY, road_rect)
        
        if self.orientation == 'horizontal':
            # Draw center divider (yellow double line)
            center_y = self.y + self.height // 2
            pygame.draw.line(screen, DIVIDER_YELLOW, 
                           (self.x, center_y - 2), 
                           (self.x + self.width, center_y - 2), 2)
            pygame.draw.line(screen, DIVIDER_YELLOW, 
                           (self.x, center_y + 2), 
                           (self.x + self.width, center_y + 2), 2)
            
            # Draw lane markings (dashed white lines)
            lane_height = self.height // 4
            for i in [1, 3]:  # Lane dividers within same direction
                lane_y = self.y + lane_height * i
                if i != 2:  # Skip center
                    for x in range(self.x, self.x + self.width, 30):
                        pygame.draw.line(screen, LANE_MARKING,
                                       (x, lane_y),
                                       (min(x + 15, self.x + self.width), lane_y), 2)
        else:  # vertical
            # Draw center divider (yellow double line)
            center_x = self.x + self.width // 2
            pygame.draw.line(screen, DIVIDER_YELLOW, 
                           (center_x - 2, self.y), 
                           (center_x - 2, self.y + self.height), 2)
            pygame.draw.line(screen, DIVIDER_YELLOW, 
                           (center_x + 2, self.y), 
                           (center_x + 2, self.y + self.height), 2)
            
            # Draw lane markings (dashed white lines)
            lane_width = self.width // 4
            for i in [1, 3]:  # Lane dividers within same direction
                lane_x = self.x + lane_width * i
                if i != 2:  # Skip center
                    for y in range(self.y, self.y + self.height, 30):
                        pygame.draw.line(screen, LANE_MARKING,
                                       (lane_x, y),
                                       (lane_x, min(y + 15, self.y + self.height)), 2)

class RealisticTrafficSimulation:
    def __init__(self):
        self.width = 1400
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("REALISTIC Traffic Simulation - Proper Lanes & Flow")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        
        # Create roads
        self.roads = []
        self.setup_roads()
        
        # Vehicles
        self.vehicles = []
        self.max_vehicles = 30
        self.spawn_timer = 0
        self.spawn_delay = 20
        
        # Control
        self.paused = False
        self.show_stats = True
        
    def setup_roads(self):
        """Create realistic road network"""
        # Main horizontal road
        self.roads.append(Road(100, 350, 1200, 100, 'horizontal'))
        
        # Main vertical roads
        self.roads.append(Road(300, 50, 100, 700, 'vertical'))
        self.roads.append(Road(700, 50, 100, 700, 'vertical'))
        self.roads.append(Road(1000, 50, 100, 700, 'vertical'))
    
    def spawn_vehicle(self):
        """Spawn vehicle in a random lane"""
        if len(self.vehicles) >= self.max_vehicles:
            return
        
        # Pick random road and lane
        road = random.choice(self.roads)
        lane = random.choice(road.lanes)
        
        # Determine spawn position based on lane direction
        if lane['direction'] == Direction.EAST:
            x = road.x + random.randint(0, 50)
            y = lane['y']
        elif lane['direction'] == Direction.WEST:
            x = road.x + road.width - random.randint(0, 50)
            y = lane['y']
        elif lane['direction'] == Direction.NORTH:
            x = lane['x']
            y = road.y + road.height - random.randint(0, 50)
        else:  # SOUTH
            x = lane['x']
            y = road.y + random.randint(0, 50)
        
        # Check if spawn position is clear
        for vehicle in self.vehicles:
            if abs(vehicle.x - x) < 40 and abs(vehicle.y - y) < 40:
                return  # Too close to existing vehicle
        
        vehicle = Vehicle(x, y, lane['type'], lane['direction'])
        self.vehicles.append(vehicle)
    
    def update(self):
        """Update simulation"""
        if not self.paused:
            # Update vehicles
            for vehicle in self.vehicles:
                vehicle.update(self.vehicles, self.roads)
            
            # Remove vehicles that have left the screen
            self.vehicles = [v for v in self.vehicles 
                           if -50 < v.x < self.width + 50 and -50 < v.y < self.height + 50]
            
            # Spawn new vehicles
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_delay:
                self.spawn_timer = 0
                self.spawn_vehicle()
    
    def draw(self):
        """Draw everything"""
        # Background
        self.screen.fill((40, 120, 40))  # Grass green
        
        # Draw roads
        for road in self.roads:
            road.draw(self.screen)
        
        # Draw intersections (cover the overlap nicely)
        for i, road1 in enumerate(self.roads):
            for road2 in self.roads[i+1:]:
                if road1.orientation != road2.orientation:
                    # Find intersection
                    inter_rect = pygame.Rect(road1.x, road1.y, road1.width, road1.height).clip(
                        pygame.Rect(road2.x, road2.y, road2.width, road2.height))
                    if inter_rect.width > 0 and inter_rect.height > 0:
                        pygame.draw.rect(self.screen, ROAD_GRAY, inter_rect)
                        # Crosswalk lines
                        for offset in range(0, inter_rect.width, 10):
                            pygame.draw.line(self.screen, WHITE,
                                           (inter_rect.x + offset, inter_rect.y),
                                           (inter_rect.x + offset, inter_rect.y + inter_rect.height), 4)
        
        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(self.screen)
        
        # Draw UI
        self.draw_ui()
    
    def draw_ui(self):
        """Draw user interface"""
        if self.show_stats:
            # Title
            title = self.font.render("REALISTIC Traffic Simulation", True, WHITE)
            title_bg = pygame.Surface((title.get_width() + 20, title.get_height() + 10))
            title_bg.fill(BLACK)
            title_bg.set_alpha(180)
            self.screen.blit(title_bg, (10, 10))
            self.screen.blit(title, (20, 15))
            
            # Stats
            stats = [
                f"Vehicles: {len(self.vehicles)}/{self.max_vehicles}",
                f"Status: {'PAUSED' if self.paused else 'RUNNING'}",
                "",
                "Features:",
                "✓ Proper lane system",
                "✓ Opposite traffic flow", 
                "✓ Center dividers",
                "✓ Lane markings",
                "✓ Brake lights",
                "✓ Safe following distance"
            ]
            
            y = 60
            for stat in stats:
                if stat.startswith("✓"):
                    color = GREEN
                else:
                    color = WHITE
                text = self.small_font.render(stat, True, color)
                if stat:  # Only draw background for non-empty lines
                    text_bg = pygame.Surface((250, text.get_height() + 4))
                    text_bg.fill(BLACK)
                    text_bg.set_alpha(180)
                    self.screen.blit(text_bg, (10, y - 2))
                self.screen.blit(text, (20, y))
                y += 25
            
            # Controls
            controls = [
                "Controls:",
                "SPACE - Pause/Resume",
                "+/- Vehicle Count",
                "S - Toggle Stats",
                "ESC - Exit"
            ]
            
            y = self.height - 140
            for control in controls:
                text = self.small_font.render(control, True, WHITE)
                text_bg = pygame.Surface((200, text.get_height() + 4))
                text_bg.fill(BLACK)
                text_bg.set_alpha(180)
                self.screen.blit(text_bg, (10, y - 2))
                self.screen.blit(text, (20, y))
                y += 25
    
    def handle_event(self, event):
        """Handle user input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_s:
                self.show_stats = not self.show_stats
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.max_vehicles = min(50, self.max_vehicles + 5)
            elif event.key == pygame.K_MINUS:
                self.max_vehicles = max(5, self.max_vehicles - 5)
                while len(self.vehicles) > self.max_vehicles:
                    self.vehicles.pop()
    
    def run(self):
        """Main loop"""
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
            self.clock.tick(60)  # 60 FPS for smooth animation
        
        pygame.quit()

if __name__ == "__main__":
    sim = RealisticTrafficSimulation()
    sim.run()