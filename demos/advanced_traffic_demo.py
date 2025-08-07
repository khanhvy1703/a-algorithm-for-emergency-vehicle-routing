#!/usr/bin/env python3
"""
ADVANCED Traffic Simulation with traffic lights, turning, indicators
This is INSANE level traffic simulation!
"""

import pygame
import random
import math
from enum import Enum
from dataclasses import dataclass
import time

pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ROAD_GRAY = (60, 60, 60)
LANE_MARKING = (255, 255, 200)
DIVIDER_YELLOW = (255, 200, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 50)
ORANGE = (255, 140, 0)
BLUE = (70, 130, 255)
DARK_WINDOW = (40, 40, 60)

# Traffic light colors
LIGHT_RED = (255, 0, 0)
LIGHT_YELLOW = (255, 255, 0)
LIGHT_GREEN = (0, 255, 0)
LIGHT_OFF = (50, 50, 50)

# Car colors
CAR_COLORS = [
    (200, 50, 50),    # Red
    (50, 100, 200),   # Blue
    (180, 180, 180),  # Silver
    (50, 50, 50),     # Black
    (240, 240, 240),  # White
    (50, 150, 50),    # Green
    (150, 50, 150),   # Purple
    (200, 150, 50),   # Gold
    (100, 50, 20),    # Brown
]

class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

class TurnIntention(Enum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"

class TrafficLightState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

class TrafficLight:
    """Traffic light controller for intersections"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.states = {
            'north_south': TrafficLightState.GREEN,
            'east_west': TrafficLightState.RED
        }
        self.timer = 0
        self.green_duration = 180  # frames (3 seconds at 60fps)
        self.yellow_duration = 60   # frames (1 second)
        self.red_duration = 240     # frames (4 seconds)
        self.current_phase = 'north_south'
        
    def update(self):
        """Update traffic light states"""
        self.timer += 1
        
        if self.current_phase == 'north_south':
            if self.states['north_south'] == TrafficLightState.GREEN:
                if self.timer > self.green_duration:
                    self.states['north_south'] = TrafficLightState.YELLOW
                    self.timer = 0
            elif self.states['north_south'] == TrafficLightState.YELLOW:
                if self.timer > self.yellow_duration:
                    self.states['north_south'] = TrafficLightState.RED
                    self.states['east_west'] = TrafficLightState.GREEN
                    self.current_phase = 'east_west'
                    self.timer = 0
        else:  # east_west phase
            if self.states['east_west'] == TrafficLightState.GREEN:
                if self.timer > self.green_duration:
                    self.states['east_west'] = TrafficLightState.YELLOW
                    self.timer = 0
            elif self.states['east_west'] == TrafficLightState.YELLOW:
                if self.timer > self.yellow_duration:
                    self.states['east_west'] = TrafficLightState.RED
                    self.states['north_south'] = TrafficLightState.GREEN
                    self.current_phase = 'north_south'
                    self.timer = 0
    
    def can_go(self, direction):
        """Check if traffic can go in given direction"""
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return self.states['north_south'] != TrafficLightState.RED
        else:
            return self.states['east_west'] != TrafficLightState.RED
    
    def draw(self, screen):
        """Draw traffic lights at intersection"""
        # North-South lights
        for offset_x, offset_y, facing in [(-30, -60, Direction.SOUTH), (30, 60, Direction.NORTH)]:
            self.draw_single_light(screen, self.x + offset_x, self.y + offset_y, 
                                  self.states['north_south'], facing)
        
        # East-West lights  
        for offset_x, offset_y, facing in [(-60, -30, Direction.EAST), (60, 30, Direction.WEST)]:
            self.draw_single_light(screen, self.x + offset_x, self.y + offset_y,
                                  self.states['east_west'], facing)
    
    def draw_single_light(self, screen, x, y, state, facing):
        """Draw a single traffic light"""
        # Light housing
        housing_rect = pygame.Rect(x - 12, y - 25, 24, 50)
        pygame.draw.rect(screen, BLACK, housing_rect)
        pygame.draw.rect(screen, YELLOW, housing_rect, 2)
        
        # Draw three lights
        colors = [
            LIGHT_RED if state == TrafficLightState.RED else LIGHT_OFF,
            LIGHT_YELLOW if state == TrafficLightState.YELLOW else LIGHT_OFF,
            LIGHT_GREEN if state == TrafficLightState.GREEN else LIGHT_OFF
        ]
        
        for i, color in enumerate(colors):
            pygame.draw.circle(screen, color, (x, y - 15 + i * 15), 6)
            if color != LIGHT_OFF:
                # Add glow effect
                for radius in range(8, 12):
                    glow_color = (*color, 50)
                    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(s, glow_color, (radius, radius), radius)
                    screen.blit(s, (x - radius, y - 15 + i * 15 - radius))

class Vehicle:
    """Advanced vehicle with turning, indicators, and traffic light awareness"""
    def __init__(self, x, y, direction, speed=1.0):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed * random.uniform(0.8, 1.2)
        self.max_speed = self.speed
        self.current_speed = self.speed
        self.color = random.choice(CAR_COLORS)
        self.length = 24
        self.width = 14
        self.angle = self.direction_to_angle(direction)
        
        # Turning
        self.turn_intention = TurnIntention.STRAIGHT
        self.is_turning = False
        self.turn_progress = 0
        self.turn_radius = 30
        self.turn_center = None
        self.target_angle = self.angle
        
        # Indicators
        self.left_indicator = False
        self.right_indicator = False
        self.indicator_timer = 0
        self.indicator_on = False
        
        # Behavior
        self.safe_distance = 40
        self.stop_distance = 15
        self.waiting_at_light = False
        self.brake_lights = False
        
        # Decide turn intention randomly
        if random.random() < 0.3:  # 30% chance to turn
            self.turn_intention = random.choice([TurnIntention.LEFT, TurnIntention.RIGHT])
            if self.turn_intention == TurnIntention.LEFT:
                self.left_indicator = True
            elif self.turn_intention == TurnIntention.RIGHT:
                self.right_indicator = True
    
    def direction_to_angle(self, direction):
        """Convert direction to angle in degrees"""
        angles = {
            Direction.NORTH: 90,
            Direction.SOUTH: 270,
            Direction.EAST: 0,
            Direction.WEST: 180
        }
        return angles[direction]
    
    def update(self, vehicles, traffic_lights, intersections):
        """Update vehicle with advanced behavior"""
        # Update indicators
        if self.left_indicator or self.right_indicator:
            self.indicator_timer += 1
            if self.indicator_timer % 30 == 0:  # Blink every 0.5 seconds
                self.indicator_on = not self.indicator_on
        
        # Check for traffic light
        ahead_light = self.check_traffic_light_ahead(traffic_lights, intersections)
        
        # Check for vehicles ahead
        front_vehicle, distance = self.get_vehicle_ahead(vehicles)
        
        # Determine speed based on conditions
        if ahead_light and not ahead_light.can_go(self.direction):
            # Red or yellow light ahead
            dist_to_light = self.distance_to_intersection(intersections)
            if dist_to_light < 100:
                if dist_to_light < self.stop_distance:
                    self.current_speed = 0
                    self.brake_lights = True
                    self.waiting_at_light = True
                else:
                    # Slow down approaching light
                    self.current_speed = max(0.5, self.max_speed * (dist_to_light / 100))
                    self.brake_lights = True
        elif front_vehicle:
            if distance < self.safe_distance:
                if distance < self.stop_distance:
                    self.current_speed = 0
                    self.brake_lights = True
                else:
                    # Match speed of vehicle ahead
                    self.current_speed = min(front_vehicle.current_speed, 
                                           self.max_speed * (distance / self.safe_distance))
                    self.brake_lights = distance < self.safe_distance * 0.7
            else:
                self.current_speed = self.max_speed
                self.brake_lights = False
                self.waiting_at_light = False
        else:
            self.current_speed = self.max_speed
            self.brake_lights = False
            self.waiting_at_light = False
        
        # Handle turning at intersections
        if not self.is_turning:
            # Check if at intersection and need to turn
            for intersection in intersections:
                if self.at_intersection(intersection) and self.turn_intention != TurnIntention.STRAIGHT:
                    if not self.waiting_at_light:  # Only turn if allowed by traffic light
                        self.start_turn(intersection)
                        break
        
        if self.is_turning:
            self.perform_turn()
        else:
            # Normal movement
            if self.current_speed > 0:
                dx = math.cos(math.radians(self.angle)) * self.current_speed
                dy = -math.sin(math.radians(self.angle)) * self.current_speed
                self.x += dx
                self.y += dy
    
    def check_traffic_light_ahead(self, traffic_lights, intersections):
        """Check if there's a traffic light ahead"""
        # Simple check - find nearest intersection ahead
        for light, intersection in zip(traffic_lights, intersections):
            if self.approaching_intersection(intersection):
                return light
        return None
    
    def approaching_intersection(self, intersection):
        """Check if approaching an intersection"""
        dist = math.sqrt((self.x - intersection[0])**2 + (self.y - intersection[1])**2)
        if dist > 150:  # Too far
            return False
        
        # Check if intersection is ahead based on direction
        dx = intersection[0] - self.x
        dy = intersection[1] - self.y
        
        if self.direction == Direction.NORTH and dy < 0:
            return True
        elif self.direction == Direction.SOUTH and dy > 0:
            return True
        elif self.direction == Direction.EAST and dx > 0:
            return True
        elif self.direction == Direction.WEST and dx < 0:
            return True
        return False
    
    def distance_to_intersection(self, intersections):
        """Get distance to nearest intersection ahead"""
        min_dist = float('inf')
        for intersection in intersections:
            if self.approaching_intersection(intersection):
                dist = math.sqrt((self.x - intersection[0])**2 + (self.y - intersection[1])**2)
                min_dist = min(min_dist, dist)
        return min_dist
    
    def at_intersection(self, intersection):
        """Check if vehicle is at intersection"""
        dist = math.sqrt((self.x - intersection[0])**2 + (self.y - intersection[1])**2)
        return dist < 20
    
    def start_turn(self, intersection):
        """Start turning at intersection"""
        self.is_turning = True
        self.turn_progress = 0
        
        # Calculate turn center and target angle
        if self.turn_intention == TurnIntention.LEFT:
            if self.direction == Direction.NORTH:
                self.turn_center = (intersection[0] - self.turn_radius, intersection[1])
                self.target_angle = 180  # West
                self.direction = Direction.WEST
            elif self.direction == Direction.SOUTH:
                self.turn_center = (intersection[0] + self.turn_radius, intersection[1])
                self.target_angle = 0  # East
                self.direction = Direction.EAST
            elif self.direction == Direction.EAST:
                self.turn_center = (intersection[0], intersection[1] - self.turn_radius)
                self.target_angle = 90  # North
                self.direction = Direction.NORTH
            else:  # West
                self.turn_center = (intersection[0], intersection[1] + self.turn_radius)
                self.target_angle = 270  # South
                self.direction = Direction.SOUTH
        else:  # Right turn
            if self.direction == Direction.NORTH:
                self.turn_center = (intersection[0] + self.turn_radius, intersection[1])
                self.target_angle = 0  # East
                self.direction = Direction.EAST
            elif self.direction == Direction.SOUTH:
                self.turn_center = (intersection[0] - self.turn_radius, intersection[1])
                self.target_angle = 180  # West
                self.direction = Direction.WEST
            elif self.direction == Direction.EAST:
                self.turn_center = (intersection[0], intersection[1] + self.turn_radius)
                self.target_angle = 270  # South
                self.direction = Direction.SOUTH
            else:  # West
                self.turn_center = (intersection[0], intersection[1] - self.turn_radius)
                self.target_angle = 90  # North
                self.direction = Direction.NORTH
    
    def perform_turn(self):
        """Execute turning animation"""
        self.turn_progress += 0.05
        
        if self.turn_progress >= 1.0:
            # Turn complete
            self.is_turning = False
            self.angle = self.target_angle
            self.left_indicator = False
            self.right_indicator = False
            self.turn_intention = TurnIntention.STRAIGHT
        else:
            # Interpolate angle
            angle_diff = self.target_angle - self.angle
            # Handle angle wrapping
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            
            # Smooth turning
            turn_speed = 3
            if abs(angle_diff) > turn_speed:
                self.angle += turn_speed if angle_diff > 0 else -turn_speed
            
            # Move along arc
            if self.turn_center:
                # Calculate position on arc
                if self.turn_intention == TurnIntention.LEFT:
                    arc_angle = self.angle + 90
                else:
                    arc_angle = self.angle - 90
                
                self.x = self.turn_center[0] + math.cos(math.radians(arc_angle)) * self.turn_radius
                self.y = self.turn_center[1] - math.sin(math.radians(arc_angle)) * self.turn_radius
    
    def get_vehicle_ahead(self, vehicles):
        """Get vehicle directly ahead"""
        min_dist = float('inf')
        closest = None
        
        # Create a cone of vision ahead
        vision_angle = 30  # degrees
        vision_distance = 100
        
        for other in vehicles:
            if other == self:
                continue
            
            # Calculate relative position
            dx = other.x - self.x
            dy = other.y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > vision_distance:
                continue
            
            # Check if in front cone
            angle_to_other = math.degrees(math.atan2(-dy, dx))
            angle_diff = abs(angle_to_other - self.angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff < vision_angle and dist < min_dist:
                min_dist = dist
                closest = other
        
        return closest, min_dist if closest else float('inf')
    
    def draw(self, screen):
        """Draw vehicle with rotation and indicators"""
        # Save original surface center
        original_center = (self.x, self.y)
        
        # Create vehicle surface
        car_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        
        # Draw car body
        pygame.draw.rect(car_surface, self.color, (0, 0, self.length, self.width))
        pygame.draw.rect(car_surface, BLACK, (0, 0, self.length, self.width), 1)
        
        # Draw windows
        window_color = DARK_WINDOW
        # Front window
        pygame.draw.rect(car_surface, window_color, (self.length - 8, 2, 6, self.width - 4))
        # Back window
        pygame.draw.rect(car_surface, window_color, (2, 2, 6, self.width - 4))
        # Side windows
        pygame.draw.rect(car_surface, window_color, (9, 1, self.length - 18, 3))
        pygame.draw.rect(car_surface, window_color, (9, self.width - 4, self.length - 18, 3))
        
        # Draw headlights
        if not self.brake_lights:
            pygame.draw.circle(car_surface, (255, 255, 200), (self.length - 2, 3), 2)
            pygame.draw.circle(car_surface, (255, 255, 200), (self.length - 2, self.width - 3), 2)
        
        # Draw brake lights
        if self.brake_lights:
            pygame.draw.circle(car_surface, RED, (2, 3), 2)
            pygame.draw.circle(car_surface, RED, (2, self.width - 3), 2)
        
        # Draw turn indicators
        if self.indicator_on:
            if self.left_indicator:
                # Left indicators (orange)
                pygame.draw.circle(car_surface, ORANGE, (self.length - 4, 1), 2)
                pygame.draw.circle(car_surface, ORANGE, (4, 1), 2)
            if self.right_indicator:
                # Right indicators (orange)
                pygame.draw.circle(car_surface, ORANGE, (self.length - 4, self.width - 1), 2)
                pygame.draw.circle(car_surface, ORANGE, (4, self.width - 1), 2)
        
        # Rotate the car surface
        rotated_surface = pygame.transform.rotate(car_surface, self.angle)
        rotated_rect = rotated_surface.get_rect(center=original_center)
        
        # Draw the rotated car
        screen.blit(rotated_surface, rotated_rect)

class AdvancedTrafficSimulation:
    def __init__(self):
        self.width = 1400
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ADVANCED Traffic - Lights, Turning, Indicators")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 18)
        
        # Intersections
        self.intersections = [
            (350, 400),
            (700, 400),
            (1050, 400)
        ]
        
        # Traffic lights
        self.traffic_lights = [TrafficLight(x, y) for x, y in self.intersections]
        
        # Vehicles
        self.vehicles = []
        self.max_vehicles = 25
        self.spawn_timer = 0
        self.spawn_delay = 60
        
        # Control
        self.paused = False
        self.show_stats = True
        
    def spawn_vehicle(self):
        """Spawn vehicles at road edges"""
        if len(self.vehicles) >= self.max_vehicles:
            return
        
        # Spawn points at road edges
        spawn_configs = [
            (50, 400, Direction.EAST),   # Left edge going east
            (1350, 400, Direction.WEST), # Right edge going west
            (350, 50, Direction.SOUTH),  # Top edges going south
            (700, 50, Direction.SOUTH),
            (1050, 50, Direction.SOUTH),
            (350, 750, Direction.NORTH), # Bottom edges going north
            (700, 750, Direction.NORTH),
            (1050, 750, Direction.NORTH),
        ]
        
        # Random spawn point
        x, y, direction = random.choice(spawn_configs)
        
        # Add some randomness to position
        if direction in [Direction.EAST, Direction.WEST]:
            y += random.randint(-30, 30)
        else:
            x += random.randint(-30, 30)
        
        # Check if spawn point is clear
        for vehicle in self.vehicles:
            if math.sqrt((vehicle.x - x)**2 + (vehicle.y - y)**2) < 50:
                return
        
        vehicle = Vehicle(x, y, direction)
        self.vehicles.append(vehicle)
    
    def update(self):
        """Update simulation"""
        if not self.paused:
            # Update traffic lights
            for light in self.traffic_lights:
                light.update()
            
            # Update vehicles
            for vehicle in self.vehicles:
                vehicle.update(self.vehicles, self.traffic_lights, self.intersections)
            
            # Remove vehicles that left screen
            self.vehicles = [v for v in self.vehicles 
                           if -100 < v.x < self.width + 100 and -100 < v.y < self.height + 100]
            
            # Spawn new vehicles
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_delay:
                self.spawn_timer = 0
                self.spawn_vehicle()
    
    def draw_road_network(self):
        """Draw roads with lanes"""
        # Horizontal main road
        pygame.draw.rect(self.screen, ROAD_GRAY, (0, 350, self.width, 100))
        
        # Vertical roads at intersections
        for x, _ in self.intersections:
            pygame.draw.rect(self.screen, ROAD_GRAY, (x - 50, 0, 100, self.height))
        
        # Lane markings - horizontal road
        for x in range(0, self.width, 40):
            pygame.draw.rect(self.screen, LANE_MARKING, (x, 375, 20, 3))
            pygame.draw.rect(self.screen, LANE_MARKING, (x, 425, 20, 3))
        
        # Center divider - horizontal road
        pygame.draw.rect(self.screen, DIVIDER_YELLOW, (0, 398, self.width, 2))
        pygame.draw.rect(self.screen, DIVIDER_YELLOW, (0, 402, self.width, 2))
        
        # Lane markings - vertical roads
        for intersection_x, _ in self.intersections:
            for y in range(0, self.height, 40):
                pygame.draw.rect(self.screen, LANE_MARKING, (intersection_x - 25, y, 3, 20))
                pygame.draw.rect(self.screen, LANE_MARKING, (intersection_x + 25, y, 3, 20))
            
            # Center dividers - vertical
            pygame.draw.rect(self.screen, DIVIDER_YELLOW, (intersection_x - 2, 0, 2, self.height))
            pygame.draw.rect(self.screen, DIVIDER_YELLOW, (intersection_x + 2, 0, 2, self.height))
        
        # Crosswalks at intersections
        for x, y in self.intersections:
            # Horizontal crosswalks
            for offset in range(-40, 40, 8):
                pygame.draw.rect(self.screen, WHITE, (x - 50, y + offset, 100, 4))
            # Vertical crosswalks
            for offset in range(-40, 40, 8):
                pygame.draw.rect(self.screen, WHITE, (x + offset, y - 50, 4, 100))
    
    def draw(self):
        """Draw everything"""
        # Background
        self.screen.fill((40, 100, 40))
        
        # Draw road network
        self.draw_road_network()
        
        # Draw traffic lights
        for light in self.traffic_lights:
            light.draw(self.screen)
        
        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(self.screen)
        
        # Draw UI
        self.draw_ui()
    
    def draw_ui(self):
        """Draw user interface"""
        if self.show_stats:
            # Title
            title = self.font.render("ADVANCED Traffic System", True, WHITE)
            title_bg = pygame.Surface((title.get_width() + 20, title.get_height() + 10))
            title_bg.fill(BLACK)
            title_bg.set_alpha(200)
            self.screen.blit(title_bg, (10, 10))
            self.screen.blit(title, (20, 15))
            
            # Features list
            features = [
                f"Vehicles: {len(self.vehicles)}/{self.max_vehicles}",
                "",
                "INSANE Features:",
                "✓ Traffic Lights",
                "✓ Turn Indicators",
                "✓ Smooth Turning",
                "✓ Lane Discipline",
                "✓ Light Obedience",
                "✓ Brake Lights",
                "✓ Smart Decisions",
                "",
                "Watch cars:",
                "• Stop at red lights",
                "• Use turn signals",
                "• Turn at intersections",
                "• Maintain safe distance"
            ]
            
            y = 60
            for feature in features:
                if feature.startswith("✓"):
                    color = GREEN
                elif feature.startswith("•"):
                    color = YELLOW
                elif feature == "INSANE Features:":
                    color = ORANGE
                else:
                    color = WHITE
                    
                if feature:
                    text = self.small_font.render(feature, True, color)
                    text_bg = pygame.Surface((280, text.get_height() + 4))
                    text_bg.fill(BLACK)
                    text_bg.set_alpha(200)
                    self.screen.blit(text_bg, (10, y - 2))
                    self.screen.blit(text, (20, y))
                y += 22
            
            # Controls
            controls = [
                "Controls:",
                "SPACE - Pause",
                "+/- - Traffic",
                "S - Stats",
                "ESC - Exit"
            ]
            
            y = self.height - 130
            for control in controls:
                text = self.small_font.render(control, True, WHITE)
                text_bg = pygame.Surface((150, text.get_height() + 4))
                text_bg.fill(BLACK)
                text_bg.set_alpha(200)
                self.screen.blit(text_bg, (10, y - 2))
                self.screen.blit(text, (20, y))
                y += 24
    
    def handle_event(self, event):
        """Handle user input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_s:
                self.show_stats = not self.show_stats
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.max_vehicles = min(40, self.max_vehicles + 5)
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
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    sim = AdvancedTrafficSimulation()
    sim.run()