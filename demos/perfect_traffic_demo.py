#!/usr/bin/env python3
"""
PERFECT Traffic Simulation - Proper lanes, manual traffic light control
Cars ACTUALLY follow lanes and obey traffic rules!
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
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 50)
ORANGE = (255, 140, 0)
BLUE = (70, 130, 255)
DARK_WINDOW = (40, 40, 60)
GRASS = (40, 100, 40)

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
]

class Direction(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

class Lane(Enum):
    """Specific lanes with exact positions"""
    # Horizontal lanes (y-position, direction)
    EAST_LANE_1 = (380, Direction.EAST)   # Top eastbound lane
    EAST_LANE_2 = (410, Direction.EAST)   # Bottom eastbound lane
    WEST_LANE_1 = (440, Direction.WEST)   # Top westbound lane
    WEST_LANE_2 = (470, Direction.WEST)   # Bottom westbound lane
    
    # Vertical lanes for each intersection (x-position, direction)
    # Intersection 1 (x=350)
    NORTH_LANE_1_INT1 = (320, Direction.NORTH)
    NORTH_LANE_2_INT1 = (340, Direction.NORTH)
    SOUTH_LANE_1_INT1 = (360, Direction.SOUTH)
    SOUTH_LANE_2_INT1 = (380, Direction.SOUTH)
    
    # Intersection 2 (x=700)
    NORTH_LANE_1_INT2 = (670, Direction.NORTH)
    NORTH_LANE_2_INT2 = (690, Direction.NORTH)
    SOUTH_LANE_1_INT2 = (710, Direction.SOUTH)
    SOUTH_LANE_2_INT2 = (730, Direction.SOUTH)
    
    # Intersection 3 (x=1050)
    NORTH_LANE_1_INT3 = (1020, Direction.NORTH)
    NORTH_LANE_2_INT3 = (1040, Direction.NORTH)
    SOUTH_LANE_1_INT3 = (1060, Direction.SOUTH)
    SOUTH_LANE_2_INT3 = (1080, Direction.SOUTH)

class TrafficLightState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

class TurnIntention(Enum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"

class TrafficLight:
    """Traffic light with manual control"""
    def __init__(self, x, y, intersection_id):
        self.x = x
        self.y = y
        self.id = intersection_id
        self.states = {
            'north_south': TrafficLightState.RED,
            'east_west': TrafficLightState.GREEN
        }
        self.manual_mode = False
        self.timer = 0
        self.green_duration = 240   # 4 seconds
        self.yellow_duration = 60   # 1 second
        self.current_phase = 'east_west'
    
    def toggle_manual_mode(self):
        """Toggle between manual and automatic mode"""
        self.manual_mode = not self.manual_mode
        return self.manual_mode
    
    def set_state(self, direction_group, state):
        """Manually set traffic light state"""
        if self.manual_mode:
            self.states[direction_group] = state
    
    def cycle_state(self, direction_group):
        """Cycle through states for manual control"""
        if self.manual_mode:
            current = self.states[direction_group]
            if current == TrafficLightState.RED:
                self.states[direction_group] = TrafficLightState.GREEN
            elif current == TrafficLightState.GREEN:
                self.states[direction_group] = TrafficLightState.YELLOW
            else:  # YELLOW
                self.states[direction_group] = TrafficLightState.RED
    
    def update(self):
        """Update traffic light (automatic mode only)"""
        if self.manual_mode:
            return
            
        self.timer += 1
        
        if self.current_phase == 'east_west':
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
        else:  # north_south phase
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
    
    def get_state_for_direction(self, direction):
        """Get light state for a specific direction"""
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return self.states['north_south']
        else:
            return self.states['east_west']
    
    def draw(self, screen, selected=False):
        """Draw traffic lights"""
        # Highlight if selected for manual control
        if selected and self.manual_mode:
            pygame.draw.circle(screen, YELLOW, (self.x, self.y), 80, 3)
        
        # Draw lights for each direction
        positions = [
            (self.x - 60, self.y, 'east_west'),    # West side
            (self.x + 60, self.y, 'east_west'),    # East side
            (self.x, self.y - 60, 'north_south'),  # North side
            (self.x, self.y + 60, 'north_south'),  # South side
        ]
        
        for px, py, direction_group in positions:
            state = self.states[direction_group]
            
            # Light housing
            housing_rect = pygame.Rect(px - 15, py - 30, 30, 60)
            pygame.draw.rect(screen, BLACK, housing_rect)
            pygame.draw.rect(screen, YELLOW if self.manual_mode else WHITE, housing_rect, 2)
            
            # Three lights
            colors = [
                LIGHT_RED if state == TrafficLightState.RED else LIGHT_OFF,
                LIGHT_YELLOW if state == TrafficLightState.YELLOW else LIGHT_OFF,
                LIGHT_GREEN if state == TrafficLightState.GREEN else LIGHT_OFF
            ]
            
            for i, color in enumerate(colors):
                y_offset = -20 + i * 20
                pygame.draw.circle(screen, color, (px, py + y_offset), 8)
                # Glow effect for active light
                if color != LIGHT_OFF:
                    for r in range(10, 15):
                        glow_surf = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surf, (*color, 30), (r, r), r)
                        screen.blit(glow_surf, (px - r, py + y_offset - r))

class Vehicle:
    """Vehicle that ACTUALLY stays in its lane"""
    def __init__(self, lane, x, y):
        self.lane = lane
        self.lane_y = lane.value[0] if 'EAST' in lane.name or 'WEST' in lane.name else y
        self.lane_x = lane.value[0] if 'NORTH' in lane.name or 'SOUTH' in lane.name else x
        self.x = x
        self.y = y
        self.direction = lane.value[1]
        self.speed = random.uniform(1.5, 2.5)
        self.current_speed = self.speed
        self.color = random.choice(CAR_COLORS)
        self.length = 24
        self.width = 14
        
        # Turning
        self.turn_intention = TurnIntention.STRAIGHT
        if random.random() < 0.3:  # 30% chance to turn
            self.turn_intention = random.choice([TurnIntention.LEFT, TurnIntention.RIGHT])
        
        # Indicators
        self.left_indicator = (self.turn_intention == TurnIntention.LEFT)
        self.right_indicator = (self.turn_intention == TurnIntention.RIGHT)
        self.indicator_timer = 0
        self.indicator_on = False
        
        # States
        self.waiting_at_light = False
        self.brake_lights = False
        self.is_turning = False
        self.turn_progress = 0
        self.turn_start_pos = None
        self.turn_end_pos = None
        self.turn_center = None
        self.original_lane = lane
    
    def update(self, vehicles, traffic_lights, intersections):
        """Update vehicle position"""
        # Update indicators
        if self.left_indicator or self.right_indicator:
            self.indicator_timer += 1
            if self.indicator_timer % 30 == 0:
                self.indicator_on = not self.indicator_on
        
        # Stay in lane (IMPORTANT!)
        if not self.is_turning:
            if self.direction in [Direction.EAST, Direction.WEST]:
                self.y = self.lane_y  # Lock to lane Y position
            else:
                self.x = self.lane_x  # Lock to lane X position
        
        # Check for traffic light ahead
        light_state = self.check_traffic_light(traffic_lights, intersections)
        
        # Check for vehicle ahead
        vehicle_ahead, distance = self.get_vehicle_ahead(vehicles)
        
        # Determine speed based on conditions
        self.current_speed = self.speed
        self.brake_lights = False
        
        # Traffic light logic
        if light_state:
            dist_to_intersection = self.distance_to_nearest_intersection(intersections)
            
            if light_state == TrafficLightState.RED:
                if dist_to_intersection < 80:
                    if dist_to_intersection < 20:
                        self.current_speed = 0
                        self.waiting_at_light = True
                    else:
                        self.current_speed = self.speed * (dist_to_intersection / 80)
                    self.brake_lights = True
                    
            elif light_state == TrafficLightState.YELLOW:
                if dist_to_intersection < 60:
                    if dist_to_intersection < 40:
                        # Too close to stop safely, go through
                        self.current_speed = self.speed
                    else:
                        # Slow down
                        self.current_speed = self.speed * 0.5
                        self.brake_lights = True
                        
            else:  # GREEN
                self.waiting_at_light = False
        
        # Vehicle ahead logic
        if vehicle_ahead and distance < 60:
            if distance < 30:
                self.current_speed = 0
                self.brake_lights = True
            else:
                self.current_speed = min(vehicle_ahead.current_speed, 
                                       self.speed * (distance / 60))
                self.brake_lights = True
        
        # Handle turning at intersections
        if not self.is_turning and self.turn_intention != TurnIntention.STRAIGHT:
            for intersection in intersections:
                if self.at_intersection(intersection) and not self.waiting_at_light:
                    self.start_turn(intersection)
                    break
        
        # Move vehicle
        if self.is_turning:
            self.perform_turn()
        elif self.current_speed > 0:
            dx, dy = self.direction.value
            self.x += dx * self.current_speed
            self.y += dy * self.current_speed
    
    def check_traffic_light(self, traffic_lights, intersections):
        """Check traffic light state ahead"""
        for light, intersection in zip(traffic_lights, intersections):
            if self.approaching_intersection(intersection):
                return light.get_state_for_direction(self.direction)
        return None
    
    def approaching_intersection(self, intersection):
        """Check if approaching intersection"""
        dist = abs(self.x - intersection[0]) + abs(self.y - intersection[1])
        if dist > 150:
            return False
        
        # Check if intersection is ahead
        dx = intersection[0] - self.x
        dy = intersection[1] - self.y
        
        if self.direction == Direction.NORTH and dy < 0:
            return abs(dx) < 100
        elif self.direction == Direction.SOUTH and dy > 0:
            return abs(dx) < 100
        elif self.direction == Direction.EAST and dx > 0:
            return abs(dy) < 100
        elif self.direction == Direction.WEST and dx < 0:
            return abs(dy) < 100
        return False
    
    def at_intersection(self, intersection):
        """Check if at intersection"""
        return (abs(self.x - intersection[0]) < 30 and 
                abs(self.y - intersection[1]) < 30)
    
    def distance_to_nearest_intersection(self, intersections):
        """Get distance to nearest intersection ahead"""
        min_dist = float('inf')
        for intersection in intersections:
            if self.approaching_intersection(intersection):
                dist = math.sqrt((self.x - intersection[0])**2 + 
                               (self.y - intersection[1])**2)
                min_dist = min(min_dist, dist)
        return min_dist
    
    def get_vehicle_ahead(self, vehicles):
        """Check for vehicle directly ahead IN SAME LANE"""
        min_dist = float('inf')
        closest = None
        
        for other in vehicles:
            if other == self:
                continue
            
            # Check if in same lane or turning
            if not self.is_turning and not other.is_turning:
                # Both not turning - check lane
                if self.direction in [Direction.EAST, Direction.WEST]:
                    # Horizontal movement - check Y position
                    if abs(self.y - other.y) > 10:  # Different lanes
                        continue
                else:
                    # Vertical movement - check X position  
                    if abs(self.x - other.x) > 10:  # Different lanes
                        continue
            
            # Check if ahead
            dx = other.x - self.x
            dy = other.y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if self.direction == Direction.EAST and dx > 0 and abs(dy) < 20:
                if dist < min_dist:
                    min_dist = dist
                    closest = other
            elif self.direction == Direction.WEST and dx < 0 and abs(dy) < 20:
                if dist < min_dist:
                    min_dist = dist
                    closest = other
            elif self.direction == Direction.NORTH and dy < 0 and abs(dx) < 20:
                if dist < min_dist:
                    min_dist = dist
                    closest = other
            elif self.direction == Direction.SOUTH and dy > 0 and abs(dx) < 20:
                if dist < min_dist:
                    min_dist = dist
                    closest = other
        
        return closest, min_dist
    
    def start_turn(self, intersection):
        """Start turning at intersection"""
        self.is_turning = True
        self.turn_progress = 0
        self.turn_start_pos = (self.x, self.y)
        
        # Calculate turn end position based on intention
        if self.turn_intention == TurnIntention.LEFT:
            if self.direction == Direction.NORTH:
                self.turn_end_pos = (intersection[0] - 40, intersection[1])
                self.direction = Direction.WEST
            elif self.direction == Direction.SOUTH:
                self.turn_end_pos = (intersection[0] + 40, intersection[1])
                self.direction = Direction.EAST
            elif self.direction == Direction.EAST:
                self.turn_end_pos = (intersection[0], intersection[1] - 40)
                self.direction = Direction.NORTH
            else:  # WEST
                self.turn_end_pos = (intersection[0], intersection[1] + 40)
                self.direction = Direction.SOUTH
        else:  # RIGHT turn
            if self.direction == Direction.NORTH:
                self.turn_end_pos = (intersection[0] + 40, intersection[1])
                self.direction = Direction.EAST
            elif self.direction == Direction.SOUTH:
                self.turn_end_pos = (intersection[0] - 40, intersection[1])
                self.direction = Direction.WEST
            elif self.direction == Direction.EAST:
                self.turn_end_pos = (intersection[0], intersection[1] + 40)
                self.direction = Direction.SOUTH
            else:  # WEST
                self.turn_end_pos = (intersection[0], intersection[1] - 40)
                self.direction = Direction.NORTH
    
    def perform_turn(self):
        """Execute smooth turn"""
        self.turn_progress += 0.05
        
        if self.turn_progress >= 1.0:
            # Turn complete
            self.is_turning = False
            self.x = self.turn_end_pos[0]
            self.y = self.turn_end_pos[1]
            self.left_indicator = False
            self.right_indicator = False
            self.turn_intention = TurnIntention.STRAIGHT
            
            # Update lane after turn
            if self.direction in [Direction.EAST, Direction.WEST]:
                self.lane_y = self.y
            else:
                self.lane_x = self.x
        else:
            # Smooth interpolation
            t = self.turn_progress
            # Use quadratic bezier curve for smooth turn
            if self.turn_intention == TurnIntention.LEFT:
                t = t * t  # Ease in
            else:
                t = 1 - (1 - t) * (1 - t)  # Ease out
            
            self.x = self.turn_start_pos[0] + (self.turn_end_pos[0] - self.turn_start_pos[0]) * t
            self.y = self.turn_start_pos[1] + (self.turn_end_pos[1] - self.turn_start_pos[1]) * t
    
    def draw(self, screen):
        """Draw vehicle"""
        # Car body
        if self.direction in [Direction.NORTH, Direction.SOUTH]:
            rect = pygame.Rect(self.x - self.width//2, self.y - self.length//2, 
                             self.width, self.length)
        else:
            rect = pygame.Rect(self.x - self.length//2, self.y - self.width//2,
                             self.length, self.width)
        
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)
        
        # Windows
        window_color = DARK_WINDOW
        if self.direction in [Direction.NORTH, Direction.SOUTH]:
            # Side windows
            pygame.draw.rect(screen, window_color, 
                           (rect.x + 2, rect.y + 4, rect.width - 4, 6))
            pygame.draw.rect(screen, window_color,
                           (rect.x + 2, rect.bottom - 10, rect.width - 4, 6))
        else:
            # Side windows
            pygame.draw.rect(screen, window_color,
                           (rect.x + 4, rect.y + 2, 6, rect.height - 4))
            pygame.draw.rect(screen, window_color,
                           (rect.right - 10, rect.y + 2, 6, rect.height - 4))
        
        # Brake lights
        if self.brake_lights:
            if self.direction == Direction.NORTH:
                pygame.draw.circle(screen, RED, (rect.x + 3, rect.bottom - 3), 2)
                pygame.draw.circle(screen, RED, (rect.right - 3, rect.bottom - 3), 2)
            elif self.direction == Direction.SOUTH:
                pygame.draw.circle(screen, RED, (rect.x + 3, rect.y + 3), 2)
                pygame.draw.circle(screen, RED, (rect.right - 3, rect.y + 3), 2)
            elif self.direction == Direction.EAST:
                pygame.draw.circle(screen, RED, (rect.x + 3, rect.y + 3), 2)
                pygame.draw.circle(screen, RED, (rect.x + 3, rect.bottom - 3), 2)
            else:  # WEST
                pygame.draw.circle(screen, RED, (rect.right - 3, rect.y + 3), 2)
                pygame.draw.circle(screen, RED, (rect.right - 3, rect.bottom - 3), 2)
        
        # Turn indicators
        if self.indicator_on:
            if self.left_indicator:
                if self.direction == Direction.NORTH:
                    pygame.draw.circle(screen, ORANGE, (rect.x, rect.centery), 3)
                elif self.direction == Direction.SOUTH:
                    pygame.draw.circle(screen, ORANGE, (rect.right, rect.centery), 3)
                elif self.direction == Direction.EAST:
                    pygame.draw.circle(screen, ORANGE, (rect.centerx, rect.y), 3)
                else:  # WEST
                    pygame.draw.circle(screen, ORANGE, (rect.centerx, rect.bottom), 3)
            
            if self.right_indicator:
                if self.direction == Direction.NORTH:
                    pygame.draw.circle(screen, ORANGE, (rect.right, rect.centery), 3)
                elif self.direction == Direction.SOUTH:
                    pygame.draw.circle(screen, ORANGE, (rect.x, rect.centery), 3)
                elif self.direction == Direction.EAST:
                    pygame.draw.circle(screen, ORANGE, (rect.centerx, rect.bottom), 3)
                else:  # WEST
                    pygame.draw.circle(screen, ORANGE, (rect.centerx, rect.y), 3)

class PerfectTrafficSimulation:
    def __init__(self):
        self.width = 1400
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("PERFECT Traffic - Manual Light Control + Proper Lanes")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Intersections
        self.intersections = [
            (350, 425),   # Intersection 1
            (700, 425),   # Intersection 2
            (1050, 425),  # Intersection 3
        ]
        
        # Traffic lights
        self.traffic_lights = [
            TrafficLight(x, y, i) for i, (x, y) in enumerate(self.intersections)
        ]
        self.selected_light = 0
        
        # Vehicles
        self.vehicles = []
        self.max_vehicles = 20
        self.spawn_timer = 0
        self.spawn_delay = 90
        
        # Control
        self.paused = False
        self.show_help = True
    
    def spawn_vehicle(self):
        """Spawn vehicle in specific lane"""
        if len(self.vehicles) >= self.max_vehicles:
            return
        
        # Spawn configurations (lane, x, y)
        spawn_configs = [
            # Eastbound lanes
            (Lane.EAST_LANE_1, 50, Lane.EAST_LANE_1.value[0]),
            (Lane.EAST_LANE_2, 50, Lane.EAST_LANE_2.value[0]),
            
            # Westbound lanes
            (Lane.WEST_LANE_1, 1350, Lane.WEST_LANE_1.value[0]),
            (Lane.WEST_LANE_2, 1350, Lane.WEST_LANE_2.value[0]),
            
            # Northbound lanes
            (Lane.NORTH_LANE_1_INT1, Lane.NORTH_LANE_1_INT1.value[0], 750),
            (Lane.NORTH_LANE_2_INT1, Lane.NORTH_LANE_2_INT1.value[0], 750),
            (Lane.NORTH_LANE_1_INT2, Lane.NORTH_LANE_1_INT2.value[0], 750),
            (Lane.NORTH_LANE_2_INT2, Lane.NORTH_LANE_2_INT2.value[0], 750),
            
            # Southbound lanes
            (Lane.SOUTH_LANE_1_INT1, Lane.SOUTH_LANE_1_INT1.value[0], 50),
            (Lane.SOUTH_LANE_2_INT1, Lane.SOUTH_LANE_2_INT1.value[0], 50),
            (Lane.SOUTH_LANE_1_INT2, Lane.SOUTH_LANE_1_INT2.value[0], 50),
            (Lane.SOUTH_LANE_2_INT2, Lane.SOUTH_LANE_2_INT2.value[0], 50),
        ]
        
        # Try random spawn point
        random.shuffle(spawn_configs)
        for lane, x, y in spawn_configs:
            # Check if clear
            clear = True
            for vehicle in self.vehicles:
                if math.sqrt((vehicle.x - x)**2 + (vehicle.y - y)**2) < 50:
                    clear = False
                    break
            
            if clear:
                vehicle = Vehicle(lane, x, y)
                self.vehicles.append(vehicle)
                break
    
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
    
    def draw_roads(self):
        """Draw road network with proper lanes"""
        # Background
        self.screen.fill(GRASS)
        
        # Main horizontal road
        pygame.draw.rect(self.screen, ROAD_GRAY, (0, 360, self.width, 140))
        
        # Vertical roads at intersections
        for x, _ in self.intersections:
            pygame.draw.rect(self.screen, ROAD_GRAY, (x - 70, 0, 140, self.height))
        
        # Center divider - horizontal
        pygame.draw.rect(self.screen, DIVIDER_YELLOW, (0, 428, self.width, 2))
        pygame.draw.rect(self.screen, DIVIDER_YELLOW, (0, 432, self.width, 2))
        
        # Lane markings - horizontal (between lanes going same direction)
        for x in range(0, self.width, 40):
            # Eastbound lanes divider
            pygame.draw.rect(self.screen, LANE_MARKING, (x, 395, 20, 2))
            # Westbound lanes divider
            pygame.draw.rect(self.screen, LANE_MARKING, (x, 455, 20, 2))
        
        # Vertical roads
        for intersection_x, _ in self.intersections:
            # Center divider - vertical
            pygame.draw.rect(self.screen, DIVIDER_YELLOW, 
                           (intersection_x - 2, 0, 2, self.height))
            pygame.draw.rect(self.screen, DIVIDER_YELLOW,
                           (intersection_x + 2, 0, 2, self.height))
            
            # Lane markings - vertical
            for y in range(0, self.height, 40):
                # Northbound lanes divider
                pygame.draw.rect(self.screen, LANE_MARKING,
                               (intersection_x - 35, y, 2, 20))
                # Southbound lanes divider
                pygame.draw.rect(self.screen, LANE_MARKING,
                               (intersection_x + 35, y, 2, 20))
        
        # Crosswalks
        for x, y in self.intersections:
            for offset in range(-60, 60, 10):
                pygame.draw.rect(self.screen, WHITE, (x - 70, y + offset, 140, 5))
                pygame.draw.rect(self.screen, WHITE, (x + offset, y - 70, 5, 140))
    
    def draw(self):
        """Draw everything"""
        # Draw roads
        self.draw_roads()
        
        # Draw traffic lights
        for i, light in enumerate(self.traffic_lights):
            light.draw(self.screen, selected=(i == self.selected_light))
        
        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(self.screen)
        
        # Draw UI
        self.draw_ui()
    
    def draw_ui(self):
        """Draw user interface"""
        if self.show_help:
            # Help panel
            help_lines = [
                "TRAFFIC LIGHT CONTROL:",
                f"Selected: Intersection {self.selected_light + 1}",
                f"Mode: {'MANUAL' if self.traffic_lights[self.selected_light].manual_mode else 'AUTO'}",
                "",
                "CONTROLS:",
                "1/2/3 - Select intersection",
                "M - Toggle Manual/Auto",
                "↑/↓ - Change N/S lights",
                "←/→ - Change E/W lights",
                "SPACE - Pause",
                "+/- - Traffic density",
                "H - Hide help",
                "",
                "LIGHT RULES:",
                "🔴 RED = STOP",
                "🟡 YELLOW = SLOW",
                "🟢 GREEN = GO",
                "",
                f"Vehicles: {len(self.vehicles)}/{self.max_vehicles}"
            ]
            
            # Background for help
            help_bg = pygame.Surface((250, len(help_lines) * 22 + 20))
            help_bg.fill(BLACK)
            help_bg.set_alpha(200)
            self.screen.blit(help_bg, (10, 10))
            
            # Draw help text
            y = 20
            for line in help_lines:
                if "MANUAL" in line and self.traffic_lights[self.selected_light].manual_mode:
                    color = YELLOW
                elif line.startswith("🔴"):
                    color = RED
                elif line.startswith("🟡"):
                    color = YELLOW
                elif line.startswith("🟢"):
                    color = GREEN
                else:
                    color = WHITE
                
                text = self.small_font.render(line, True, color)
                self.screen.blit(text, (20, y))
                y += 22
        else:
            # Minimal UI
            text = self.small_font.render("Press H for help", True, WHITE)
            text_bg = pygame.Surface((150, 25))
            text_bg.fill(BLACK)
            text_bg.set_alpha(200)
            self.screen.blit(text_bg, (10, 10))
            self.screen.blit(text, (20, 15))
    
    def handle_event(self, event):
        """Handle user input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
            
            # Select intersection
            elif event.key == pygame.K_1:
                self.selected_light = 0
            elif event.key == pygame.K_2:
                self.selected_light = 1
            elif event.key == pygame.K_3:
                self.selected_light = 2
            
            # Toggle manual mode
            elif event.key == pygame.K_m:
                is_manual = self.traffic_lights[self.selected_light].toggle_manual_mode()
                print(f"Intersection {self.selected_light + 1}: {'MANUAL' if is_manual else 'AUTO'} mode")
            
            # Control lights manually
            elif event.key == pygame.K_UP:
                self.traffic_lights[self.selected_light].cycle_state('north_south')
            elif event.key == pygame.K_DOWN:
                self.traffic_lights[self.selected_light].cycle_state('north_south')
            elif event.key == pygame.K_LEFT:
                self.traffic_lights[self.selected_light].cycle_state('east_west')
            elif event.key == pygame.K_RIGHT:
                self.traffic_lights[self.selected_light].cycle_state('east_west')
            
            # Traffic density
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.max_vehicles = min(30, self.max_vehicles + 5)
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
    sim = PerfectTrafficSimulation()
    sim.run()