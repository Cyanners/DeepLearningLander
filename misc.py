import pygame
import random
import math
from noise import pnoise1


def myround(x, base):
    return base * round(x/base)


class LanderObj:
    def __init__(self, size, x, y, thrust_v, thrust_h):

        if pygame.display.get_surface(): self.image = pygame.image.load("lander.png").convert_alpha()
        else: self.image = pygame.image.load("lander.png")
        self.image = pygame.transform.scale(self.image, (size, size))

        self.size = size
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.thrust_v = thrust_v
        self.thrust_h = thrust_h
        self.crashed = False
        self.landed = False
        self.particles = []
        self.fuel = 100.0
        self.pad_dist = 9999
        self.pad_dist_x = 9999
        self.terrain_range = 9999
        self.above_pad = False

        # Lander edges
        self.lander_left = self.x - self.size / 2
        self.lander_right = self.x + self.size / 2
        self.lander_top = self.y - self.size / 2
        self.lander_bot = self.y + self.size / 2

        # Create surface and mask for collision
        self.surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(self.surface, (255, 255, 255), (0, 0, size, size))
        self.mask = pygame.mask.from_surface(self.surface)


    def update(self, screen, gravity, terrain_mask, landing_pad, terrain_points, segment_width):
        self.vy += gravity
        self.x += self.vx
        self.y += self.vy

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # Get height above terrain
        self.terrain_range = self.terrain_ranging(terrain_points, segment_width)

        # Check if we are above the pad
        self.above_pad = self.above_pad_check(landing_pad)

        # Calculate distance to landing pad
        self.pad_dist = math.sqrt((self.x - landing_pad.centerx) ** 2 + (self.y - landing_pad.centery) ** 2)
        self.pad_dist_x = self.x - landing_pad.centerx

        # Recalculate lander edges
        self.lander_left = self.x - self.size / 2
        self.lander_right = self.x + self.size / 2
        self.lander_top = self.y - self.size / 2
        self.lander_bot = self.y + self.size / 2

        # Collision check
        offset = (int(self.x - self.size / 2), int(self.y - self.size / 2))

        # If there has been a collision with the terrain...
        if terrain_mask.overlap(self.mask, offset):

            # Get coordinates of top corners of landing pad
            pad_x1, pad_y = landing_pad.topleft
            pad_x2 = pad_x1 + landing_pad.width

            # Get magnitude of lander velocity
            v_total = math.sqrt(self.vx ** 2 + self.vy ** 2)

            # Check if the collision was with the pad and impact speed was not excessive
            if self.lander_left >= pad_x1 and self.lander_right <= pad_x2 and v_total < 1.0:
                self.y = (pad_y - self.size / 2) + 1 # Extra pixel to retrigger collision logic
                self.vy = 0
                # self.vx = 0 # No sliding
                if self.vx > 0.01 or self.vx < -0.01:
                    self.vx -= self.vx / 25
                else:
                    # Successful landing
                    self.vx = 0
                    self.landed = True
            else:
                self.crashed = True

        # Out of bounds check
        screen_width, screen_height = screen.get_size()
        if self.lander_left < 0 or self.lander_right > screen_width or self.lander_top < 0:
            self.crashed = True


    def terrain_ranging(self, terrain_points, segment_width):
        nearest_point_x = myround(self.x, segment_width)
        nearest_point_index = nearest_point_x // 5
        nearest_point = terrain_points[nearest_point_index]
        terrain_range = nearest_point[1] - self.lander_bot
        return terrain_range


    def terrain_ranging_line(self, terrain_points, segment_width, screen):
        nearest_point_x = myround(self.x, segment_width)
        nearest_point_index = nearest_point_x // 5
        nearest_point = terrain_points[nearest_point_index]
        self.lander_bot = self.y + self.size / 2
        pygame.draw.line(screen,(255, 0, 0),(nearest_point_x, self.lander_bot), (nearest_point_x, nearest_point[1]),2)


    def above_pad_check(self, landing_pad):
        pad_x1, pad_y = landing_pad.topleft
        pad_x2 = pad_x1 + landing_pad.width
        if self.lander_left >= pad_x1 and self.lander_right <= pad_x2:
            return True
        else:
            return False


    def spawn_particle(self, offset_x, offset_y, vx, vy):
        px = self.x + offset_x
        py = self.y + offset_y
        pvx = vx
        pvy = vy
        plf = random.uniform(8, 20)
        p_col = (random.randint(200, 255), random.randint(50, 150), 0)
        self.particles.append(Particle(px, py, pvx, pvy, plf, p_col))


    def thrust_up(self):
        self.vy -= self.thrust_v
        self.fuel -= 0.25
        pvx = random.uniform(-0.5, 0.5)
        pvy = random.uniform(3, 4.5)
        self.spawn_particle(0, 12, pvx, pvy)


    def thrust_left(self):
        self.vx -= self.thrust_h
        self.fuel -= 0.25
        pvx = random.uniform(3, 4.5)
        pvy = random.uniform(-0.5, 0.5)
        self.spawn_particle(15, -8, pvx, pvy)


    def thrust_right(self):
        self.vx += self.thrust_h
        self.fuel -= 0.25
        pvx = random.uniform(-3, -4.5)
        pvy = random.uniform(-0.5, 0.5)
        self.spawn_particle(-15, -8, pvx, pvy)


    def draw(self, screen):
        # Draw particles first so they're behind the lander
        for p in self.particles:
            p.draw(screen)

        screen.blit(self.image, (self.x - self.size / 2, self.y - self.size / 2))


    def draw_text(self, screen):
        font = pygame.font.SysFont(None, 32)

        fuel_text = f"Fuel: {self.fuel:.2f}%"
        fuel_col = ((100 - self.fuel) * 2.55, 255 - (100 - self.fuel) * 2.55, 0)
        fuel_surface = font.render(fuel_text, True, fuel_col)
        screen.blit(fuel_surface, (10, 42))

        v_total = math.sqrt(self.vx ** 2 + self.vy ** 2)
        vel_text = f"Velo: {v_total:.2f}"
        if v_total < 1.0:
            v_color = (0, 255, 0)
        else:
            v_color = (255, 0, 0)
        vel_surface = font.render(vel_text, True, v_color)
        screen.blit(vel_surface, (10, 74))


    def get_state(self):
        return (
            # self.x,
            # self.y,
            self.vx,
            self.vy,
            self.terrain_range,
            self.above_pad,
            self.pad_dist_x
            # self.fuel
        )


    def reset(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.fuel = 100.0
        self.landed = False
        self.crashed = False
        self.terrain_range = 9999
        self.above_pad = False
        self.particles.clear()


def generate_terrain(screen, segment_width, pad_width, terrain_seed, scale=100.0, octaves=4, persistence=0.5, lacunarity=2.0):
    screen_width = screen.get_width()
    screen_height = screen.get_height()

    points = []
    num_points = (screen_width // segment_width) + 1

    # Landing pad placement
    rng = random.Random(terrain_seed)
    pad_start_x = myround(rng.randint(0, screen_width - pad_width), segment_width)
    pad_end_x = pad_start_x + pad_width
    pad_height = 0

    # Random offset to make the terrain different each time
    # noise_offset = random.uniform(0, 1000)
    noise_offset = terrain_seed

    for i in range(num_points):
        x = i * segment_width

        if pad_start_x <= x <= pad_end_x:
            # Flat height for landing pad
            y = pad_height
        else:
            # Generate Perlin-based height
            noise_val = pnoise1((i + noise_offset) / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeat=999999)
            # Map Perlin output (-1 to 1) to screen height range
            y = screen_height - int(((noise_val + 1.5) / 2) * screen_height - 150) + 150
            if x <= pad_start_x:
                pad_height = y

        points.append((x, y))

    # Close the polygon at the bottom
    points.append((screen_width, screen_height))
    points.append((0, screen_height))

    return points, pygame.Rect(pad_start_x, pad_height, pad_width, 20)


def draw_terrain(screen, terrain_surface):
    screen.blit(terrain_surface, (0, 0))


def draw_landing_pad(screen, pad_color, landing_pad):
    pygame.draw.rect(screen, pad_color, landing_pad)


class Particle:
    def __init__(self, x, y, vx, vy, lifetime, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.lifetime = lifetime
        self.color = color
        self.age = 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        # Slowly fade color
        fade = max(0, 255 - int((self.age / self.lifetime) * 255))
        self.color = (self.color[0], self.color[1], self.color[2], fade)
        return self.age < self.lifetime  # still alive?

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), 2)