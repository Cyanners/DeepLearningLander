import random

import pygame
import math
from misc import LanderObj, generate_terrain, draw_terrain, draw_landing_pad


class LunarLanderEnv:

    def __init__(self, screen_x, screen_y, terrain_seed):

        pygame.init()

        self.rendering = False
        self.screen_x, self.screen_y = screen_x, screen_y
        self.screen = pygame.Surface((self.screen_x, self.screen_y))

        self.clock = pygame.time.Clock()
        #self.gravity = 0.0162 # Moon gravity / 100
        self.gravity = 0.162 # Moon gravity / 10

        # Terrain generation and masking
        self.segment_width = 5
        self.pad_width = 150
        self.terrain_points, self.landing_pad = generate_terrain(
            self.screen, self.segment_width, self.pad_width, terrain_seed,75.0, 5, 0.6, 2.0
        )
        self.terrain_surface = pygame.Surface((self.screen_x, self.screen_y), pygame.SRCALPHA)
        pygame.draw.polygon(self.terrain_surface, (125, 125, 125), self.terrain_points)
        self.terrain_mask = pygame.mask.from_surface(self.terrain_surface)

        self.action_space = 4  # 0: do nothing, 1: up, 2: left, 3: right

        self.lander = LanderObj(40, 100, 100, 0.35, 0.25)
        self.start_time = pygame.time.get_ticks()
        self.elapsed_ms = 0


    def set_rendering(self, rendering: bool):
        self.rendering = rendering
        if rendering:
            self.screen = pygame.display.set_mode((self.screen_x, self.screen_y))
        else:
            self.screen = pygame.Surface((self.screen_x, self.screen_y))


    def reset(self, start_pos_x, start_pos_y):
        # self.lander.reset(random.randint(50, 750), random.randint(50, 150))
        self.lander.reset(start_pos_x, start_pos_y)
        self.start_time = pygame.time.get_ticks()
        self.elapsed_ms = 0
        return self.lander.get_state()


    def step(self, action):
        if self.lander.fuel > 0:
            if action == 1:
                self.lander.thrust_up()
            elif action == 2:
                self.lander.thrust_left()
            elif action == 3:
                self.lander.thrust_right()
        else:
            self.lander.fuel = 0.0

        self.lander.update(self.screen, self.gravity, self.terrain_mask, self.landing_pad, self.terrain_points, self.segment_width)

        # Calculate elapsed time
        self.elapsed_ms = pygame.time.get_ticks() - self.start_time

        lander_pad_accuracy = None



        reward = 0
        done = False

        # Distance to pad
        distance_to_pad = abs(self.landing_pad.centerx - self.lander.x)
        max_distance = self.screen_x / 2
        distance_factor = max(0, 1 - distance_to_pad / max_distance)
        distance_reward = distance_factor * 1.5

        # Velocity penalty (encourage slow, controlled movement)
        v_total = math.sqrt(self.lander.vx ** 2 + self.lander.vy ** 2)
        if v_total > 3.0:
            velocity_penalty = -(v_total - 3.0) * 0.2  # Only penalize above threshold
        else:
            velocity_penalty = 0  # No penalty for reasonable speeds

        # Height reward (encourage staying at reasonable altitude)
        height_reward = -abs(self.lander.terrain_range - 50) * 0.01

        if self.lander.landed:
            # Successful landing bonus
            reward = 100.0

            # Accuracy bonus
            effective_half_pad_width = (self.landing_pad.width - self.lander.size) / 2
            accuracy = max(0, (effective_half_pad_width - distance_to_pad) / effective_half_pad_width)
            reward += accuracy * 50

            # Fuel efficiency bonus
            fuel_bonus = (self.lander.fuel / 100.0) * 20
            reward += fuel_bonus

            # Velocity bonus (soft landing)
            if v_total < 2.0:
                reward += 20

            distance_to_pad_center = abs(self.landing_pad.centerx - self.lander.x)
            effective_half_pad_width = ((self.landing_pad.width - self.lander.size) / 2)
            lander_pad_accuracy = ((effective_half_pad_width - distance_to_pad_center) / effective_half_pad_width) * 100

            done = True

        elif self.lander.crashed:
            reward = -30.0  # Crash penalty

            # Less penalty if crashed near the pad
            if self.lander.above_pad or distance_to_pad < 50:
                reward = -15.0

            done = True

        else:
            # Continuous rewards during flight
            reward = distance_reward + velocity_penalty + height_reward

            if self.lander.above_pad:
                # Base reward for being above pad
                reward += 0.5

                if self.lander.vy > 0:
                    if 0.1 >= self.lander.vy > 0:  # Slow descent
                        reward += 0.2  # Small reward for any descent
                    elif 0.25 >= self.lander.vy > 0.1:
                        reward += 0.5
                    elif 0.5 >= self.lander.vy > 0.25:
                        reward += 0.8
                    elif 2.0 >= self.lander.vy > 0.5:
                        reward += 1.5

                    # Extra bonus for controlled descent
                    if 0.8 <= self.lander.vy <= 2.0 and abs(self.lander.vx) < 1.0 and v_total < 2.0:
                        reward += 1.5
                else:
                    reward -= 1.0

            # Small fuel efficiency reward
            reward -= 0.5  # Small constant penalty for time/fuel





        # Return info dictionary
        info = {
            "landed": self.lander.landed,
            "crashed": self.lander.crashed,
            "elapsed_ms": self.elapsed_ms,
            "landing_accuracy": lander_pad_accuracy
        }

        return self.lander.get_state(), reward, done, info


    def render(self, text_wait):
        self.screen.fill((0, 0, 0))

        draw_terrain(self.screen, self.terrain_surface)
        draw_landing_pad(self.screen, (0, 255, 0), self.landing_pad)

        self.lander.draw(self.screen)
        self.lander.draw_text(self.screen)
        # self.lander.terrain_ranging_line(self.terrain_points, self.segment_width, self.screen)

        self.elapsed_ms = pygame.time.get_ticks() - self.start_time
        seconds = self.elapsed_ms // 1000
        milliseconds = self.elapsed_ms % 1000
        timer_text = f"Time: {seconds}.{milliseconds:03d} s"
        font = pygame.font.SysFont(None, 32)
        self.screen.blit(font.render(timer_text, True, (255, 255, 0)), (10, 10))

        if self.lander.landed or self.lander.crashed:
            if self.lander.landed:
                message = "LANDED!"
                color = (0, 255, 0)

                distance_to_pad_center = abs(self.landing_pad.centerx - self.lander.x)
                effective_half_pad_width = ((self.landing_pad.width - self.lander.size) / 2)
                lander_pad_accuracy = ((effective_half_pad_width - distance_to_pad_center) / effective_half_pad_width) * 100
                accuracy_message = f"Accuracy: {lander_pad_accuracy:.2f}%"
                accuracy_surface = font.render(accuracy_message, True, color)
                accuracy_rect = accuracy_surface.get_rect(center=(self.screen_x // 2, 200))
                self.screen.blit(accuracy_surface, accuracy_rect)
            else:
                message = "CRASHED!"
                color = (255, 0, 0)

            end_font = pygame.font.SysFont(None, 96)
            text_surface = end_font.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.screen_x // 2, 150))
            self.screen.blit(text_surface, text_rect)

            pygame.display.flip()
            if text_wait:
                pygame.time.wait(2000)

        pygame.display.flip()
