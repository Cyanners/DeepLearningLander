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
        self.state_dim = 6     # [x, y, vx, vy, fuel, distance_to_pad_center]

        #self.lander = LanderObj(40, 100, 100, 0.035, 0.025)
        self.lander = LanderObj(40, 100, 100, 0.35, 0.25)
        self.start_time = pygame.time.get_ticks()
        self.elapsed_ms = 0


    def set_rendering(self, rendering: bool):
        self.rendering = rendering
        if rendering:
            self.screen = pygame.display.set_mode((self.screen_x, self.screen_y))
        else:
            self.screen = pygame.Surface((self.screen_x, self.screen_y))


    def reset(self):
        self.lander.reset(100, 100)
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

        reward = 0
        done = False

        lander_pad_accuracy = None

        if self.lander.landed:
            # CALCULATE WEIGHTED REWARD FROM TIME, FUEL, ACCURACY HERE

            # Calculate distance to pad center for accuracy
            distance_to_pad_center = abs(self.landing_pad.centerx - self.lander.x)
            effective_half_pad_width = ((self.landing_pad.width - self.lander.size) / 2)
            lander_pad_accuracy = ((effective_half_pad_width - distance_to_pad_center) / effective_half_pad_width) * 100

            reward = 10000.0
            done = True
        elif self.lander.crashed:
            v_total = math.sqrt(self.lander.vx ** 2 + self.lander.vy ** 2)
            reward = - self.lander.pad_dist / 100
            #if v_total > 1.0:
            #    reward -= 10.0

            if self.lander.above_pad:
                reward += 100
                if self.lander.vx > 2 or self.lander.vx < -2:
                    reward -= 50

            done = True
        else:
            # Shaping: small negative reward for fuel/time usage
            # reward -= 0.00025

            # Encourage being above pad
            if self.lander.above_pad:
                if 0.5 > self.lander.vx > -0.5 and 0 < self.lander.vy < 1:
                    reward += 10
                else:
                    reward += 0.01

            v_total = math.sqrt(self.lander.vx ** 2 + self.lander.vy ** 2)
            if v_total > 5.0:
                reward -= 0.02

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
