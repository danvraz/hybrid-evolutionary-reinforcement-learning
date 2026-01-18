import pygame

class MazeRenderer:
    def __init__(self, env):
        self.env = env
        self.cell = env.cell_size

        self.width_px = env.width * self.cell
        self.height_px = env.height * self.cell

        self.screen = pygame.display.set_mode(
            (self.width_px, self.height_px)
        )
        pygame.display.set_caption("Maze Agent")

        self.colors = {
            "bg": (30, 30, 30),
            "wall": (230, 230, 230),
            "start": (0, 200, 0),
            "goal": (200, 0, 0),
            "agent": (50, 100, 255),
        }

    # =====================
    # DRAW MAZE
    # =====================
    def draw(self):
        self.screen.fill(self.colors["bg"])

        for y in range(self.env.height):
            for x in range(self.env.width):
                rect = pygame.Rect(
                    x * self.cell,
                    y * self.cell,
                    self.cell,
                    self.cell
                )

                if self.env.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, self.colors["wall"], rect)

        sx, sy = self.env.start
        gx, gy = self.env.goal

        pygame.draw.rect(
            self.screen,
            self.colors["start"],
            (sx * self.cell, sy * self.cell, self.cell, self.cell)
        )

        pygame.draw.rect(
            self.screen,
            self.colors["goal"],
            (gx * self.cell, gy * self.cell, self.cell, self.cell)
        )

    # =====================
    # DRAW AGENT
    # =====================
    def draw_agent(self, agent):
        pygame.draw.circle(
            self.screen,
            self.colors["agent"],
            (int(agent.x), int(agent.y)),
            agent.radius
        )

        pygame.display.flip()
