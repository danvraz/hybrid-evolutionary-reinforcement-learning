"""
env/maze_env.py

Pure task definition for 2D continuous maze navigation.

This module defines ONLY the geometry and physics constraints.
It contains NO reward logic, NO learning, NO policy, NO episode evaluation.

Responsibilities:
- Procedural maze generation
- Collision detection (agent vs walls)
- Ray-casting (for agent sensors)
- Position sampling (start/goal placement)

The environment is deterministic given a seed and provides query
functions for external evaluators and agents.
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque


class MazeEnv:
    """
    Continuous 2D maze environment with procedural generation.
    
    The maze is represented as a binary grid (0=free, 1=wall) but
    exists in continuous (x, y) coordinate space via cell_size scaling.
    
    This class provides:
    - Deterministic maze generation
    - Collision queries for agent physics
    - Ray-casting for agent sensors
    - Start/goal position sampling
    
    This class does NOT provide:
    - Reward computation
    - Policy evaluation
    - Learning updates
    - Episode management beyond reset
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        cell_size: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize maze environment.
        
        Args:
            width: Number of grid cells horizontally
            height: Number of grid cells vertically
            cell_size: Size of each grid cell in continuous coordinates (meters)
            seed: Random seed for reproducible generation
        """
        assert width > 0 and height > 0, "Maze dimensions must be positive"
        assert cell_size > 0, "Cell size must be positive"
        
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Initialize random number generator
        self.rng = np.random.RandomState(seed)
        
        # Maze grid: 0 = free space, 1 = wall
        # Will be populated by reset()
        self.grid = None
        
        # Episode state (populated by reset)
        self.start_pos = None
        self.goal_pos = None
    
    def reset(self) -> None:
        """
        Generate new maze and sample start/goal positions.
        
        Uses depth-first search to create a perfect maze (single solution path).
        Then samples spatially distant start and goal positions.
        """
        # Generate maze structure
        self.grid = self._generate_maze_dfs()
        
        # Sample start position
        self.start_pos = self.sample_free_position()
        
        # Sample goal position (far from start)
        self.goal_pos = self._sample_distant_position(self.start_pos)
    
    def _generate_maze_dfs(self) -> np.ndarray:
        """
        Generate maze using depth-first search (DFS) algorithm.
        
        Creates a perfect maze (all cells reachable, no loops) with walls
        forming a connected structure.
        
        Returns:
            grid: Binary array (height, width) where 1=wall, 0=free
        """
        # Initialize grid: all walls
        grid = np.ones((self.height, self.width), dtype=np.uint8)
        
        # Start from random cell
        start_x = self.rng.randint(0, self.width)
        start_y = self.rng.randint(0, self.height)
        
        # Mark starting cell as free
        grid[start_y, start_x] = 0
        
        # DFS stack: (x, y) positions
        stack = [(start_x, start_y)]
        
        # Cardinal directions: right, down, left, up
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        while stack:
            x, y = stack[-1]
            
            # Find unvisited neighbors (2 cells away to leave wall between)
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + 2 * dx, y + 2 * dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if grid[ny, nx] == 1:  # Unvisited (still wall)
                        neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Choose random unvisited neighbor
                idx = self.rng.randint(len(neighbors))
                nx, ny, dx, dy = neighbors[idx]
                
                # Carve path: remove wall between current and neighbor
                grid[y + dy, x + dx] = 0  # Wall between
                grid[ny, nx] = 0          # Neighbor cell
                
                stack.append((nx, ny))
            else:
                # Dead end: backtrack
                stack.pop()
        
        return grid
    
    def sample_free_position(
        self,
        exclude: Optional[Tuple[float, float]] = None,
        min_distance: float = 0.5
    ) -> Tuple[float, float]:
        """
        Sample a random position in free space.
        
        Args:
            exclude: Optional (x, y) position to avoid
            min_distance: Minimum distance from excluded position (in cells)
        
        Returns:
            position: (x, y) in continuous coordinates
        """
        max_attempts = 1000
        
        for _ in range(max_attempts):
            # Sample random grid cell
            gx = self.rng.randint(0, self.width)
            gy = self.rng.randint(0, self.height)
            
            # Check if cell is free
            if self.grid[gy, gx] == 1:
                continue
            
            # Convert to continuous coordinates (center of cell)
            x = (gx + 0.5) * self.cell_size
            y = (gy + 0.5) * self.cell_size
            
            # Check exclusion constraint
            if exclude is not None:
                dist = np.sqrt((x - exclude[0])**2 + (y - exclude[1])**2)
                if dist < min_distance * self.cell_size:
                    continue
            
            return (x, y)
        
        # Fallback: return first free cell found
        for gy in range(self.height):
            for gx in range(self.width):
                if self.grid[gy, gx] == 0:
                    x = (gx + 0.5) * self.cell_size
                    y = (gy + 0.5) * self.cell_size
                    return (x, y)
        
        raise RuntimeError("No free space in maze")
    
    def _sample_distant_position(
        self,
        reference: Tuple[float, float],
        min_distance_cells: int = None
    ) -> Tuple[float, float]:
        """
        Sample position far from reference point.
        
        Args:
            reference: (x, y) position to be far from
            min_distance_cells: Minimum distance in grid cells
        
        Returns:
            position: (x, y) in continuous coordinates
        """
        if min_distance_cells is None:
            # Default: at least half the maze diagonal
            min_distance_cells = int(0.5 * np.sqrt(self.width**2 + self.height**2))
        
        min_distance = min_distance_cells * self.cell_size
        
        max_attempts = 1000
        best_pos = None
        best_dist = 0.0
        
        for _ in range(max_attempts):
            candidate = self.sample_free_position()
            dist = np.sqrt(
                (candidate[0] - reference[0])**2 +
                (candidate[1] - reference[1])**2
            )
            
            if dist >= min_distance:
                return candidate
            
            if dist > best_dist:
                best_dist = dist
                best_pos = candidate
        
        # Return best found (may not meet constraint)
        return best_pos if best_pos is not None else self.sample_free_position()
    
    def collides(self, x: float, y: float, radius: float) -> bool:
        """
        Check if circle collides with walls or boundary.
        
        Tests if a circle centered at (x, y) with given radius
        intersects any wall cells or the maze boundary.
        
        Args:
            x: X position in continuous coordinates
            y: Y position in continuous coordinates
            radius: Circle radius
        
        Returns:
            collision: True if collision detected
        """
        # Boundary check
        if x - radius < 0 or x + radius > self.width * self.cell_size:
            return True
        if y - radius < 0 or y + radius > self.height * self.cell_size:
            return True
        
        # Convert to grid coordinates
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        
        # Check cells in radius (conservative AABB)
        search_radius = int(np.ceil(radius / self.cell_size)) + 1
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_gx = gx + dx
                check_gy = gy + dy
                
                # Skip out-of-bounds
                if not (0 <= check_gx < self.width and 0 <= check_gy < self.height):
                    continue
                
                # Skip free cells
                if self.grid[check_gy, check_gx] == 0:
                    continue
                
                # Wall cell found - check circle-square collision
                # Cell bounds in continuous space
                cell_x = check_gx * self.cell_size
                cell_y = check_gy * self.cell_size
                
                # Closest point on cell to circle center
                closest_x = np.clip(x, cell_x, cell_x + self.cell_size)
                closest_y = np.clip(y, cell_y, cell_y + self.cell_size)
                
                # Distance from circle center to closest point
                dist = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                
                if dist < radius:
                    return True
        
        return False
    
    def ray_distance(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        max_distance: float = 10.0,
        step: float = 0.05
    ) -> float:
        """
        Cast ray and return distance to nearest wall.
        
        Marches along ray direction until hitting a wall or reaching max_distance.
        
        Args:
            x: Ray origin X
            y: Ray origin Y
            dx: Ray direction X (need not be normalized)
            dy: Ray direction Y (need not be normalized)
            max_distance: Maximum ray length
            step: Step size for ray marching
        
        Returns:
            distance: Distance to wall, or max_distance if no hit
        """
        # Normalize direction
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-8:
            return max_distance
        
        dx = dx / length
        dy = dy / length
        
        # March along ray
        dist = 0.0
        while dist < max_distance:
            # Current position along ray
            rx = x + dx * dist
            ry = y + dy * dist
            
            # Check if position is in wall
            # Convert to grid coordinates
            gx = int(rx / self.cell_size)
            gy = int(ry / self.cell_size)
            
            # Boundary check
            if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
                return dist
            
            # Wall check
            if self.grid[gy, gx] == 1:
                return dist
            
            # Advance along ray
            dist += step
        
        return max_distance
    
    def get_start(self) -> Tuple[float, float]:
        """
        Get start position for current episode.
        
        Returns:
            position: (x, y) in continuous coordinates
        """
        assert self.start_pos is not None, "Must call reset() first"
        return self.start_pos
    
    def get_goal(self) -> Tuple[float, float]:
        """
        Get goal position for current episode.
        
        Returns:
            position: (x, y) in continuous coordinates
        """
        assert self.goal_pos is not None, "Must call reset() first"
        return self.goal_pos
    
    def get_grid(self) -> np.ndarray:
        """
        Get maze grid for visualization/debugging.
        
        Returns:
            grid: Binary array (height, width) where 1=wall, 0=free
        """
        return self.grid.copy()
    
    def to_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert continuous coordinates to grid cell indices.
        
        Args:
            x: Continuous X coordinate
            y: Continuous Y coordinate
        
        Returns:
            grid_pos: (gx, gy) grid indices
        """
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        return (gx, gy)
    
    def to_continuous_coords(self, gx: int, gy: int) -> Tuple[float, float]:
        """
        Convert grid cell indices to continuous coordinates (cell center).
        
        Args:
            gx: Grid X index
            gy: Grid Y index
        
        Returns:
            position: (x, y) continuous coordinates
        """
        x = (gx + 0.5) * self.cell_size
        y = (gy + 0.5) * self.cell_size
        return (x, y)


# Validation
if __name__ == "__main__":
    print("MazeEnv - Research-grade 2D maze environment")
    print("=" * 60)
    
    # Create environment
    env = MazeEnv(width=20, height=20, cell_size=1.0, seed=42)
    env.reset()
    
    print(f"Maze size: {env.width} x {env.height}")
    print(f"Cell size: {env.cell_size}")
    print(f"Start: {env.get_start()}")
    print(f"Goal: {env.get_goal()}")
    print()
    
    # Test collision
    start_x, start_y = env.get_start()
    collides = env.collides(start_x, start_y, radius=0.2)
    print(f"Start position collision (r=0.2): {collides}")
    
    # Test ray casting
    distance = env.ray_distance(start_x, start_y, 1.0, 0.0, max_distance=10.0)
    print(f"Ray distance (east): {distance:.2f}")
    
    # Test sampling
    pos = env.sample_free_position()
    print(f"Random free position: {pos}")
    
    print()
    print("✓ Environment implementation complete")
    print("✓ Ready for integration with sim/agent.py")

