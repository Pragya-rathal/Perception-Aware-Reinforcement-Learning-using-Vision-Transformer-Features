"""
Grid-to-image renderer for the visual navigation environment.

Converts the discrete grid state into an 84x84 RGB image with clear
color coding for different object types.
"""

import numpy as np
from typing import Tuple, Dict


class GridRenderer:
    """
    Renders a grid-based environment state as an RGB image.
    
    Attributes:
        grid_size: Size of the grid (e.g., 10 for 10x10)
        render_size: Output image size in pixels (e.g., 84 for 84x84)
        colors: Dictionary mapping object types to RGB values
    """
    
    # Color definitions (RGB values 0-255)
    DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
        "background": (240, 240, 240),  # Light gray
        "agent": (65, 105, 225),         # Royal blue
        "target": (50, 205, 50),         # Lime green
        "obstacle": (220, 20, 60),       # Crimson red
        "grid_line": (200, 200, 200),    # Gray grid lines
    }
    
    def __init__(
        self,
        grid_size: int = 10,
        render_size: int = 84,
        colors: Dict[str, Tuple[int, int, int]] = None,
        draw_grid_lines: bool = True
    ):
        """
        Initialize the renderer.
        
        Args:
            grid_size: Number of cells in each dimension
            render_size: Output image size in pixels
            colors: Optional custom color mapping
            draw_grid_lines: Whether to draw grid lines
        """
        self.grid_size = grid_size
        self.render_size = render_size
        self.colors = colors if colors is not None else self.DEFAULT_COLORS.copy()
        self.draw_grid_lines = draw_grid_lines
        
        # Calculate cell size in pixels
        self.cell_size = render_size / grid_size
        
    def render(
        self,
        agent_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        obstacle_positions: list
    ) -> np.ndarray:
        """
        Render the current grid state as an RGB image.
        
        Args:
            agent_pos: (row, col) position of the agent
            target_pos: (row, col) position of the target
            obstacle_positions: List of (row, col) positions for obstacles
            
        Returns:
            numpy array of shape (render_size, render_size, 3) with dtype uint8
        """
        # Initialize image with background color
        image = np.full(
            (self.render_size, self.render_size, 3),
            self.colors["background"],
            dtype=np.uint8
        )
        
        # Draw grid lines if enabled
        if self.draw_grid_lines:
            self._draw_grid_lines(image)
        
        # Draw obstacles first (so they're behind if overlapping)
        for obs_pos in obstacle_positions:
            self._fill_cell(image, obs_pos, self.colors["obstacle"])
        
        # Draw target
        self._fill_cell(image, target_pos, self.colors["target"])
        
        # Draw agent on top
        self._fill_cell(image, agent_pos, self.colors["agent"])
        
        return image
    
    def _fill_cell(
        self,
        image: np.ndarray,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        padding: int = 1
    ) -> None:
        """
        Fill a grid cell with the specified color.
        
        Args:
            image: Image array to modify in-place
            position: (row, col) grid position
            color: RGB color tuple
            padding: Pixels of padding inside the cell
        """
        row, col = position
        
        # Calculate pixel coordinates
        y_start = int(row * self.cell_size) + padding
        y_end = int((row + 1) * self.cell_size) - padding
        x_start = int(col * self.cell_size) + padding
        x_end = int((col + 1) * self.cell_size) - padding
        
        # Ensure bounds are valid
        y_start = max(0, y_start)
        y_end = min(self.render_size, y_end)
        x_start = max(0, x_start)
        x_end = min(self.render_size, x_end)
        
        # Fill the cell
        image[y_start:y_end, x_start:x_end] = color
    
    def _draw_grid_lines(self, image: np.ndarray) -> None:
        """
        Draw grid lines on the image.
        
        Args:
            image: Image array to modify in-place
        """
        color = self.colors["grid_line"]
        
        for i in range(self.grid_size + 1):
            # Pixel position for this grid line
            pos = int(i * self.cell_size)
            if pos >= self.render_size:
                pos = self.render_size - 1
            
            # Horizontal line
            image[pos, :] = color
            # Vertical line
            image[:, pos] = color
    
    def position_to_pixels(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert grid position to pixel center coordinates.
        
        Args:
            position: (row, col) grid position
            
        Returns:
            (y, x) pixel coordinates of cell center
        """
        row, col = position
        y = int((row + 0.5) * self.cell_size)
        x = int((col + 0.5) * self.cell_size)
        return (y, x)
