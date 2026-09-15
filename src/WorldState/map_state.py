"""
MapState Module
===============
Represents the warehouse grid map with cell types.
"""

from enum import Enum, auto
from typing import List, Tuple, Set

from Config.config_loader import MapConfig, PodZoneConfig


class CellType(Enum):
    """Types of cells on the warehouse grid."""
    FREE = auto()
    OBSTACLE = auto()
    STATION = auto()
    POD_HOME = auto()
    STATION_SERVICE = auto()
    STATION_QUEUE = auto()
    STATION_BUFFER = auto()
    STATION_EXIT = auto()
    STATION_ENTRY = auto()


class MapState:
    """
    Grid-based warehouse map.

    属性
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    grid : list[list[CellType]]
        2D grid of cell types.
    station_positions : dict[int, tuple[int, int]]
        Mapping of station_id -> (row, col).
    pod_home_positions : list[tuple[int, int]]
        List of all pod home positions derived from pod_zones.
    """

    def __init__(self, config: MapConfig):
        self.rows = config.rows
        self.cols = config.cols
        self.station_positions: dict[int, Tuple[int, int]] = {}
        self.pod_home_positions: List[Tuple[int, int]] = []

        # Initialize grid with FREE cells
        self.grid: List[List[CellType]] = [
            [CellType.FREE for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

        # Place obstacles
        for r, c in config.obstacles:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r][c] = CellType.OBSTACLE

        # Place stations
        for station in config.stations:
            r, c = station.row, station.col
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r][c] = CellType.STATION
                self.station_positions[station.id] = (r, c)

        # Expand pod zones into individual pod home positions
        for pz in config.pod_zones:
            for dr in range(pz.num_rows):
                for dc in range(pz.num_cols):
                    r = pz.origin_row + dr
                    c = pz.origin_col + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        if self.grid[r][c] == CellType.FREE:
                            self.grid[r][c] = CellType.POD_HOME
                            self.pod_home_positions.append((r, c))

    def is_walkable(self, row: int, col: int) -> bool:
        """Check if a cell can be traversed by a robot."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col] not in (
                CellType.OBSTACLE, CellType.STATION,
                CellType.STATION_SERVICE, CellType.STATION_QUEUE, CellType.STATION_BUFFER,
            )
        return False

    def mark_station_zone(self, positions_by_type: dict):
        """Mark station zone cells on the grid.

        Overwrites existing cell types (including POD_HOME). Any POD_HOME
        positions that are overwritten are removed from pod_home_positions.

        Parameters
        ----------
        positions_by_type : dict[CellType, list[tuple[int,int]]]
        """
        pod_home_set = set(self.pod_home_positions)
        for cell_type, positions in positions_by_type.items():
            for r, c in positions:
                if self.in_bounds(r, c):
                    if (r, c) in pod_home_set:
                        pod_home_set.discard((r, c))
                    self.grid[r][c] = cell_type
        self.pod_home_positions = [p for p in self.pod_home_positions if p in pod_home_set]

    def in_bounds(self, row: int, col: int) -> bool:
        """Check if coordinates are within the grid."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Return walkable 4-connected neighbor cells."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_walkable(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def __repr__(self) -> str:
        return f"MapState({self.rows}x{self.cols}, stations={len(self.station_positions)}, pods={len(self.pod_home_positions)})"
