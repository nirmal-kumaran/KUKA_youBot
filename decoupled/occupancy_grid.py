import numpy as np
import matplotlib.pyplot as plt

class OccupancyGrid:
    def __init__(self, sim, cell_size=0.1):
        self.sim = sim
        self.cell_size = cell_size
        self.grid = None
        self.map_dims = self._get_floor_dimensions()
        self.grid_width = int((self.map_dims[1] - self.map_dims[0]) / cell_size)
        self.grid_height = int((self.map_dims[3] - self.map_dims[2]) / cell_size)
        self.obstacles = self._get_obstacles()
        self.special_objects = self._get_special_objects()
        self._generate_grid()

    def _get_floor_dimensions(self):
        """Retrieve the dimensions of the floor."""
        floor_handle = self.sim.getObject('/ResizableFloor_5_25_visibleElement')
        min_x = self.sim.getObjectFloatParam(floor_handle, self.sim.objfloatparam_objbbox_min_x)
        max_x = self.sim.getObjectFloatParam(floor_handle, self.sim.objfloatparam_objbbox_max_x)
        min_y = self.sim.getObjectFloatParam(floor_handle, self.sim.objfloatparam_objbbox_min_y)
        max_y = self.sim.getObjectFloatParam(floor_handle, self.sim.objfloatparam_objbbox_max_y)
        return min_x, max_x, min_y, max_y

    def _get_obstacles(self):
        """Retrieve handles for all obstacles in the scene."""
        obstacle_names = [
            '/Obstacles/Cylinder[0]', '/Obstacles/Cylinder[1]', '/Obstacles/Cylinder[2]',
            '/Obstacles/Wall[0]', '/Obstacles/Wall[1]', '/Obstacles/Wall[2]', '/Obstacles/Wall[3]'
        ]
        obstacles = []
        for name in obstacle_names:
            handle = self.sim.getObject(name)
            if handle != -1:
                obstacles.append(handle)
            else:
                print(f"Warning: Obstacle '{name}' not found.")
        return obstacles

    # def _get_obstacles(self):
    #     """Retrieve handles for all obstacles in the scene."""
    #     obstacles = []
    #     i = 0
    #     while True:
    #         obstacle_name = f'/Obstacles/Cylinder[{i}]' # Iterate through all obstacles
    #         obstacle_handle = self.sim.getObject(obstacle_name)
    #         if obstacle_handle != -1:
    #             obstacles.append(obstacle_handle)
    #             i += 1 # next obstacle
    #         else:
    #             break # stop
    #         i=0
    #     while True:
    #         obstacle_name = f'/Obstacles/Wall[{i}]' # Iterate through all obstacles
    #         obstacle_handle = self.sim.getObject(obstacle_name)
    #         if obstacle_handle != -1:
    #             obstacles.append(obstacle_handle)
    #             i += 1 # next obstacle
    #         else:
    #             break # stop
    #     return obstacles #list of all cylinders

    def _get_special_objects(self):
        """Retrieve handles for special objects like youBot, Cuboid_initial, and Cuboid_goal."""
        special_objects = {}
        special_objects['youBot'] = self.sim.getObject('/youBot')
        special_objects['Cuboid_initial'] = self.sim.getObject('/Cuboid_initial')
        special_objects['Cuboid_goal'] = self.sim.getObject('/Cuboid_goal')
        
        for name, handle in special_objects.items():
            if handle == -1:
                print(f"Warning: Special object '{name}' not found.")
        
        return special_objects

    def _mark_obstacle(self, min_x, max_x, min_y, max_y):
        """Mark cells occupied."""
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Calculate cell center coordinates
                cell_center_x = self.map_dims[0] + (j + 0.5) * self.cell_size
                cell_center_y = self.map_dims[3] - (i + 0.5) * self.cell_size
                # Check if the cell center is within the obstacle's bounds
                # if min_x <= cell_center_x <= max_x and min_y <= cell_center_y <= max_y:
                cell_min_x = cell_center_x - self.cell_size/2
                cell_max_x = cell_center_x + self.cell_size/2
                cell_min_y = cell_center_y - self.cell_size/2
                cell_max_y = cell_center_y + self.cell_size/2

                if (cell_min_x < max_x and cell_max_x > min_x) and \
                (cell_min_y < max_y and cell_max_y > min_y):
                    # print(f"Marking cell ({i}, {j}) as occupied with center ({cell_center_x}, {cell_center_y}) and min_x, max_x, min_y, max_y: {min_x}, {max_x}, {min_y}, {max_y}")
                    self.grid[i][j] = 1

    def _generate_grid(self):
        """Generate the occupancy grid."""
        print(f"Grid size: {self.grid_height}x{self.grid_width}")
        
        # Create an empty grid
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

        # Process each obstacle and mark it on the grid
        for obstacle in self.obstacles:
            name = self.sim.getObjectAlias(obstacle)
            position = self.sim.getObjectPosition(obstacle, -1)  # World frame
            print(f"Processing obstacle '{name}' at ({position[0]}, {position[1]})")
            
            if 'Cylinder' in name:
                min_x = self.sim.getObjectFloatParam(obstacle, self.sim.objfloatparam_objbbox_min_x)
                max_x = self.sim.getObjectFloatParam(obstacle, self.sim.objfloatparam_objbbox_max_x)
                min_y = self.sim.getObjectFloatParam(obstacle, self.sim.objfloatparam_objbbox_min_y)
                max_y = self.sim.getObjectFloatParam(obstacle, self.sim.objfloatparam_objbbox_max_y)
                # Adjust bounds based on position
                min_x += position[0]
                max_x += position[0]
                min_y += position[1]
                max_y += position[1]

                self._mark_obstacle(min_x, max_x, min_y, max_y)
            elif 'Wall' in name:
                # Mark rectangular wall obstacle using bounding box dimensions
                min_x = self.sim.getObjectFloatParam(obstacle, self.sim.objfloatparam_objbbox_min_x)
                max_x = self.sim.getObjectFloatParam(obstacle, self.sim.objfloatparam_objbbox_max_x)
                min_y = self.sim.getObjectFloatParam(obstacle, self.sim.objfloatparam_objbbox_min_y)
                max_y = self.sim.getObjectFloatParam(obstacle, self.sim.objfloatparam_objbbox_max_y)
                
                # Adjust bounds based on position
                min_x += position[0]
                max_x += position[0]
                min_y += position[1]
                max_y += position[1]
                
                # Mark wall on grid
                self._mark_obstacle(min_x, max_x, min_y, max_y)

        # Mark special objects (youBot and Cuboids)
        for name, handle in self.special_objects.items():
            if handle != -1:
                position = self.sim.getObjectPosition(handle, -1)  # World frame
                i, j = int((self.map_dims[3] - position[1]) / self.cell_size), int((position[0] - self.map_dims[0]) / self.cell_size)
                
                if 0 <= i < self.grid_height and 0 <= j < self.grid_width:
                    marker_value_map = {'youBot': 2, 'Cuboid_initial': 3, 'Cuboid_goal': 4}
                    marker_value = marker_value_map.get(name, 1)  # 1 as default for any other obstacle
                    print(f"Marking {name} at ({i}, {j})")
                    self.grid[i][j] = marker_value # mark them in the grid with the given value
                    if name == 'youBot':
                        self.start_position = (i, j)
                    elif name == 'Cuboid_goal':
                        self.goal_position = (i, j)
                    elif name == 'Cuboid_initial':
                        self.payload_position = (i, j)

        print("GRID")
        print(self.grid)

    def visualize(self, path=None, b_spline_path=None):
        """
        Visualize the occupancy grid with gridlines shifted by half a cell size and special objects in different colors.
        Args:
            path: Optional list of waypoints [(i, j)] representing a trajectory (A* path).
            b_spline_path: Optional list of waypoints [(i, j)] representing a smoothed trajectory (B-spline path).
        """
        fig, ax = plt.subplots()

        # Plot occupancy grid with adjusted extent to shift gridlines
        ax.imshow(self.grid, cmap='Greys', origin='upper',
                extent=[0, self.grid_width * self.cell_size,
                        0, self.grid_height * self.cell_size])

        # Plot A* path if provided
        if path:
            # Convert grid indices to plot coordinates (cell centers)
            path_x = [(j + 0.5) * self.cell_size for (i, j) in path]
            path_y = [(self.grid_height - i - 0.5) * self.cell_size for (i, j) in path]
            ax.plot(path_x, path_y, 'y-', linewidth=2, label='A* Path')  # Yellow line for A* path

        # Plot B-spline smoothed path if provided
        if b_spline_path:
            # Convert grid indices to plot coordinates (cell centers)
            spline_x = [(j + 0.5) * self.cell_size for (i, j) in b_spline_path]
            spline_y = [(self.grid_height - i - 0.5) * self.cell_size for (i, j) in b_spline_path]
            ax.plot(spline_x, spline_y, 'c-', linewidth=2.5, label='B-Spline Smoothed Path')  # Cyan line for smoothed path

        # Mark special objects with different colors
        if 'youBot' in self.special_objects and self.special_objects['youBot'] != -1:
            pos = self.sim.getObjectPosition(self.special_objects['youBot'], -1)
            j = (pos[0] - self.map_dims[0]) / self.cell_size
            i = (self.map_dims[3] - pos[1]) / self.cell_size
            ax.plot(j + 0.5 * self.cell_size, i + 0.5 * self.cell_size,
                    'ro', markersize=8, label='YouBot (Red)')  # Red for robot

        if 'Cuboid_initial' in self.special_objects and self.special_objects['Cuboid_initial'] != -1:
            pos = self.sim.getObjectPosition(self.special_objects['Cuboid_initial'], -1)
            j = (pos[0] - self.map_dims[0]) / self.cell_size
            i = (self.map_dims[3] - pos[1]) / self.cell_size
            ax.plot(j + 0.5 * self.cell_size, i + 0.5 * self.cell_size,
                    'bo', markersize=8, label='Payload Position (Blue)')  # Blue for start

        if 'Cuboid_goal' in self.special_objects and self.special_objects['Cuboid_goal'] != -1:
            pos = self.sim.getObjectPosition(self.special_objects['Cuboid_goal'], -1)
            j = (pos[0] - self.map_dims[0]) / self.cell_size
            i = (self.map_dims[3] - pos[1]) / self.cell_size
            ax.plot(j + 0.5 * self.cell_size, i + 0.5 * self.cell_size,
                    'go', markersize=8, label='Goal Position (Green)')  # Green for goal

        # Add grid lines shifted by half a cell size
        ax.set_xticks(np.arange(0, self.grid_width * self.cell_size + 0.5 * self.cell_size, self.cell_size))  # Shift x-axis gridlines
        ax.set_yticks(np.arange(0, self.grid_height * self.cell_size + 0.5 * self.cell_size, self.cell_size))  # Shift y-axis gridlines
        ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)

        # Label the axes and title the plot
        ax.set_xlabel('X Grid Index')
        ax.set_ylabel('Y Grid Index')
        ax.set_title('Occupancy Grid')

        # Add legend for special objects and paths
        ax.legend()

        # Add colorbar for occupancy grid values
        plt.colorbar(ax.imshow(self.grid, cmap='viridis', origin='upper',
                            extent=[0, self.grid_width * self.cell_size,
                                    0, self.grid_height * self.cell_size]),
                    label="Occupied (1) / Free (0) / YouBot (2) / Payload (3) / Goal (4)")

        plt.show()


if __name__ == "__main__":
    print("This is just an Occupancy Grid class implementation. It cannot be run independently.")
    # grid.visualize()
