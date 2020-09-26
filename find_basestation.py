import math
#FIND THE CLOSEST BASE STATION

class NearestNeighborRANModel():
    """
    Assigns each user to the closest base station (euclidian distance)
    """
    def __init__(self, base_stations, grid_resolution):
        self.__base_stations = base_stations
        self.__grid_resolution = grid_resolution
        min_x, max_x, min_y, max_y = self.__get_base_station_area_dimensions()
        max_dimension = max(max_x-min_x, max_y-min_y)
        self.__tile_size = max_dimension/(self.__grid_resolution-1)
        self.__inv_tile_size = 1.0/self.__tile_size
        self.__grid_offset_x = min_x
        self.__grid_offset_y = min_y

        self.__initialize_grid()
        self.__insert_base_stations_into_grid()

    def __insert_base_stations_into_grid(self):
        for i in self.__base_stations:
            x, y = self.__base_stations[i]
            print(x,y)
            cx,cy = self.__world_coordinates_to_cell_index(x, y)
            assert(cx >=0 and cx < self.__grid_resolution and cy >=0 and cy < self.__grid_resolution)
            self.__grid[cx][cy].append(i)

    def __initialize_grid(self):
        self.__grid = []
        for ix in range(self.__grid_resolution):
            row = []
            for iy in range(self.__grid_resolution):
                cell = []
                row.append(cell)
            self.__grid.append(row)

    def __world_to_grid_coordinates(self, x, y):
        '''print("Self.__grid_offset_x",self.__grid_offset_x)
        print("Self.__grid_offset_y",self.__grid_offset_y)
        print(x)
        print(y)'''
        tx, ty = x - self.__grid_offset_x, y - self.__grid_offset_y  # translating the coordinates to the grids origin.
        return tx * self.__inv_tile_size, ty * self.__inv_tile_size  # scaling the coordinates to the grid-scale

    def __world_coordinates_to_cell_index(self, x, y):
        """
        gets the cell index of a point in world-coordinates
        :param x: x-xoordinate of the point in world-coordinates
        :param y: y-coordinate of the point in world-coordinates
        :return: ix,iy, the cell-coordinates. Note that this might also be outside the allocated grid.
        """
        gx, gy = self.__world_to_grid_coordinates(x, y)
        return int(gx), int(gy)


    def __get_base_station_area_dimensions(self):
        min_x = math.inf
        max_x = -math.inf
        min_y = math.inf
        max_y = -math.inf

        for i in self.__base_stations:
            x,y = self.__base_stations[i]
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        return min_x, max_x, min_y, max_y

    def __get_ring_cells(self, smaller_box, larger_box):
        """
        a generator that yields the cells of a ring in the grid.
        a ring is specified by a smaller and a larger ring.
        yield all cells that are part of the larger ring, but not the smaller.
        if there is no smaller ring, return the whole box.
        :param smaller_ring: (min_x, max_x, min_y, max_y) / None
        :param larger_ring: (min_x, max_x, min_y, max_y) : all int, s.t. the yielded cells are >=min_x and <max_x, y analogous
        :return: None, Yield the cells
        """
        larger_min_x, larger_max_x, larger_min_y, larger_max_y = larger_box
        if not smaller_box:
            for ix in range(max(0,larger_min_x), min(self.__grid_resolution,larger_max_x)):
                for iy in range(larger_min_y, larger_max_y):
                    if ix >= 0 and ix < self.__grid_resolution and iy >= 0 and iy < self.__grid_resolution:
                        yield self.__grid[ix][iy]
        else:
            smaller_min_x, smaller_max_x, smaller_min_y, smaller_max_y = smaller_box
            for ix in range(max(0,larger_min_x), min(self.__grid_resolution,larger_max_x)):
                for iy in range(max(0,larger_min_y), min(self.__grid_resolution,larger_max_y)):
                    if not (ix >= smaller_min_x and ix < smaller_max_x and iy >= smaller_min_y and iy < smaller_max_y): #check if the cell is contained in the smaller box
                        if ix >= 0 and ix < self.__grid_resolution and iy >= 0 and iy < self.__grid_resolution:
                            yield self.__grid[ix][iy]

    def get_closest_base_station(self, user_position, blacklist = []):
        gx, gy = self.__world_to_grid_coordinates(user_position[0], user_position[1])
        cx, cy = self.__world_coordinates_to_cell_index(user_position[0], user_position[1])
        closest_base_station = None
        closest_base_station_sq_distance = math.inf
        smaller_box = None #the smaller box of the next iteration
        larger_box = (cx, cx+1, cy, cy+1) # the larger box of the next iteration
        while smaller_box != larger_box:
            #get the closest base station from the next ring
            for cell in self.__get_ring_cells(smaller_box,larger_box):
                for base_station in cell:
                    #if base_station not in blacklist:
                    bs_x, bs_y = self.__base_stations[base_station]
                    bs_gx, bs_gy = self.__world_to_grid_coordinates(bs_x,bs_y)
                    sq_distance = (bs_gx-gx)**2 + (bs_gy-gy)**2
                    if sq_distance < closest_base_station_sq_distance:
                        closest_base_station_sq_distance = sq_distance
                        closest_base_station = base_station
            smaller_box = larger_box

            if closest_base_station is not None: # a base station was found. no, make sue it's actually the closest one. There might be closer ones in the difference between the circle and the rectangle.
                # next iteration, check all cells that could contain a closer base station than the one that's closest at the moment (all cells that touch the circumference)
                closest_base_station_distance = math.sqrt(closest_base_station_sq_distance)
                larger_box = (int(gx-closest_base_station_distance),
                              int(gx+closest_base_station_distance)+1,
                              int(gy-closest_base_station_distance),
                              int(gy+closest_base_station_distance)+1)
            else: # no base station was found yet. => expand the search radius
                larger_box = (larger_box[0]-1,
                              larger_box[1]+1,
                              larger_box[2]-1,
                              larger_box[3]+1)
        return closest_base_station


    def __get_closest_base_station_brute_force(self, user_position, blacklist=[]):
        """
        Brute forces the estimation of the nearest base station
        :param user_position: position of the user (Vec2)
        :param cloud_network: cloud network that contains the base stations
        :return: The base station with the shorest euclidian distance from the user's position
        """
        closest_sq_distance = math.inf
        closest_sq_distance_base_station = None
        for base_station in self.__base_stations:
            if base_station not in blacklist:
                sq_distance = (Vec2.from_tuple(base_station.get_pos()) - Vec2.from_tuple(user_position)).sq_length()
                if sq_distance < closest_sq_distance:
                    closest_sq_distance = sq_distance
                    closest_sq_distance_base_station = base_station
        return closest_sq_distance_base_station

    def update_user_access_points(self, users):
        user_base = []
        for user in users:
            x = self.get_closest_base_station(user)
            user_base.append(x)
        return user_base