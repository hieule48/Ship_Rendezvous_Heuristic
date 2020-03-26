
import numpy as np

if __name__ == '__main__':
    PROBLEM_FILE = 'sample_srp_data.csv'
    main(PROBLEM_FILE)
    
def main(csv_file):
    '''
    Main function for running and solving an instance of the SRP.

    Keyword arguments:
    csv_file -- the csv file name and path to a data for an instance of
                ship rendezvous problem.
    '''
    # read in the csv file to a numpy array
    ships = read_srp_input_data(csv_file)
    faster_ships = find_faster_ships(ships)
    solution, distance_from_port = greedy_heuristic_srp(ships, faster_ships)
    kpi = kpi_input(ships, solution, distance_from_port)
    np.savetxt('solution.csv', solution, delimiter=',',
               header='ship_index, interception_x_coordinate, \
interception_y_coordinate, estimated_time_of_interception',
               comments='')
    np.savetxt('kpi.csv', kpi, delimiter=',')


def read_srp_input_data(csv_file):
    '''
    Problem instances for the SRP are stored within a .csv file
    This function reads the problem instance into Python.
    Returns a 2D np.ndarray (4 columns).
    Skips the headers and the first column.
    Columns are:
    x-coordinate, y-coordinate, x-speed and y-speed

    Keyword arguments:
    csv_file -- the file path and name to the problem instance file
    '''

    input_data = np.genfromtxt(csv_file, delimiter=',',
                               dtype=np.float64,
                               skip_header=1,
                               usecols=tuple(range(1, 5)))

    return input_data


def service_ship_distance(ships, current_ships):
    '''
    Returns distance between service ship and the port

    Key arguments:
    ships -- 2d array with coordinates and velocity of ships
    current_ships -- 2d array with current coordinates of ships
    '''

    return np.sqrt((current_ships[0, 0] - ships[0, 0])**2
                   + (current_ships[0, 1] - ships[0, 1])**2)


def estimated_intercept_time(ships):
    '''
    Solves quadratic equations to find intercept time.
    Returns nympy array presenting all intercept times.

    Key arguments:
    ships -- 2d array with coordinates and velocity of ships
    '''

    n_ships = ships.shape[0]
    intercept_time = np.zeros(n_ships - 1)

    for i in range(1, n_ships):
        # Find coefficients
        a = ships[i, 2]**2 + ships[i, 3]**2 - (ships[0, 2]**2 + ships[0, 3]**2)
        b = 2 * (ships[i, 2] * (ships[i, 0]-ships[0, 0])
                 + ships[i, 3] * (ships[i, 1]-ships[0, 1]))
        c = (ships[i, 0]-ships[0, 0])**2 + (ships[i, 1]-ships[0, 1])**2

        d = b**2 - 4*a*c   # discriminant

        if a < 0:    # service ship is faster
            x_1 = (-b+np.sqrt(d)) / (2*a)
            x_2 = (-b-np.sqrt(d)) / (2*a)
            if x_1 >= 0 and x_2 >= 0:
                interception = min(x_1, x_2)
            elif (x_2 < 0 <= x_1) or (x_1 < 0 <= x_2):
                interception = max(x_1, x_2)
            else:
                interception = -1
        else:
            interception = -1

        intercept_time[i-1] = interception

    return intercept_time


def find_y_max(ships_unvisited):
    '''
    Finds index of ship with highest y coordinate.
    In case of a tie, returns smallest index.

    Key arguments:
    ships_unvisited -- 2d numpy array of unvisited ships
    '''

    y_ships_unvisited = ships_unvisited[:, 1]
    max_y_indexes = np.where(y_ships_unvisited == max(y_ships_unvisited))
    max_y_index = max_y_indexes[0][0]

    return max_y_index


def update_positions(ships, intercept_time):
    '''
    Returns updated positions of all ships after interception.

    Key arguments:
    ships -- 2d numpy array of initial coordinates and velocity
    intercept_time -- time updates happen
    '''

    current_ships = np.copy(ships)
    current_ships[:, 0] = ships[:, 0] + intercept_time*ships[:, 2]
    current_ships[:, 1] = ships[:, 1] + intercept_time*ships[:, 3]

    return current_ships


def find_faster_ships(ships):
    '''
    Finds indexes of cruise ships with faster speed.
    Returns 1d numpy array with indexes of faster ships.

    Keyword argument:
    ships -- 2d numpy array with velocity of ships.
    '''

    speed_cruise_ships = np.zeros(len(ships) - 1)
    speed_service_ship = np.sqrt(ships[0, 2]**2 + ships[0, 3]**2)
    for ship in range(len(ships) - 1):
        speed_cruise_ships[ship] = np.sqrt(ships[ship+1, 2]**2
                                           + ships[ship+1, 3]**2)

    faster_ships = np.where(speed_cruise_ships >= speed_service_ship)[0]

    return faster_ships


def greedy_heuristic_srp(ships, faster_ships):
    '''
    Greedy heuristic for SRP problem.
    Returns numpy array with solution to the problem and
    list of distances between support ship and the port.

    Keyword arguments:
    ships -- 2d numpy array of ships coordinates and velocity
    faster_ships -- 1d numpy array of indexes of faster ships
    '''

    cruise_ships = ships[1:, :]
    n_cruise_ships = cruise_ships.shape[0]
    ship_visited = np.full(n_cruise_ships, False, dtype=np.bool)
    solution = np.zeros((n_cruise_ships, 4), dtype=object)
    current_ships = np.copy(ships)
    distance_from_port = []
    estimated_time_of_interception = 0

    if len(faster_ships) > 0:
        solution[n_cruise_ships-len(faster_ships):][:] = np.full(4, -1,
                                                                 dtype=np.int64)
        ship_visited[faster_ships] = True

    for ship in range(n_cruise_ships - len(faster_ships)):
        intercept_time = estimated_intercept_time(current_ships)
        estimated_interception = np.full(n_cruise_ships, -2, dtype=np.float64)
        estimated_interception[~ship_visited] = intercept_time[~ship_visited]
        nearest_unvisited_index = np.where(estimated_interception
                                           == min(estimated_interception[~ship_visited]))
        ship_to_be_visited = nearest_unvisited_index[0][0]

        # Ties break
        if len(nearest_unvisited_index[0]) > 1:
            nearest_ships_unvisited = cruise_ships[nearest_unvisited_index[0]]
            max_y_index = find_y_max(nearest_ships_unvisited)
            ship_to_be_visited = nearest_unvisited_index[0][max_y_index]

        shortest_time = estimated_interception[ship_to_be_visited]
        ship_visited[ship_to_be_visited] = True

        # Update positions
        estimated_time_of_interception += shortest_time
        current_ships = update_positions(ships, estimated_time_of_interception)
        current_ships[0, :2] = current_ships[ship_to_be_visited+1, :2]
        solution[ship] = [ship_to_be_visited, current_ships[0, 0],
                          current_ships[0, 1], estimated_time_of_interception]

        distance_from_port.append(service_ship_distance(ships, current_ships))

    return solution, distance_from_port


def kpi_input(ships, solution, distance_from_port):
    '''
    Returns a numpy array with kpi inputs.

    Keyword arguments:
    ships -- 2d numpy array of ships coordinates and velocity
    solution -- 2d numpy array of cruise ships  in visting order
    distance_from_port -- list with distances of the service ship from port
    '''

    reached_ships = solution[solution[:, 0] != -1]
    n_ships = len(reached_ships)
    y_port = ships[0, 1]

    if n_ships > 0:
        max_wait = reached_ships[-1][-1]

        last_distance_from_port = distance_from_port[-1]
        speed_support_ship = np.sqrt(ships[0, 2]**2 + ships[0, 3]**2)
        time_return_to_port = last_distance_from_port / speed_support_ship
        total_time = max_wait + time_return_to_port

        max_y = max(max(reached_ships[:, 2]), y_port)
        furthest_distance = max(distance_from_port)
        avg_time = sum(reached_ships[:, 3]) / n_ships
    elif n_ships == 0:
        total_time = 0
        max_wait = 0
        max_y = y_port
        furthest_distance = 0
        avg_time = -1

    kpi_file = np.array([n_ships,
                         total_time,
                         max_wait,
                         max_y,
                         furthest_distance,
                         avg_time])

    return kpi_file



