import numpy as np
from matplotlib import pyplot as plt

from src.utils import distance_euc


class ProblemInstance:

    def __init__(self, name_tsp):
        self.exist_opt = False
        self.optimal_tour = None
        self.dist_matrix = None

        # read raw data
        self.file_name = name_tsp
        file_object = open(self.file_name)
        data = file_object.read()
        file_object.close()
        self.lines = data.splitlines()

        # store data set information
        self.name = self.lines[0].split(' ')[2]
        self.nPoints = np.int(self.lines[3].split(' ')[2])
        self.best_sol = np.float(self.lines[5].split(' ')[2])

        # read all data points and store them
        self.points = np.zeros((self.nPoints, 3))
        for i in range(self.nPoints):
            line_i = self.lines[7 + i].split(' ')
            self.points[i, 0] = int(line_i[0])
            self.points[i, 1] = line_i[1]
            self.points[i, 2] = line_i[2]

        self.create_dist_matrix()
        if self.file_name in ["./problems/eil76.tsp", "./problems/kroA100.tsp", "../problems/eil76.tsp",
                              "../problems/kroA100.tsp"]:
            self.exist_opt = True
            file_object = open(self.file_name.replace(".tsp", ".opt.tour"))
            data = file_object.read()
            file_object.close()
            lines = data.splitlines()

            # read all data points and store them
            self.optimal_tour = np.zeros(self.nPoints, dtype=np.int)
            for i in range(self.nPoints):
                line_i = lines[5 + i].split(' ')
                self.optimal_tour[i] = int(line_i[0]) - 1

    def print_info(self):
        print("\n\n#############################")
        print('name: ' + self.name)
        print('nPoints: ' + str(self.nPoints))
        print('best_sol: ' + str(self.best_sol))
        print('exist optimal: ' + str(self.exist_opt))

    def plot_data(self, show_numbers=False):
        plt.figure(figsize=(8, 8))
        plt.title(self.name)
        plt.scatter(self.points[:, 1], self.points[:, 2])
        if show_numbers:
            for i, txt in enumerate(np.arange(self.nPoints)):  # tour_found[:-1]
                plt.annotate(txt, (self.points[i, 1], self.points[i, 2]))
        plt.show()

    def create_dist_matrix(self):
        self.dist_matrix = np.zeros((self.nPoints, self.nPoints))

        for i in range(self.nPoints):
            for j in range(i, self.nPoints):
                self.dist_matrix[i, j] = distance_euc(self.points[i][1:3], self.points[j][1:3])
        self.dist_matrix += self.dist_matrix.T
