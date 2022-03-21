"""
Project13 Phase-1 - A* Algorithm for path planning of mobile robot
Author: Mohammed Maaruf Vazifdar
UID: 117509717
GitHub: https://github.com/maarufvazifdar/AStar_pathplanning
"""

import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import cv2


class Node:
    def __init__(self, position, cost, parent):
        self.position = position
        self.cost = cost
        self.parent = parent

    def getPos(self):
        return self.position

    def getCost(self):
        return self.cost


"""
Generating Map with obstacles
"""


def halfPlane(empty_map, pt1, pt2, right_side_fill=True):
    map = empty_map.copy()

    x_arranged = np.arange(400)
    y_arranged = np.arange(250)
    X, Y = np.meshgrid(x_arranged, y_arranged)

    slope = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0] + 1e-7)
    # diff = pt1[1] - slope * pt1[0]
    res = Y - slope * X - (pt1[1] - slope * pt1[0])

    if right_side_fill:
        map[res > 0] = 0
    else:
        map[res <= 0] = 0

    return map


def createQuad(map):
    # Creating the convex quadraliteral
    # Points of convex quadraliteral
    p1 = [36, 65]
    p2 = [115, 40]
    p3 = [80, 70]
    p4 = [105, 150]

    # Upper triangle
    hp1 = halfPlane(map, p1, p2, True)
    hp2 = halfPlane(map, p2, p3, False)
    hp3 = halfPlane(map, p3, p1, False)

    triangle1 = hp1 + hp2 + hp3

    # Lower triangle
    hp4 = halfPlane(map, p1, p3, True)
    hp5 = halfPlane(map, p3, p4, True)
    hp6 = halfPlane(map, p4, p1, False)

    triangle2 = hp4 + hp5 + hp6

    map = cv2.bitwise_and(triangle1, triangle2)

    return map


def createCircle(map):
    # Creating circle
    center = [300, 65]
    radius = 40
    c_x = np.arange(400)
    c_y = np.arange(250)
    c_X, c_Y = np.meshgrid(c_x, c_y)
    map[(c_X - center[0])**2 + (c_Y - center[1])**2 - radius**2 <= 0] = 0
    return map


def createHexagon(map):

    h1 = [200, 150 - (70 / np.sqrt(3))]
    h2 = [200 + 35, 150 - (35 / np.sqrt(3))]
    h3 = [200 + 35, 150 + (35 / np.sqrt(3))]
    h4 = [200, 150 + (70 / np.sqrt(3))]
    h5 = [200 - 35, 150 + (35 / np.sqrt(3))]
    h6 = [200 - 35, 150 - (35 / np.sqrt(3))]

    hp1 = halfPlane(map, h1, h2, True)
    hp2 = halfPlane(map, h2, h3, False)
    hp3 = halfPlane(map, h3, h4, False)
    hp4 = halfPlane(map, h4, h5, False)
    hp5 = halfPlane(map, h5, h6, False)
    hp6 = halfPlane(map, h6, h1, True)
    map = hp1 + hp2 + hp3 + hp4 + hp5 + hp6

    return map


def createMap():
    empty_map = np.ones((250, 400), np.uint8) * 255

    quad = createQuad(empty_map)
    circle = createCircle(quad)
    hexagon = createHexagon(circle)
    map = hexagon

    return map


def checkObstacle(pose):
    """Check if pose is in obstacle space."""
    if pose[0] < (robot_radius + clearence) \
            or pose[0] >= (400 - robot_radius + clearence) \
            or pose[1] < (robot_radius + clearence) \
            or pose[1] >= (250 - robot_radius + clearence) \
            or inflated_map[int(pose[1])][int(pose[0])] == 0:
        return False
    return True


def CW30(th):
    new_th = th - 30
    if new_th <= -360:
        new_th = 360 + new_th
    return new_th


def ACW30(th):
    new_th = th + 30
    if new_th >= 360:
        new_th = new_th - 360
    return new_th


def CW60(th):
    new_th = th - 60
    if new_th <= -360:
        new_th = 360 + th
    return new_th


def ACW60(th):
    new_th = th + 60
    if new_th >= 360:
        new_th = new_th - 360
    return new_th


def actionSet(px, py, th, step_size):
    """Retuurns new  positions of neighbours"""

    actionSet = [
        (px + (step_size * np.cos(np.radians(CW30(th)))),
         py + (step_size * np.sin(np.radians(CW30(th)))),
         CW30(th)),
        (px + (step_size * np.cos(np.radians(ACW30(th)))),
         py + (step_size * np.sin(np.radians(ACW30(th)))),
         ACW30(th)),
        (px + (step_size * np.cos(np.radians(CW60(th)))),
         py + (step_size * np.sin(np.radians(CW60(th)))),
         CW60(th)),
        (px + (step_size * np.cos(np.radians(ACW60(th)))),
         py + (step_size * np.sin(np.radians(ACW60(th)))),
         ACW60(th)),
        (px + (step_size * np.cos(np.radians(th))),
         py + (step_size * np.sin(np.radians(th))),
         th)]
    return actionSet


def heuristicCost(node, goal_node):
    """Calculates heuristic distance between points"""
    x = node[0]
    y = node[1]
    dist = np.hypot(x - goal_node.position[0],
                    y - goal_node.position[1])
    return dist


def findNeighbours(node, step_size):
    """Returns explored paths and corresponding costs"""
    x = node.position[0]
    y = node.position[1]
    th = node.position[2]
    action = actionSet(x, y, th, step_size)
    neighbours = []
    for i, path in enumerate(action):
        if checkObstacle(path):
            neighbours.append(Node(path, 10 + node.getCost(), node))
    return neighbours


def visitedNodes(node, goal_node, visited):
    """Check if pose is already visited"""
    node_pos = node.getPos()
    x = node_pos[0]
    y = node_pos[1]
    th = node_pos[2]
    x = int((round(2 * x) / 2) / 0.5)
    y = int((round(2 * y) / 2) / 0.5)
    th = int(th / 30)
    if ((node.getCost() + heuristicCost(node_pos, goal_node))
            < visited[x, y, th]):
        return True
    else:
        return False


def goalReached(node, goal_node, goal_tolerance):
    node_pos = node.getPos()

    dist = np.hypot((node_pos[0] - goal_node[0]), (node_pos[1] - goal_node[1]))
    if dist < goal_tolerance:
        return True
    else:
        return False


"""
Starts Here
"""
start_x = int(input("Enter Start x: ") or 20)
start_y = int(input("Enter Start y: ") or 230)
start_th = int(input("Enter Start theta: ") or 0)

goal_x = int(input("Enter Goal x: ") or 250)
goal_y = int(input("Enter Goal y: ") or 150)
goal_th = int(input("Enter Goal Theta: ") or 0)

step_size = int(input("Step size (1 to 10): ") or 10)
robot_radius = int(input("Robot Radius (default 10): ") or 10)
clearence = int(input("Robot-obstacle clearence (default 5): ") or 5)

start_pose = [start_x, 250 - start_y, start_th]
goal_pose = [goal_x, 250 - goal_y, goal_th]
goal_tolerance = 1.5

map = createMap()
kernel_size = 2 * (robot_radius + clearence) + 1
kernel = np.ones((kernel_size, kernel_size), 'uint8')
inflated_map = cv2.erode(map, kernel, iterations=1)

if checkObstacle(start_pose) is False:
    print('Invalid starting pose\n')
    exit()
if checkObstacle(goal_pose) is False:
    print('Invalid Goal pose\n')
    exit()

fig1 = plt.figure()
plt.imshow(map)

plt.scatter(start_pose[0], start_pose[1], color='green', s=50)
plt.scatter(goal_pose[0], goal_pose[1], color='red', s=50)

visited = np.array([[[np.inf for k in range(12)]
                     for j in range(int(250 / 0.5))]
                   for i in range(int(400 / 0.5))])
arrows = []
pq = PriorityQueue()
start_node = Node(start_pose, 0, None)
goal_node = Node(goal_pose, 0, None)
temp = 0
pq.put((start_node.cost, temp, start_node))

while not pq.empty():
    current_node = pq.get()[2]

    if goalReached(current_node, goal_pose, goal_tolerance):

        print("Goal Reached !!")
        break
    else:
        explortion = findNeighbours(current_node, step_size)
        # print(explortion)
        for i in explortion:

            explore_pos = i.getPos()
            # print(explore_pos)
            if visitedNodes(i, goal_node, visited):
                visited[int((round(2 * explore_pos[0]) / 2) / 0.5), int((round(2 * explore_pos[1]) / 2) / 0.5), int(
                    (round(2 * explore_pos[2]) / 2) / 30)] = i.getCost() + heuristicCost(explore_pos, goal_node)

                # Append x,y, dx and dy to plot arrows
                arrows.append([current_node.getPos()[0],
                              current_node.getPos()[1],
                              (explore_pos[0] - current_node.getPos()[0]),
                              (explore_pos[1] - current_node.getPos()[1])])

                pq.put((i.getCost() + heuristicCost(
                        explore_pos, goal_node), temp, i))
                temp += 1

# Visualizing explored nodes
print('Visualizing explored nodes')
for i in range(0, len(arrows)):
    plt.arrow(
        arrows[i][0],
        arrows[i][1],
        arrows[i][2],
        arrows[i][3],
        head_width=1,
        head_length=1,
        fc='k',
        ec='k')
    plt.pause(1e-8)

# Backtracking
print('Visualizing Backtracking...')
path_x = []
path_y = []
while current_node:
    path_x.append(current_node.position[0])
    path_y.append(current_node.position[1])
    current_node = current_node.parent
    plt.plot(path_x, path_y, "-b", linewidth=2)
    plt.pause(0.1)
plt.show()
