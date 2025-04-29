import numpy as np
import matplotlib.pyplot as plt
import math

WRITE = False # set to True if you want to write the data to a CSV file and process in another language

wall = [(0.1,4.), (3.,0.1)]

true_a = -1.3421249130989874
true_b = 4.1298880303712995


robot = [0.,0.]

def fit_line(data):
    # add your least-squares code here
    x = []
    y = []

    for dist, ang in data:
        x.append((dist * np.cos(ang), 1.))
        y.append((dist * np.sin(ang)))

    a, b = np.linalg.lstsq(x, y)[0]

    return a, b

# plots results (will open a window / requires Xforwarding to view over SSH)
def PlotResults(data, a, b, wall, robot):
    fig, ax = plt.subplots()

    # visualize raw data
    remap_data = np.array([[d*np.cos(theta), d*np.sin(theta)] for (d, theta) in data])
    ax.scatter(remap_data[:,0], remap_data[:,1], color='red', marker='o', label="raw data")

    # visualize robot with heading at theta=0
    robot = plt.Circle(robot, 0.5, fill=False, edgecolor='purple')
    ax.add_patch(robot)
    heading = [[0, 0.5], [0,0]]
    ax.plot(heading[0], heading[1], color='purple')

    # visualize estimated wall line
    x = np.linspace(0,5,10)
    y = a*x+b
    ax.plot(x, y, color='blue', linestyle="--", label="estimated line")

    ax.legend()

    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 5])
    ax.axis('equal')
    plt.show()

def FindDistances(wall, robot, sigma):
    wall = np.array(wall)
    robot = np.array(robot)
    data = []
    for theta in np.linspace(0, np.pi, 20):
        t, u, pt = ShootRay(robot, theta, wall[0], wall[1])
        if t > 0 and (0 < u) and (1 > u):
            dist = np.linalg.norm(pt - robot)
            data.append((round(dist,2), round(theta,2)))
    noise = np.random.normal(loc=0.0, scale=sigma, size=len(data))
    data = np.array(data)
    dat = []
    i = 0
    for dist, ang in data:
        dat.append((dist + noise[i], ang))
        i = i + 1

    return np.array(dat)

def ShootRay(pt, theta, v1, v2):
    '''
    # find line intersection parameter of edge (v1,v2)
    # https://stackoverflow.com/questions/14307158/how-do-you-check-for-intersection-between-a-line-segment-and-a-line-ray-emanatin/32146853
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    '''
    def Cross2d(p,q):
        return p[0]*q[1] - p[1]*q[0]

    # points on ray are (x1,y1) + t*r
    r = (np.cos(theta), np.sin(theta))
    # points on segment are (x2,y2) + u*s
    s = v2-v1
    rXs = Cross2d(r,s)

    # if ray and target edge are parallel, will get divide by zero
    u = Cross2d(v1-pt, r)/rXs
    t = Cross2d(v1-pt, s)/rXs
    pint = np.array([pt[0] + np.cos(theta)*t, pt[1] + np.sin(theta)*t])
    return t, u, pint

def calcMse(data, a, b):
    err = 0.
    for dist, ang in data:
        d = abs((b)/(np.sin(ang)-a*np.cos(ang)))
        temp_err = (dist - d)**2
        err = err + temp_err
    err = err
    return err

def calcVar(data):
    mean = 0.
    for dist, ang in data:
        mean = mean + dist
    mean = mean/len(data)
    var = 0.
    for dist, ang in data:
        var = var + (dist - mean)**2
    var = var
    return var

def calcR2(data, a, b):
    mse = calcMse(data, a, b)
    var = calcVar(data)
    r2 = 1 - (mse/var)
    return r2

def main():
    for i in range(10):
        stdDev = float(i) * 0.1
        avgR2 = 0.
        avgA = 0.
        avgB = 0.
        for j in range(100):
            dat = FindDistances(wall, robot, stdDev)
            # print("raw data (distance, heading):", dat)
            # optionally, write data to file to process in another language
            if WRITE:
                np.savetxt('sensor-data.csv', dat, delimiter=',', fmt='%f')
            a, b = fit_line(dat)
            # print(f"a: {a}, b: {b}")
            r2 = calcR2(dat, a, b)
            avgR2 = avgR2 + r2
            # print(f"R2: {r2}")
            # PlotResults(dat, a, b, wall, robot)
            avgA = avgA + a
            avgB = avgB + b
        avgR2 = avgR2 / 100.
        print(f"Sigma: {stdDev}, R2: {avgR2}")

if __name__ == '__main__':
    main()
