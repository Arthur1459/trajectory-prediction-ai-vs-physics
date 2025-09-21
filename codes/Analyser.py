import matplotlib.pyplot as plt
import tools as t
import glob
from numpy import mean, polyfit
import numpy as np
from math import radians, cos, sin
from aero import integrate
from ModelAI import getRegr, auto_reg_V2, ConvertToRealCoord

def getFilesNames(directory, file_extension="txt"):
    return [f for f in glob.glob(directory + "/*." + file_extension)]

def extractAnalyse_V2(file_name) -> dict:
    file = open(file_name, "r")
    content = file.readlines()
    file.close()

    traj = {"coordx": [], "coordy": [], "angle": [], "t": []}
    for line in content:
        line_unwrap = line.split(sep="||")
        try:
            x = float(line_unwrap[0].replace("\n", ""))
            y = float(line_unwrap[1].replace("\n", ""))
            angle = float(line_unwrap[2].replace("\n", ""))
            t = float(line_unwrap[3].replace("\n", ""))
            traj["coordx"].append(x)
            traj["coordy"].append(y)
            traj["angle"].append(angle)
            traj["t"].append(t)
        except:
            print(f"Error while reading file : {file_name}")
            continue
    return traj

def getInitialConditions(traj):
    XY0 = (traj['coordx'][0], traj['coordy'][0])
    ANGLE0 = traj['angle'][0]
    V0 = t.v_mult((traj['coordx'][1] - traj['coordx'][0], traj['coordy'][1] - traj['coordy'][0]), t.inv(traj['t'][1] - traj['t'][0]))
    return XY0, ANGLE0, V0

def getThreeFirstPoint(traj):
    p1, p2, p3 = (traj['coordx'][0], traj['coordy'][0]), (traj['coordx'][1], traj['coordy'][1]), (traj['coordx'][2], traj['coordy'][2])
    a1, a2, a3 = traj['angle'][0], traj['angle'][1], traj['angle'][2]
    t1, t2, t3 = traj['t'][0], traj['t'][1], traj['t'][2]
    return (p1, a1, t1), (p2, a2, t2), (p3, a3, t3)

def CompareBased(file, regr, normalisation_factors=None):
    """
    Make the physic and ia prediction according to the initial parameter of real traj.
    :param file: File for traj reference
    :param regr: The IA regression
    :param normalisation_factors: the factor of normalisation of the ai
    :return: (real traj, phy traj, ai traj)
    """
    traj = extractAnalyse_V2(file)
    if len(traj['coordx']) < 2:
        return ([], []), ([], []), ([], [])
    XY0, ANGLE0, V0 = getInitialConditions(traj)
    s1, s2, s3 = getThreeFirstPoint(traj)

    t0, tf = traj['t'][0], traj['t'][-1]
    delta_t = tf - t0
    nb_pts = len(traj['coordx'])
    dt = delta_t / nb_pts

    sample_i = [s1[0][0], s1[0][1], s1[1], s1[2], s2[0][0], s2[0][1], s2[1], s2[2], s3[0][0], s3[0][1], s3[1], s3[2], s3[2] + dt]

    physics_model = integrate(XY0, V0, ANGLE0, delta_t / nb_pts, delta_t, nb_pts=nb_pts)

    ia_model = ConvertToRealCoord(auto_reg_V2(regr, normalisation_factors, sample_i, dt, nb_pts=nb_pts), normalisation_factors)

    return (traj['coordx'], traj['coordy']), (physics_model['coordx'], physics_model['coordy']), ([ia_model[i][0] for i in range(len(ia_model))], [ia_model[i][1] for i in range(len(ia_model))])

def DistToReal(real, traj, r=2):
    dist = 0
    for i in range(len(real[0])):
        x, y = traj[0][i], traj[1][i]
        x_r, y_r = real[0][i], real[1][i]
        dist += ((x_r - x)**2 + (y_r - y)**2)**0.5
    return round(dist/(len(real[0]) if len(real[0]) != 0 else 1), r)

def MeasureDistances(files, regr, normalisation_factors):
    distances = {'phy': [], 'ia': []}
    for file in files:
        real, physic, ia = CompareBased(file, regr, normalisation_factors)
        distances['phy'].append(DistToReal(real, physic))
        distances['ia'].append(DistToReal(real, ia))
    return distances

def PlotDistance(dists):
    """
    Plot an histogram of distances
    :param dists: {'ia': [], 'phy': []}
    :return:
    """
    graph, ((ax0), (ax1)) = plt.subplots(nrows=1, ncols=2)

    colors = ['steelblue', 'gold']
    n_bins = 10
    x_multi = [dists['phy'], dists['ia']]
    ax0.hist(x_multi, n_bins, color=colors)
    ax0.set_title('Average Error')
    ax0.legend(['Physic error', 'IA error'])

    tot = mean(dists['phy']) + mean(dists['ia'])
    bar = ax1.bar(['physic', 'IA'], [mean(dists['phy']) * 100 / tot, mean(dists['ia']) * 100 / tot], color=colors)
    ax1.bar_label(bar, label_type='center')
    ax1.set_title('Relative Error %')

    #ax1.text(f"{(tot * 100)/mean(dists['phy'])}%")

    plt.show()

def PlotTrajs(trajs):
    """
    Plot the comparison of the real traj, the AI and the physic's one.
    :param trajs: (Real traj, Physic traj, AI traj)
    :return:
    """
    graph, plots = plt.subplots(1, len(trajs) + 1)

    y_lim = (0.75 * min(trajs[0][1]), 1.5 * max(trajs[0][1]))
    x_lim = (0.75 * min(trajs[0][0]), 1.5 * max(trajs[0][0]))

    for i in range(len(plots) - 1):
        plots[i].scatter(trajs[i][0], trajs[i][1], color=((0, 0, 0) if i==0 else ((0, 0.7, 0) if i==2 else (1, 0, 0))))
        plots[i].set_ylim(y_lim)
        plots[i].set_xlim(x_lim)
        plots[i].invert_yaxis()

    for i in range(len(trajs)):
        if i != 0:
            plots[-1].plot(trajs[i][0], trajs[i][1], color=((0, 0, 0) if i==0 else ((0, 0.7, 0) if i==2 else (1, 0, 0))), linewidth=2)
        else:
            plots[-1].scatter(trajs[i][0], trajs[i][1], color=((0, 0, 0) if i==0 else ((0, 0.7, 0) if i==2 else (1, 0, 0))))
    plots[-1].set_ylim(y_lim)
    plots[-1].set_xlim(x_lim)
    plots[-1].invert_yaxis()
    plots[-1].legend([f"Real points",
                      f"Physique",
                      f"IA"])
    plots[0].set_title("Real Traj")
    plots[1].set_title("Physic Traj")
    plots[2].set_title("IA Traj")
    plots[3].set_title("Superposition")

    plt.show()
    return

def PlotsExtractTraj():
    """
    Plot all the trajectories.
    :return:
    """
    files_names = getFilesNames("./set_plane_2_extract")
    files_names += getFilesNames("./set_plane_2_extract/original_traj")
    trajs = []
    for file in files_names:
        trajs.append(extractAnalyse_V2(file))
    graph, plots = plt.subplots(1, 1)
    for traj in trajs:
        plots.plot(traj['coordx'], traj['coordy'])
    plots.invert_yaxis()
    plt.show()

def PlotsAngleTraj(file=None):
    """
    Plot the traj and the angles
    :param file: traj to plot
    :return: None
    """
    file = getFilesNames("./set_plane_2_extract")[8] if file is None else file
    traj = extractAnalyse_V2(file)

    v_anglesx, v_anglesy = [], []
    for i in range(len(traj['coordx'])):
        x, y, angle = traj['coordx'][i], traj['coordy'][i], traj['angle'][i]
        v_anglesx.append((x, x + 0.4 * cos(radians(angle))))
        v_anglesy.append((y, y - 0.4 * sin(radians(angle))))
        #print((x, x + 50 * cos(radians(angle))), (y, y + 50 * sin(radians(angle))))

    graph, plots = plt.subplots(1, 1)

    plots.plot(traj['coordx'], traj['coordy'])
    for i in range(len(v_anglesx)):
        plots.plot(v_anglesx[i], v_anglesy[i], 'red', linewidth=0.8)

    plots.invert_yaxis()
    plt.show()

def PlotDataInfluence(pas=5):
    """
    Plot the relative error between AI and physic depending of the amount of training datas
    :param pas: step of percentage
    :return: None
    """
    def get_reg(split_percent):
        return getRegr(("./set_plane_2_extract",), split_percent=split_percent, shuffle_bool=True)

    split_percent_list = []
    error_percent_phy = []
    error_ia_phy = []
    for i in range(0, 101, pas):
        print(i, "% of datas...")
        split_pecent = i/100

        regr, normalisation_factors = get_reg(split_pecent)

        folders = ["./set_plane_2_extract"]
        files_names = []
        for folder in folders:
            files_names = files_names + getFilesNames(folder)
        files_names = files_names

        print('Measuring distances...')
        dists = MeasureDistances(files_names, regr, normalisation_factors)
        tot = mean(dists['phy']) + mean(dists['ia'])
        percent_phy, percent_ia = mean(dists['phy']) * 100 / tot, mean(dists['ia']) * 100 / tot

        split_percent_list.append(split_pecent)
        error_percent_phy.append(percent_phy)
        error_ia_phy.append(percent_ia)
        print("DONE.")

    def func_lin(x, m, n):
        return x*m + n

    m, n = polyfit(split_percent_list, error_ia_phy, 1)
    reg_percent = [p / 100 for p in range(0, 100)]
    error_ia_reg = [func_lin(x, m, n) for x in reg_percent]

    g, ax = plt.subplots(1, 1)
    ax.plot(split_percent_list, error_percent_phy, linewidth=2)
    ax.plot(split_percent_list, error_ia_phy, linewidth=2)
    ax.plot(reg_percent, error_ia_reg, linewidth=2, linestyle="--")
    ax.plot(reg_percent, [50 for i in range(len(reg_percent))], linewidth=2, linestyle="--")
    ax.legend(['Physique relative error', 'IA relative error', f'Regression {round(m, 2)}*p + {round(n, 2)}'])
    plt.show()

def getCxCz(p=1., display=True):
    MASSE = 0.1  # kg
    SURFACE = (0.2 * 0.3) / 2  # m^2
    RHO_AIR = 1.3  # kg.m^-3
    g = 9.81

    def norm(vec):
        return (vec[0]**2 + vec[1]**2)**0.5

    def getSpeed(trajectory, point):
        if 0 < point < len(trajectory['coordx']):
            dt = (trajectory['t'][point] - trajectory['t'][point - 1])
            vx = (trajectory['coordx'][point] - trajectory['coordx'][point - 1]) / dt
            vz = (trajectory['coordy'][point] - trajectory['coordy'][point - 1]) / dt
            return (vx, vz)
        else:
            print("Error of index")
            return (0, 0)

    def getAcceleration(trajectory, point):
        if 1 < point < len(trajectory['coordx']) - 1:
            dt = (trajectory['t'][point] - trajectory['t'][point - 1])
            v2, v1 = getSpeed(trajectory, point), getSpeed(trajectory, point - 1)
            return (norm(v2) - norm(v1)) / dt
        else:
            print("Error of index")
            return (0, 0)

    def getAngleSpeed(trajectory, point):
        if 0 < point < len(trajectory['angle']):
            dt = (trajectory['t'][point] - trajectory['t'][point - 1])
            angle_speed = (radians(trajectory['angle'][point]) - radians(trajectory['angle'][point - 1])) / dt
            return angle_speed
        else:
            print("Error of index")
            return 0

    def cx(vitesse, acceleration, theta):
        return -2 * MASSE * (acceleration + g * sin(theta)) / (RHO_AIR * SURFACE * (vitesse ** 2))
    def cz(vitesse, theta_point, theta):
        return -2 * MASSE * (speed * theta_point - g * cos(theta)) / (RHO_AIR * SURFACE * (vitesse ** 2))

    folders = ["./set_plane_2_extract"]
    files_names = []
    for folder in folders:
        files_names = files_names + getFilesNames(folder)
    files_names = files_names[:int(len(files_names) * p)]

    Cx, Cz = [], []
    for trajectory_name in files_names:
        trajectory = extractAnalyse_V2(trajectory_name)
        last_theta_point, last_accel, last_speed = [], [], []

        for i in range(2, len(trajectory['coordx']) - 1):
            dt = (trajectory['t'][i] - trajectory['t'][i - 1])
            speed = np.mean([norm(getSpeed(trajectory, i))] + last_speed[len(last_speed) - 5:])
            accel = (speed - last_speed[-1])/dt if len(last_speed) > 1 else getAcceleration(trajectory, i)
            theta_point = np.mean([getAngleSpeed(trajectory, i)] + last_theta_point[len(last_theta_point) - 5:])
            theta = radians(trajectory['angle'][i])

            Cx.append(cx(speed, accel, theta))
            Cz.append(cz(speed, theta_point, theta))

            last_accel.append(accel)
            last_speed.append(speed)
            last_theta_point.append(theta_point)

    if display:
        graph, ((ax0), (ax1)) = plt.subplots(nrows=1, ncols=2)

        colors = ['red']
        n_bins = 30
        x_multi = [Cx]
        ax0.hist(x_multi, n_bins, color=colors)
        ax0.set_xlabel("Cx")
        ax0.set_ylabel("Proportion")
        ax0.set_title('Cx')
        ax0.legend([f'Cx mean : {round(np.mean(Cx), 4)} +- {round(np.std(Cx)/np.sqrt(len(Cx)), 4)}'])

        colors = ['blue']
        n_bins = 30
        x_multi = [Cz]
        ax1.hist(x_multi, n_bins, color=colors)
        ax1.set_xlabel("Cz")
        ax1.set_ylabel("Proportion")
        ax1.set_title('Cz')
        ax1.legend([f'Cz mean : {round(np.mean(Cz), 4)} +- {round(np.std(Cz)/np.sqrt(len(Cz)), 4)}'])

        plt.show()
    return round(np.mean(Cx), 4), round(np.mean(Cz), 4)

def PlotCxCzDatainfluence():
    Cx, Cz, propor = [], [], []
    for i in range(1, 100):
        print(f"{i}%...")
        p = i/100
        cx, cz = getCxCz(p=p, display=False)
        Cx.append(cx)
        Cz.append(cz)
        propor.append(p)
    graph, ((ax0), (ax1)) = plt.subplots(nrows=1, ncols=2)
    ax0.plot(propor, Cx)
    ax1.plot(propor, Cz)
    plt.show()

#getCxCz()
#PlotDataInfluence(pas=5)
#PlotCxCzDatainfluence()
#PlotsAngleTraj()

"""
print('Training ia...')
regr, normalisation_factors = getRegr(("./set_plane_2_extract",), split_percent=1.)
print('Done.')
#"""
"""
folders = ["./set_plane_2_extract"]

files_names = []
for folder in folders:
    files_names = files_names + getFilesNames(folder)
files_names = files_names

print('Measuring distances...')
dists = MeasureDistances(files_names, regr, normalisation_factors)
print('Done.')
PlotDistance(dists)
#"""

# 2024-02-05 16:29:15.txt IA
# 2024-02-05 16:25:34.txt Close
# 2024-02-05 16:33:54.txt Phy
#comparison = CompareBased('./set_plane_2_extract/2024-02-05 16:33:54.txt', regr, normalisation_factors)
#PlotTrajs(comparison)
