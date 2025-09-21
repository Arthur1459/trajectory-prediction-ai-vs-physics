import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from random import shuffle
import glob

# -- V1 -- #

"""
def extractAnalyse(file_name):
    file = open(file_name, "r")
    content = file.readlines()
    file.close()

    datas = {"coord": [], "t": []}
    for line in content:
        line_unwrap = line.split(sep="||")
        try:
            x = float(line_unwrap[0].replace("\n", ""))
            y = float(line_unwrap[1].replace("\n", ""))
            t = float(line_unwrap[2].replace("\n", ""))
            datas["coord"].append([x, y])
            datas["t"].append(t)
        except:
            continue
    return datas

def getDistinctRndIndex(nb, bound, start=0):
    if bound - start < nb:
        print("ERROR in getDistinctRndIndex()")
        return []
    index = []
    for cpt in range(nb):
        new_index = randint(start, bound)
        while new_index in index:
            new_index = randint(start, bound)
        index.append(new_index)
    return index

def NewSample(datas, size=3, pts=None):
    index = getDistinctRndIndex(size + 1, len(datas["coord"]) - 1) if pts is None else pts
    sample = []
    for cpt in range(len(index) - 1):
        sample.append(datas["coord"][index[cpt]][0])
        sample.append(datas["coord"][index[cpt]][1])
        sample.append(datas["t"][index[cpt]])

    t_target = datas['t'][index[len(index) - 1] if pts is None else pts[3]]
    return sample + [t_target], datas["coord"][index[len(index) - 1]]

def MakeSamples(datas, points=None, nb_samples=None,  sample_size=3):
    X, y = [], []
    for i in range(nb_samples if (points is None or (nb_samples is not None and nb_samples < len(points))) else len(points)):
        if points is not None:
            new_sample = NewSample(datas, pts=[points[i][0], points[i][1], points[i][2], points[i][3]])
        else:
            new_sample = NewSample(datas, size=sample_size)
        X.append(new_sample[0])
        y.append(new_sample[1])
    return np.array(X), np.array(y)

def SelectPoints(data, offset=0, periode=2, random=False, nb_samples=None, sample_size=4):
    points = []
    if random is False:
        if len(data["coord"]) >= periode:
            for i in range(0, (len(data["coord"]) - 3)//periode):
                pt = [offset + size + i*periode for size in range(sample_size)]
                points.append(pt)
            return points
        else:
            return []
    else:
        if len(data["coord"]) >= periode:
            for i in range(0, len(data["coord"]) if nb_samples is None else nb_samples):
                i1, i2, i3, target = getDistinctRndIndex(sample_size, len(data["coord"]) - 1)
                pt = (offset + i1, offset + i2, offset + i3, offset + target)
                points.append(pt)
            return points
        
def normaliseSample(pt1, a1, t1, pt2, a2, t2, pt3, a3, t3, pt, at, t):
    Mx, mx, My, my = max([pt1[0], pt2[0], pt3[0]]), min([pt1[0], pt2[0], pt3[0]]), max([pt1[1], pt2[1], pt3[1]]), min([pt1[1], pt2[1], pt3[1]])

    x1, y1 = (pt1[0] - mx) / (Mx - mx), (pt1[1] - my) / (My - my)
    x2, y2 = (pt2[0] - mx) / (Mx - mx), (pt2[1] - my) / (My - my)
    x3, y3 = (pt3[0] - mx) / (Mx - mx), (pt3[1] - my) / (My - my)

    x_target, y_target = (pt[0] - mx) / (Mx - mx), (pt[1] - my) / (My - my)
    return x1, y1, a1, 0, x2, y2, a2, t2 - t1, x3, y3, a3, t3 - t1, x_target, y_target, at, t - t1
"""

# -- V2 -- #
def extractAnalyse_V2(file_name) -> dict:
    file = open(file_name, "r")
    content = file.readlines()
    file.close()

    traj = {"coord": [], "angle": [], "t": []}
    for line in content:
        line_unwrap = line.split(sep="||")
        try:
            x = float(line_unwrap[0].replace("\n", ""))
            y = float(line_unwrap[1].replace("\n", ""))
            angle = float(line_unwrap[2].replace("\n", ""))
            t = float(line_unwrap[3].replace("\n", ""))
            traj["coord"].append([x, y])
            traj["angle"].append(angle)
            traj["t"].append(t)
        except:
            print(f"Error while reading file : {file_name}")
            continue
    return traj

"""
def normaliseSampleV2(datas):
    Mx, mx, My, my = max([datas[0], datas[4], datas[8]]), min([datas[0], datas[4], datas[8]]), max([datas[1], datas[5], datas[9]]), min([datas[1], datas[5], datas[9]])
    #print(Mx, My, mx, my)
    x1, y1 = (datas[0] - mx) / (Mx - mx), (datas[1] - my) / (My - my)
    x2, y2 = (datas[4] - mx) / (Mx - mx), (datas[5] - my) / (My - my)
    x3, y3 = (datas[8] - mx) / (Mx - mx), (datas[9] - my) / (My - my)
    get_coord = lambda pt: (pt[0] * (Mx - mx) + mx, pt[1] * (My - my) + my)

    return [x1, y1, datas[2], 0, x2, y2, datas[6], datas[7] - datas[3], x3, y3, datas[10], datas[11] - datas[3], datas[12] - datas[3]], get_coord
"""

# -- V3 -- #

def getTrajectories(files) -> list:
    trajectories = []
    for file in files:
        trajectories.append(extractAnalyse_V2(file))
    return trajectories

def normaliseSampleV3(sample, factors):
    x1, y1, a1, t1, x2, y2, a2, t2, x3, y3, a3, t3, t = sample
    x_max, x_min, y_max, y_min = factors

    x1 = (x1 - x_min) / (x_max - x_min)
    x2 = (x2 - x_min) / (x_max - x_min)
    x3 = (x3 - x_min) / (x_max - x_min)

    y1 = (y1 - y_min) / (y_max - y_min)
    y2 = (y2 - y_min) / (y_max - y_min)
    y3 = (y3 - y_min) / (y_max - y_min)

    get_coord = lambda pt: (pt[0] * (x_max - x_min) + x_min, pt[1] * (y_max - y_min) + y_min)

    return (x1, y1, a1, 0, x2, y2, a2, t2 - t1, x3, y3, a3, t3 - t1, t - t1), get_coord

def MakeSet(trajectories):
    set, targets = [], []
    for traj in trajectories:
        # Traj : {'coord', 'angle', 't'}
        for i in range(1, len(traj['coord'])//3 - 1):
            x1, y1 = traj['coord'][3*i + 0]
            x2, y2 = traj['coord'][3*i + 1]
            x3, y3 = traj['coord'][3*i + 2]
            angle1, angle2, angle3 = traj['angle'][3*i + 0], traj['angle'][3*i + 1], traj['angle'][3*i + 2]
            t1, t2, t3 = 0, traj['t'][3*i + 1] - traj['t'][3*i + 0], traj['t'][3*i + 2] - traj['t'][3*i + 0]
            x_t, y_t = traj['coord'][3*i + 3]
            angle_target = traj['angle'][3*i + 3]
            t_target = traj['t'][3*i + 3] - traj['t'][3*i + 0]

            set.append((x1, y1, angle1, t1, x2, y2, angle2, t2, x3, y3, angle3, t3, t_target))
            targets.append((x_t, y_t, angle_target, t_target))
    return set, targets

def extractCoordFromPrediction(predictions):
    coordx, coordy = [], []
    for sample in predictions:
        coordx.append(sample[0])
        coordy.append(sample[1])
    return coordx, coordy

def extractCoordFromSample(sample):
    x1, y1, a1, t1, x2, y2, a2, t2, x3, y3, a3, t3, t = sample
    sample_coordx, sample_coordy = [], []

    sample_coordx.append(x1)
    sample_coordx.append(x2)
    sample_coordx.append(x3)

    sample_coordy.append(y1)
    sample_coordy.append(y2)
    sample_coordy.append(y3)
    return sample_coordx, sample_coordy

def extractCoordsFromSet(set) -> tuple | tuple:
    coordx, coordy = [], []
    for sample in set:
        sample_coordx, sample_coordy = extractCoordFromSample(sample)
        coordx.append(sample_coordx)
        coordy.append(sample_coordy)
    return coordx, coordy

# ------- #

def getFilesNames(directory, file_extension="txt"):
    return [f for f in glob.glob(directory + "/*." + file_extension)]

# --------------------------------------------------------- #

def Train(X_train, y_train, net_size=(50, 50,), show_datas=False):

    if show_datas is True:
        plt.figure()
        plt.axis((-0.2, 1.5, 1.5, -0.2))
        for u in range(3):
            plt.scatter([X_train[i][0 + u*4] for i in range(len(X_train))], [X_train[i][1 + u*4] for i in range(len(X_train))], color="orange")
        plt.scatter([y_train[i][0] for i in range(len(y_train))], [y_train[i][1] for i in range(len(y_train))], color="green")

        for i in range(len(y_train)):
            for u in range(3):
                plt.text(X_train[i][0 + 4 * u], X_train[i][1 + 4 * u], str(u))
            plt.text(y_train[i][0], y_train[i][1], '3')
        plt.show()

    network_size = net_size
    regr = MLPRegressor(hidden_layer_sizes=network_size, random_state=3, max_iter=12000,
                        tol=1e-12, activation="logistic", solver='adam', learning_rate='adaptive',
                        shuffle=False, epsilon=1e-8).fit(X_train, y_train)
    return regr

# -- auto reg -- #

"""
def auto_reg(regr, x1, y1, t1, x2, y2, t2, x3, y3, t3, dt, nb_pts=20):
    sample_initial = [x1, y1, t1, x2, y2, t2, x3, y3, t3, 3*dt]
    X_test_manual = [sample_initial]
    auto_reg = [[x1, y1], [x2, y2], [x3, y3]]
    for i in range(nb_pts):
        next_pt = regr.predict([X_test_manual[-1]])
        auto_reg.append(list(next_pt[0]))
        X_test_manual.append([X_test_manual[-1][3], X_test_manual[-1][4], X_test_manual[-1][5], X_test_manual[-1][6], X_test_manual[-1][7], X_test_manual[-1][8], next_pt[0][0], next_pt[0][1], X_test_manual[-1][9], X_test_manual[-1][9] + dt])
    return auto_reg
"""

def auto_reg_V2(regr, normalisation_factors, pts_i, dt, nb_pts=20, already_normalised=False):
    if not already_normalised:
        sample_init, _ = normaliseSampleV3(pts_i, normalisation_factors)
    else:
        sample_init = pts_i

    X_auto_reg = [sample_init]
    coordx, coordy = extractCoordFromSample(sample_init)
    auto_reg = [(coordx[0], coordy[0]), (coordx[1], coordy[1]), (coordx[2], coordy[2])]

    while len(X_auto_reg) < nb_pts:
        x, y, angle, t = regr.predict([X_auto_reg[-1]])[0]
        auto_reg.append((x, y))

        x1, y1, a1, t1, x2, y2, a2, t2, x3, y3, a3, t3, t = X_auto_reg[-1]
        new_sample = (x2, y2, a2, 0, x3, y3, a3, t3 - t2, x, y, angle, t - t2, t - t2 + dt)
        X_auto_reg.append(new_sample)

    return auto_reg

"""
def make_auto_reg(regr, X_base, n_factors, nb=2, random=True, dt=None):
    auto_regs = []
    for i in range(nb):
        u = randint(0, len(X_base) - 1)
        dt = X_base[u][12] - X_base[u][11]
        auto_regs.append(auto_reg_V2(regr, n_factors, X_base[u], dt, 30, already_normalised=True))
    return auto_regs
"""

def getRegr(training_folders=("./set_plane_2_extract"), split_percent=0.75, shuffle_bool=False):
    files_names = []
    for folder in training_folders:
        files_names = files_names + getFilesNames(folder)
    if shuffle_bool:
        shuffle(files_names)
    trajectories, normalisation_factors = NormaliseTrajectories(getTrajectories(files_names))  # List of dict
    train_traj, test_traj = trajectories[:int(max(1, len(trajectories) * split_percent))], trajectories[int(max(len(trajectories) * split_percent, 1)):]

    X_train, y_train = MakeSet(train_traj)

    return Train(X_train, y_train), normalisation_factors

def ConvertToRealCoord(points, n_factors) -> list[tuple]:
    coord = []
    x_max, x_min, y_max, y_min = n_factors
    def convert(x, y):
        return x * (x_max - x_min) + x_min, y * (y_max - y_min) + y_min

    for x, y in points:
        coord.append(convert(x, y))

    return coord

def ConvertToRealCoord_x(x_list, n_factors) -> list:
    coordx = []
    x_max, x_min, y_max, y_min = n_factors

    def convert(x):
        return x * (x_max - x_min) + x_min

    for x in x_list:
        coordx.append(convert(x))

    return coordx

def ConvertToRealCoord_y(y_list, n_factors) -> list:
    coordy = []
    x_max, x_min, y_max, y_min = n_factors

    def convert(y):
        return y * (y_max - y_min) + y_min

    for y in y_list:
        coordy.append(convert(y))

    return coordy

def NormaliseTrajectories(trajs) -> list | tuple:
    coordx, coordy = [], []
    for traj in trajs:
        for x, y in traj['coord']:
            coordx.append(x)
            coordy.append(y)
    x_max, x_min = 0.8 * max(coordx), 1.2 * min(coordx)
    y_max, y_min = 0.8 * max(coordy), 1.2 * min(coordy)

    for traj in trajs:
        for i, coord in enumerate(traj['coord']):
            traj['coord'][i][0] = (coord[0] - x_min) / (x_max - x_min)
            traj['coord'][i][1] = (coord[1] - y_min) / (y_max - y_min)
            traj['t'][i] = traj['t'][i] - traj['t'][0]

    return trajs, (x_max, x_min, y_max, y_min)

# ------------------------------------------ #

def operator():
    debug = True

    training_folder = ["./set_plane_2_extract", "./set_plane_2_extract/original_traj"]
    files_names = []
    for folder in training_folder:
        files_names = files_names + getFilesNames(folder)
    trajectories = getTrajectories(files_names) # List of dict
    normalised_trajs, normalisation_factors = NormaliseTrajectories(trajectories)
    train_traj, test_traj = trajectories[:2 * len(normalised_trajs)//3], normalised_trajs[2 * len(normalised_trajs)//3:]

    X_train, y_train = MakeSet(train_traj)
    X_test, y_test = MakeSet(test_traj)

    regr = Train(X_train, y_train)
    loss = regr.loss_curve_

    dt = 0.02
    sample_i = [0.7, 0.3, 0, 1.75, 1., 0.43, -19, 1.8, 1.16, 0.54, -29, 1.85, 1.85 + dt]
    auto_reg_test = auto_reg_V2(regr, normalisation_factors, sample_i, dt, nb_pts=20)

    auto_regs = [auto_reg_test] #make_auto_reg(regr, X_test, normalisation_factors, nb=4)

    y_test_answer = regr.predict(X_test)
    y_train_answer = regr.predict(X_train)
    Train_Score = regr.score(X_train, y_train)
    Test_Score = regr.score(X_test, y_test)
    print("Train Score : ", Train_Score)
    print("Test Score : ", Test_Score)

    graph, (p1, p2, p3, p4) = plt.subplots(1, 4)

    p1.axes.invert_yaxis()

    in_x, in_y = extractCoordsFromSet(X_train)
    trajx, trajy = extractCoordFromPrediction(y_train)
    for i in range(len(in_x)):
        p1.scatter(ConvertToRealCoord_x(in_x[i], normalisation_factors), ConvertToRealCoord_y(in_y[i], normalisation_factors), color="orange")
    p1.scatter(ConvertToRealCoord_x(trajx, normalisation_factors), ConvertToRealCoord_y(trajy, normalisation_factors), color="green")

    traj_predictx, traj_predicty = extractCoordFromPrediction(y_train_answer)
    p1.scatter(ConvertToRealCoord_x(traj_predictx, normalisation_factors), ConvertToRealCoord_y(traj_predicty, normalisation_factors), color="red")

    p1.set_title(f"Train Score : {round(Train_Score, 3)}", fontstyle='italic')

    p2.axes.invert_yaxis()

    in_x, in_y = extractCoordsFromSet(X_test)
    trajx, trajy = extractCoordFromPrediction(y_test)
    for i in range(len(in_x)):
        p2.scatter(ConvertToRealCoord_x(in_x[i], normalisation_factors), ConvertToRealCoord_y(in_y[i], normalisation_factors), color="orange")
    p2.scatter(ConvertToRealCoord_x(trajx, normalisation_factors), ConvertToRealCoord_y(trajy, normalisation_factors), color="green")

    traj_predictx, traj_predicty = extractCoordFromPrediction(y_test_answer)
    p2.scatter(ConvertToRealCoord_x(traj_predictx, normalisation_factors), ConvertToRealCoord_y(traj_predicty, normalisation_factors), color="red")

    p2.set_title(f'Test Score {round(Test_Score, 3)}', fontstyle='italic')

    colors = ("red", "purple", "yellow", "cyan")
    p3.axes.invert_yaxis()
    for u, auto_reg in enumerate(auto_regs):
        auto_reg = ConvertToRealCoord(auto_reg, normalisation_factors)
        p3.plot([auto_reg[i][0] for i in range(len(auto_reg))], [auto_reg[i][1] for i in range(len(auto_reg))], color=colors[u])
        p3.scatter([auto_reg[0][0], auto_reg[1][0], auto_reg[2][0]], [auto_reg[0][1], auto_reg[1][1], auto_reg[2][1]], color=colors[u])

    p4.plot(loss)

    plt.show()
    return

debug = False
#operator()
