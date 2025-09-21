import glob
import math
import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO
import tools
from tools import *

# --------------- constant ------------------- #
# Boule Exp 20/09/23 : 500px -> 40 cm = 0.40 m donc 1px -> 0.40 / 500 = 0.0008 m/px
# Data Set 2 : 250 -> 22 cm = 0.22 donc 1px -> 0.22 / 250 = 0.00088 m/px
# Data Set Plane 1 : 453 -> 84.5 cm = 0.845 m donc 1px -> 0.845 / 453 = 0.00189 m/px
# Data Set Plane 2 : 477 -> 87.5 cm = 0.875 m donc 1px -> 0.875 / 477 = 0.00183 m/px

scale = 0.00183  # m/px 89
fps = 1 / 60  # s/frame
x_treshold = 150
y_treshold = 1000
object_size_min = 500
object_size_max = 5000
CONFIDENCE_THRESHOLD = 0.2
# -------------------------------------------- #
MASSE = 0.1 # kg
SURFACE = (0.2 * 0.3) / 2 # m^2
RHO_AIR = 1.2 # kg.m^-3
g = 9.81
# -------------------------------------------- #

def detect_object(video_path, object_size_min=object_size_min, object_size_max=object_size_max, delay=0., display=False, dt=fps, x_treshold=x_treshold, y_treshold=y_treshold, disable_prog=False):
    trajectory = {'coordx': [], 'coordy': [], 'timestamp': []}

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create a background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=10)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialise variables
    current_frame = 0
    current_time = 0
    first_frame = None
    multiplicator = 1
    traj_start, traj_end = False, False
    end = False

    # video processing frame by frame
    for i in tqdm(range(total_frames), unit='frames', desc='Frames processing', disable=disable_prog):
        # Read the current frame from the video
        still, frame = video.read()
        if first_frame is None:
            first_frame = frame.copy()  # For a fancy graph display

        current_frame += 1
        current_time += dt

        if not still:
            # End of video
            end = True
            continue

        # --- threshold red ---
        lower = np.array([0, 0, 0])
        upper = np.array([40, 40, 255])
        thresh = cv2.inRange(frame, lower, upper)
        result = frame.copy()
        result[thresh != 255] = (255, 255, 255)

        # Apply background subtraction to obtain a mask
        mask = bg_subtractor.apply(result)  # Else frame for just movement

        # Apply some morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours and current_frame > 0:
            biggest = cv2.contourArea(contours[0])
            index = 0
            for contour in contours:
                if cv2.contourArea(contour) < object_size_min or object_size_max < cv2.contourArea(contour):
                    if display:
                        cv2.destroyAllWindows()
                        cv2.imshow("Object Detection Processing..." + str(current_frame) + " / " + str(total_frames),
                                   frame)
                        continue
                    if cv2.waitKey(1) == ord('q'):
                        # Exit if 'q' is pressed
                        continue
                    continue
                if cv2.contourArea(contour) >= biggest:
                    try:
                        index = contours.index(contour)
                        biggest = cv2.contourArea(contour)
                    except:
                        pass

            # Calculate the bounding box of the biggest contour found
            x, y, w, h = cv2.boundingRect(contours[index])

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center coordinates of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Points Filtering
            cv2.rectangle(frame, (0, 0), (x_treshold, int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))), (255, 0, 0), 2)
            cv2.rectangle(frame, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) - (x_treshold // 2), 0), (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))), (255, 0, 0), 2)
            cv2.rectangle(frame, (0, y_treshold), (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), y_treshold), (255, 0, 0), 2)

            # Display the frame with bounding boxes
            if display:
                numpy_concat = np.concatenate((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), axis=0)
                cv2.destroyAllWindows()
                cv2.imshow("Object Detection Processing..." + str(current_frame) + " / " + str(total_frames), numpy_concat)
            if cv2.waitKey(1) == ord('q'):
                # Exit if 'q' is pressed
                continue

            if center_x < x_treshold or int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) - (x_treshold / 2) < center_x:
                if not disable_prog:
                    print("OUT OF BORDER.")
                continue
            else:
                traj_start = True

            if len(trajectory['coordx']) > 4:
                try:
                    dist_x, dist_y = abs(center_x - trajectory['coordx'][-1]), abs(center_y - trajectory['coordy'][-1])

                    if dist_x - 50 > abs(trajectory['coordx'][-1] - trajectory['coordx'][len(trajectory['coordx']) - 2]) * 2 * multiplicator\
                    or dist_y - 50 > abs(trajectory['coordy'][-1] - trajectory['coordy'][len(trajectory['coordy']) - 2]) * 2 * multiplicator:

                        multiplicator = multiplicator * 2
                        if not disable_prog:
                            print("Next Point to far. POINT REJECTED")
                        continue
                except:
                    print("Error unexpected.")
                    continue

            if center_y > y_treshold and traj_start or traj_end:
                if len(trajectory['coordy']) > 4:
                    traj_end = True
                    break
                else:
                    continue

            if traj_start and traj_end is False:
                trajectory['coordx'].append(center_x)
                trajectory['coordy'].append(center_y)
                trajectory['timestamp'].append(current_time)
                multiplicator = 1

            if delay != 0 and end is False and display is True:
                time.sleep(delay)
        else:
            if display:
                cv2.destroyAllWindows()
                cv2.imshow("Object Detection Processing..." + str(current_frame) + " / " + str(total_frames), frame)
            if cv2.waitKey(1) == ord('q'):
                # Exit if 'q' is pressed
                break

    # Removing the First Point
    for i in range(1):
        trajectory['coordx'].pop(0)
        trajectory['coordy'].pop(0)
        trajectory['timestamp'].pop(0)
        trajectory['timestamp'] = [i - dt for i in trajectory['timestamp']]

    # Release the video and close windows
    video.release()
    cv2.destroyAllWindows()

    return trajectory, first_frame

# ----------------------------------------------------- #

# Return : Trajectory -> {'coordx', 'coordy', 'angle', 'timestamp'}, First_Frame -> Fancy graph

def detect_object_YOLO(video_path, delay=0., display=False, dt=fps, x_treshold=x_treshold,
                       y_treshold=y_treshold, disable_prog=False):
    trajectory = {'coordx': [], 'coordy': [], 'angle': [], 'timestamp': []}

    model = YOLO("yolov8n.pt")

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialise variables
    current_frame = 0
    current_time = 0
    first_frame = None
    multiplicator = 1
    traj_start, traj_end = False, False
    end = False

    # video processing frame by frame
    for i in tqdm(range(total_frames), unit='frames', desc='Frames processing', disable=disable_prog):
        # Read the current frame from the video
        still, frame = video.read()
        if first_frame is None:
            first_frame = frame.copy()  # For a fancy graph display

        current_frame += 1
        current_time += dt

        if display:
            cv2.destroyAllWindows()
            cv2.imshow("Object Detection Processing..." + str(current_frame) + " / " + str(total_frames), frame)

        if not still:
            # End of video
            end = True
            continue

        detections = model(frame, verbose=False)[0].boxes.data.tolist()
        if len(detections) == 0:
            continue
        main_detection = detections[0]

        x, y, w, h, confidence = main_detection[0], main_detection[1], main_detection[2] - main_detection[0], \
                                 main_detection[3] - main_detection[1], main_detection[4]
        x, y, w, h = int(x), int(y), int(w), int(h)
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Calculate the coord of the plane's nose
        head_x = x + w
        head_y = y + h

        # Calculate the angle  (angle between nose coord and last nose coord)
        if len(trajectory['coordx']) > 0:
            Delta_pos = (head_x - trajectory['coordx'][-1], -1 * (head_y - trajectory['coordy'][-1]))
            angle = atan(Delta_pos[1] * inv(Delta_pos[0]))
        else:
            angle = 0

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (head_x, head_y), (0, 255, 0), 2)

        # Draw the angle detected
        if len(trajectory['coordx']) != 0:
            angle_vect = angle_to_vect(angle, norm=50)
            cv2.line(frame, (head_x, head_y), (head_x + int(angle_vect[0]), head_y + int(angle_vect[1])),
                     (0, 0, 255), thickness=2)

        # Points Filtering
        cv2.rectangle(frame, (0, 0), (x_treshold, int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))), (255, 0, 0), 2)
        cv2.rectangle(frame, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) - (x_treshold // 2), 0),
                      (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                      (255, 0, 0), 2)
        cv2.rectangle(frame, (0, y_treshold), (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), y_treshold), (255, 0, 0), 2)

        # Display the frame with bounding boxes
        if display:
            cv2.destroyAllWindows()
            cv2.imshow("Object Detection Processing..." + str(current_frame) + " / " + str(total_frames), frame)
        if cv2.waitKey(1) == ord('q'):
            # Exit if 'q' is pressed
            continue

        if len(trajectory['coordx']) > 4:
            try:
                dist_x, dist_y = abs(head_x - trajectory['coordx'][-1]), abs(head_y - trajectory['coordy'][-1])

                if dist_x - 50 > abs(trajectory['coordx'][-1] - trajectory['coordx'][len(trajectory['coordx']) - 2]) * 2 * multiplicator \
                or dist_y - 50 > abs(trajectory['coordy'][-1] - trajectory['coordy'][len(trajectory['coordy']) - 2]) * 2 * multiplicator:
                    multiplicator = multiplicator * 2
                    if not disable_prog:
                        print("Next Point to far. POINT REJECTED")
                    continue
            except:
                print("Error unexpected.")
                continue

        if head_x < x_treshold or int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) - (x_treshold / 2) < head_x:
            if not disable_prog:
                print("OUT OF BORDER.")
            continue
        else:
            traj_start = True

        if (head_y > y_treshold and traj_start) or traj_end:
            if len(trajectory['coordy']) > 4:
                traj_end = True
                break
            else:
                continue

        if traj_start and traj_end is False:
            trajectory['coordx'].append(head_x)
            trajectory['coordy'].append(head_y)
            trajectory['angle'].append(angle)
            trajectory['timestamp'].append(current_time)
            multiplicator = 1

        if delay != 0 and end is False and display is True:
            time.sleep(delay)
        else:
            if display:
                cv2.destroyAllWindows()
                cv2.imshow("Object Detection Processing..." + str(current_frame) + " / " + str(total_frames), frame)
            if cv2.waitKey(1) == ord('q'):
                # Exit if 'q' is pressed
                break

    # Removing the First Point
    for i in range(1):
        trajectory['coordx'].pop(0)
        trajectory['coordy'].pop(0)
        trajectory['angle'].pop(0)
        trajectory['timestamp'].pop(0)
        trajectory['timestamp'] = [i - dt for i in trajectory['timestamp']]

    # Release the video and close windows
    video.release()
    cv2.destroyAllWindows()

    return trajectory, first_frame

# ----------------------------------------------------- #

def analyse(video=None, delay=0.1, display=False, save=True, save_name='default', save_folder=None, save_encoder=2,
            disable_prog=False):
    # Path to the video file
    video_path = video if video else './Test/test.mov'
    print("Analysing... : " + video)
    # detect objects in the video : return np.array(coordx), np.array(coordy), first_frame, nb_frames
    trajectory, first_frame = detect_object_YOLO(video_path, delay=delay, display=display, disable_prog=disable_prog)

    to_pop = []
    deltax = 0
    for point_ind in range(1, len(trajectory['coordx'])):
        if trajectory['coordx'][point_ind] - deltax <= trajectory['coordx'][point_ind - 1]:
            to_pop.append(point_ind)

        deltax = (trajectory['coordx'][point_ind] - trajectory['coordx'][point_ind - 1]) / 2

    offset = 0
    for ind in to_pop:
        trajectory['coordx'].pop(ind - offset)
        trajectory['coordy'].pop(ind - offset)
        trajectory['timestamp'].pop(ind - offset)
        offset += 1

    if save:
        name = ("" if save_name == 'default' else save_name + "-") + str(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
        file = ("./Datas/" if save_folder is None else save_folder) + name + ".txt"
        if save_encoder == 1:
            SaveAnalyse(trajectory, file=file)
            print("\nAnalyse Saved. as : " + name)
        elif save_encoder == 2:
            SaveAnalyse_V2(trajectory, file=file)
            print("\nAnalyse Saved. as : " + name)
        else:
            print("Error : Save encoder unavailable.")

    return trajectory, first_frame

def getSpeed(trajectory, point):
    if 0 < point < len(trajectory['coordx']):
        dt = (trajectory['timestamp'][point] - trajectory['timestamp'][point - 1])
        vx = (trajectory['coordx'][point] - trajectory['coordx'][point - 1]) / dt
        vy = (trajectory['coordy'][point] - trajectory['coordy'][point - 1]) / dt
        return (vx, vy)
    else:
        return (0, 0)

def getAcceleration(trajectory, point):
    if 1 < point < len(trajectory['coordx']) - 1:
        dt = (trajectory['timestamp'][point] - trajectory['timestamp'][point - 1])
        v2, v1 = getSpeed(trajectory, point), getSpeed(trajectory, point - 1)
        ax = (v2[0] - v1[0]) / dt
        ay = (v2[1] - v1[1]) / dt
        return (ax, ay)
    else:
        return (0, 0)

def norm(vect):
    return math.sqrt(vect[0]**2 + vect[1]**2)

def px_to_si(a, factor=scale):
    return a * factor  # a -> px | factor -> m/px => m

def si_to_px(a, factor=scale):
    return a / factor  # a -> m | factor -> m/px => px

def convert_vect_px_to_si(vect):
    return [px_to_si(i) for i in vect]

def convert_vect_si_to_px(vect):
    return [si_to_px(i) for i in vect]

def angles_to_vects(angles, norm=1):
    vect = []
    for angle in angles:
        vect.append((norm * cos(angle), -1 * norm * sin(angle)))
    return vect

def angle_to_vect(angle, norm=1):
    return (norm * cos(angle), -1 * norm * sin(angle))

def Discretise(function, bounds, points):
    x = np.linspace(bounds[0], bounds[1], points)
    y = []
    for i in x:
        y.append(function(i))
    return (x, y)
# ------------------------------------------------------------------------------------- #

def SaveAnalyse(traj_raw, file=None):
    if file is None:
        ask = int(input("Saving data ? 1 - YES | 2 - NO : "))
    else:
        ask = 1
    if ask == 1:
        file_path = "/Users/arthuroudeyer/Library/Mobile Documents/com~apple~CloudDocs/Documents - MacArthur/PycharmProjects/TIPE/PaperAirplane/Datas/" + str(input("Save Name : ")) + ".txt" if file is None else file
        file = open(file_path, 'w')
        lines = []
        for i in range(len(traj_raw['coordx'])):
            lines.append(str(round(px_to_si(traj_raw['coordx'][i]), 3)) + "||" + str(round(px_to_si(traj_raw['coordy'][i]), 3)) + "||" + str(round(traj_raw['timestamp'][i], 5)))
        for i in range(len(lines) - 1):
            file.write(lines[i] + "\n")
        file.write(lines[-1])
        file.close()

def SaveAnalyse_V2(traj_raw, file=None):
    if file is None:
        ask = int(input("Saving data ? 1 - YES | 2 - NO : "))
    else:
        ask = 1
    if ask == 1:
        file_path = "/Users/arthuroudeyer/Library/Mobile Documents/com~apple~CloudDocs/Documents - MacArthur/PycharmProjects/TIPE/PaperAirplane/Datas/" + str(input("Save Name : ")) + ".txt" if file is None else file
        file = open(file_path, 'w')
        lines = []
        for i in range(len(traj_raw['coordx'])):
            lines.append(str(round(px_to_si(traj_raw['coordx'][i]), 3)) + "||" + str(round(px_to_si(traj_raw['coordy'][i]), 3)) + "||" + str(round(deg(traj_raw['angle'][i]))) + "||" + str(round(traj_raw['timestamp'][i], 5)))
        for i in range(len(lines) - 1):
            file.write(lines[i] + "\n")
        file.write(lines[-1])
        file.close()

def DisplayAnalysis(video, save=False):

    traj_raw, frame = analyse(video=video, display=True, delay=0.1, save=save, save_folder="./Default_Saves/", disable_prog=False)

    bounds = (traj_raw['coordx'][0], traj_raw['coordx'][-1])

    # regression
    # noinspection PyTupleAssignmentBalance
    a, b, c, d, e = np.polyfit(traj_raw['coordx'], traj_raw['coordy'], deg=4)
    reg_str = f'{round(a, 3)}X^4 + {round(b, 3)}X^3 + {round(c, 3)}X^2 + {round(d, 3)}X + {round(e, 2)}'
    y_reg = [a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e for x in traj_raw['coordx']]
    reg_function = lambda x: a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e

    speeds = [getSpeed(traj_raw, i) for i in range(1, len(traj_raw['coordx']))]
    accels = [getAcceleration(traj_raw, i) for i in range(2, len(traj_raw['coordx']))]

    speeds_x, speeds_y = np.array([speed[0] for speed in speeds]), np.array([speed[1] for speed in speeds])
    accels_x, accels_y = np.array([accel[0] for accel in accels]), np.array([accel[1] for accel in accels])

    speed_x_reg_coefs = tuple(np.polyfit(traj_raw['timestamp'][:len(speeds_x)], speeds_x, deg=1))
    speed_y_reg_coefs = tuple(np.polyfit(traj_raw['timestamp'][:len(speeds_y)], speeds_y, deg=1))
    #print(f"{speed_y_reg_coefs[0] * scale} * X + {speed_y_reg_coefs[1] * scale}")
    speed_x_reg = [(speed_x_reg_coefs[0] * x + speed_x_reg_coefs[1]) * scale for x in traj_raw['timestamp']]
    speed_y_reg = [(speed_y_reg_coefs[0] * x + speed_y_reg_coefs[1]) * scale for x in traj_raw['timestamp']]

    # Angle
    angles = traj_raw['angle']
    angles_vect = angles_to_vects(angles, norm=50)

    # Parabolic
    top = traj_raw['coordy'].index(min(traj_raw['coordy']))
    index_top = top + 1

    dt = (traj_raw['timestamp'][index_top] - traj_raw['timestamp'][index_top - 1])
    vxo, vyo = (traj_raw['coordx'][index_top] - traj_raw['coordx'][index_top - 1]) / dt, (traj_raw['coordy'][index_top] - traj_raw['coordy'][index_top - 1]) / dt
    xo, yo = traj_raw['coordx'][index_top], traj_raw['coordy'][index_top]
    parab_mid = lambda x: 0.5 * (9.8 / scale) * ((x - xo)/(vxo * 1))**2 + vyo * ((x - xo)/(vxo * 0.9)) + yo
    parab_min = lambda x: 0.5 * (9.8 / scale) * ((x - xo)/(vxo * 0.9))**2 + vyo * ((x - xo)/(vxo * 0.9)) + yo
    parab_max = lambda x: 0.5 * (9.8 / scale) * ((x - xo)/(vxo * 1.1))**2 + vyo * ((x - xo)/(vxo * 1.1)) + yo

    parabolic_mid = Discretise(parab_mid, bounds, 100)
    parabolic_min = Discretise(parab_min, bounds, 100)
    parabolic_max = Discretise(parab_max, bounds, 100)

    def cx(index):
        return (2 * MASSE * accels_x[index]) / (RHO_AIR * SURFACE * sin(angles[index]) * (norm(speeds[index]) ** 2))
    def cz(index):
        return (2 * MASSE * (accels_y[index] + g)) / (RHO_AIR * SURFACE * cos(angles[index]) * (norm(speeds[index]) ** 2))

    Cx, Cz, times = [], [], []
    for index in range(len(accels_x)):
        Cx.append(cx(index))
        Cz.append(cz(index))
        times.append(traj_raw["timestamp"][index])

    # Data show
    speed_init = norm(getSpeed(traj_raw, 1))

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.scatter(traj_raw['coordx'], traj_raw['coordy'], label='Raw Datas', color='blue', linestyle='--')
    for i in range(len(traj_raw['coordx'])):
        plt.text(traj_raw['coordx'][i], traj_raw['coordy'][i], str(i))
    plt.plot(parabolic_mid[0], parabolic_mid[1], label='Parabolic Data', color='pink', linestyle='-.')
    plt.plot(parabolic_min[0], parabolic_min[1], label='Parabolic Data', color='purple', linestyle='-.')
    plt.plot(parabolic_max[0], parabolic_max[1], label='Parabolic Data', color='purple', linestyle='-.')
    plt.plot(traj_raw['coordx'], y_reg, label='Regression : ', color='yellow', linestyle=':')
    for i in range(len(traj_raw['coordx'])):
        angle_vect = [(traj_raw['coordx'][i], traj_raw['coordx'][i] + angles_vect[i][0]), (traj_raw['coordy'][i], traj_raw['coordy'][i] + angles_vect[i][1])]
        plt.plot(angle_vect[0], angle_vect[1], color='red')
    plt.text(10, 50, str(px_to_si(speed_init)) + " m/s", fontsize=10)
    plt.grid(True, color='black')
    plt.title('Object Trajectory')
    plt.show()

    t_init, t_end = traj_raw['timestamp'][0], traj_raw['timestamp'][-1]

    fig, axs = plt.subplots(3)

    axs[0].set_ylim([-10, 10])
    axs[0].plot(traj_raw['timestamp'][:len(speeds_x)], speeds_x * scale, color='blue')
    #axs[0].plot(traj_raw['timestamp'][:len(accels_x)], accels_x * scale, color='red')
    axs[0].plot(traj_raw['timestamp'][:len(speed_x_reg)], speed_x_reg, color="red", linestyle='-.')
    #axs[0].text(t_init, 8, f"accel reg : {round(speed_x_reg_coefs[0] * scale, 2)}*t + {round(speed_x_reg_coefs[1] * scale, 2)}", fontsize=10)
    axs[0].text(t_init, 12, "vx mean = " + str(round((speeds_x * scale).mean(), 2)) + " m/s", fontsize=10)
    axs[0].set_title("Horizontal Speed")

    axs[1].set_ylim([-10, 10])
    axs[1].plot(traj_raw['timestamp'][:len(speeds_y)], speeds_y * scale, color='blue')
    #axs[1].plot(traj_raw['timestamp'][:len(accels_y)], accels_y * scale, color='red')
    axs[1].plot(traj_raw['timestamp'][:len(speed_y_reg)], speed_y_reg, color="red", linestyle='-.')
    #axs[1].text(t_init, 8, f"accel reg : {round(speed_y_reg_coefs[0] * scale, 2)}*t + {round(speed_y_reg_coefs[1] * scale, 2)}", fontsize=10)
    axs[1].text(t_init, 12, "vy mean = " + str(round((speeds_y * scale).mean(), 2)) + " m/s", fontsize=10)
    axs[1].set_title("Vertical Speed")

    axs[2].set_ylim([-0.4, 0.4])
    axs[2].plot(times, Cx, color='blue')
    axs[2].plot(times, Cz, color='red')
    axs[2].text(min(times), 0.15, ("Cx mean = " + str(round(abs(moy(Cx)), 4))), fontsize=12)
    axs[2].text(min(times), 0.25, ("Cz mean = " + str(round(abs(moy(Cz)), 4))), fontsize=12)
    axs[2].set_title("Cx (BLUE) and Cz (RED)")

    plt.show()
    return

def AnalyseFolder(folder="./Data_set_plane_2/exp/"):
    files = glob.glob(folder + "/*.MOV")
    times = []
    for i in range(len(files)):
        print(f"\n--- Analysing file : {i + 1}/{len(files)} ---")
        t_start = time.time()
        analyse(video=files[i], display=True, delay=0, save=True, save_folder="./set_plane_2_extract/", disable_prog=False)
        times.append(time.time() - t_start)

        time_estimated = tools.moy(times) * (len(files) - (i + 1))
        hours = time_estimated // 3600
        minutes = (time_estimated - hours * 3600) // 60
        sec = (time_estimated - hours * 3600 - minutes * 60)
        print(f"--- Time left estimated : {int(hours)}h {int(minutes)}min {int(sec)}s ---")

def DisplayAnalyse():
    file_name = "IMG_2589.MOV"
    DisplayAnalysis(video="./Data_set_plane_2/exp/" + file_name, save=False)
    #analyse(video="./Data_set_plane_2/exp/" + file_name, display=True, delay=0.1, save=True,
    # save_folder="./Default_Saves/", disable_prog=False)

DisplayAnalyse()
