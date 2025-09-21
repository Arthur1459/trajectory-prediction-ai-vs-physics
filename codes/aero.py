import matplotlib.pyplot as plt
from tools import *

# ----- Global ----- #
MASSE = 0.1 # kg
SURFACE = (0.2 * 0.3) / 2 # m^2
XY0 = (0.56, 0.326) # m
V0 = (2., -1.) # m.s^-1
ANGLE0, ANGLE_TRESHOLD = rad(0), rad(5)
RHO_AIR = 1.2 # kg.m^-3
g = 9.81
T_simul = 2

# Drag Coefficient
CX = 0.77
CZ = 3.01

# ------------------------------------ #
def force_aero(angle, speed):

    S = SURFACE
    Cx, Cz = CX, CZ

    Fu = -0.5 * RHO_AIR * S * Cx * (speed**2)
    Fw = 0.5 * RHO_AIR * S * Cz * (speed**2)

    Fx = Fu * cos(angle) + Fw * sin(angle)
    Fz = - Fu * sin(angle) - Fw * cos(angle)

    return (Fx, Fz)

def acceleration(angle, speed):
    aero = force_aero(angle, speed)
    poid = (0, 1 * MASSE * g)
    accel = v_mult(big_vect_sum([aero, poid]), 1/MASSE)
    angle_speed = (g*cos(rad(angle)) - (RHO_AIR * SURFACE * speed**2)/(2*MASSE))/speed
    return accel, angle_speed

def vect_to_angle(vect):
    return atan(vect[1] * inv(vect[0]))

def angles_to_vects(angles, norm=1):
    vect = []
    for angle in angles:
        vect.append((norm * cos(angle), norm * sin(angle)))
    return vect

def integrate(p, v, angle, dt, delta_t, nb_pts=-1):
    t = 0
    trajectory = {"coordx": [p[0]], "coordy": [p[1]], "angle": [angle], "timestamp": [0]}
    while t < delta_t and (len(trajectory["coordx"]) < nb_pts if nb_pts != -1 else True):
        t += dt
        accel, angle_accel = acceleration(angle, norm(v))
        v = vect_sum(v, v_mult(accel, dt))
        p = vect_sum(p, v_mult(v, dt))
        trajectory["coordx"].append(p[0])
        trajectory["coordy"].append(p[1])
        angle = angle + deg(angle_accel * dt)
        trajectory["angle"].append(angle)
        trajectory["timestamp"].append(t)
    return trajectory

def displayIntegrate():
    traj = integrate(XY0, V0, ANGLE0, 0.01, T_simul)

    graph, p1 = plt.subplots(1, 1)

    p1.axes.invert_yaxis()
    p1.plot(traj["coordx"], traj["coordy"], color='blue')

    plt.show()
    return

#displayIntegrate()

""