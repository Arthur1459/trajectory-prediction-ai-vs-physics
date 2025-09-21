from math import sqrt, cos, sin, pi, atan2, atan, log10

def norm(v1):
    n = 0
    for p in v1:
        n += p**2
    return sqrt(n)

def vect_sum(v1, v2):
    res = []
    for i in range(min(len(v1), len(v2))):
        res.append(0)
        res[i] = v1[i] + v2[i]
    return res

def big_vect_sum(vects):
    length = min([len(vects[u]) for u in range(len(vects))])
    res = [0 for i in range(length)]
    for vect in vects:
        for coord in range(length):
            res[coord] += vect[coord]
    return res

def vect_mult(v1, v2):
    res = []
    for i in range(min(len(v1), len(v2))):
        res.append(0)
        res[i] = v1[i] * v2[i]
    return res

def v_add(v, coef):
    res = []
    for i in range(len(v)):
        res.append(0)
        res[i] = v[i] + coef
    return res

def v_mult(v, coef):
    res = []
    for i in range(len(v)):
        res.append(0)
        res[i] = v[i] * coef
    return res

def scalar(v1, v2):
    if len(v1) != len(v2):
        print("\nERROR : Dot Product with two vector of different length.\n")
        return 0
    dot = 0
    for i in range(min(len(v1), len(v2))):
        dot += v1[i] * v2[i]
    return dot

def rad(x):
    return (2*pi)/360 * x

def deg(x):
    return 360/(2*pi) * x

def normal(v):
    return [v[1], -v[0]]

def s(a):
    if a >= 0:
        return 1
    else:
        return -1

def inv(x) -> float:
    if x != 0:
        return 1/x
    else:
        return 1e-9

def moy(array):
    if len(array) == 0:
        return 0
    tot = 0
    for elt in array:
        tot += elt
    return tot / len(array)

def v_abs(v):
    v_res = []
    for i in range(len(v)):
        v_res.append(abs(v[i]))
    return v_res

def normalize(v):
    return [v[0]/norm(v), v[1]/norm(v)]

def angle_bt(v1, v2):  # Marche pas
    v1, v2 = normalize(v1), normalize(v2)
    dx, dy = v2[0] - v1[0], v2[1] - v1[1]
    return deg(atan(dy / dx))

def affine(pt1, pt2, x):
    if (pt2[0] - pt1[0]) == 0:
        return (x - pt1[0]) * 1000000
    else:
        a = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        y0 = pt1[1] - a * pt1[0]
        return a * x + y0

def slope(pt1, pt2):
    if (pt2[0] - pt1[0]) == 0:
        return -(pt2[1] - pt1[1]) * 1000000
    else:
        return -(pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

def controlVect(v, value):
    for i in range(len(v)):
        if abs(v[i]) > value:
            v[i] = value * s(v[i])
    return v

def smart_control_vect(vect, control):
    out = False
    for i in range(len(control)):
        minimum, maximum = min_list(control[i]), max_list(control[i])
        if vect[i] <= minimum:
            out = True
            vect[i] = minimum
        elif vect[i] >= maximum:
            out = True
            vect[i] = maximum
    return out

def value_control(value, bounds):
    if value < bounds[0]:
        return bounds[0]
    elif value > bounds[1]:
        return bounds[1]
    else:
        return value

def max_list(l):
    if len(l) == 0:
        return None
    res = l[0]
    for elt in l:
        if elt > res:
            res = elt
    return res

def min_list(l):
    if len(l) == 0:
        return None
    res = l[0]
    for elt in l:
        if elt < res:
            res = elt
    return res

def vect_round(vect, param):
    for i in range(len(vect)):
        if isinstance(vect[i], list):
            vect_round(vect[i], param)
        else:
            vect[i] = round(vect[i], 3)
    return vect

def switch(List, i, j):
    List[i], List[j] = List[j], List[i]
    return
