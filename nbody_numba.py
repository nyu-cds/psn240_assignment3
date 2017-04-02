"""
    Added vec_deltas, vectorized method
    Added jit to all the methods with signature
    Realtive Speed R = 32/8 = 4
"""

from numba import jit, int32, char, void, vectorize, float64
import numpy as np

@vectorize([float64(float64, float64)])
def vec_deltas(x, y):
    return x - y

@jit('void(float64, float64[:,:,:])', nopython = True)    
def advance(dt, BODIES):
    '''
        advance the system one timestep
    '''

    for (body1, body2) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        
        [x1, y1, z1] = BODIES[body1][0]
        #p1 = BODIES[body1][0]
        v1 = BODIES[body1][1]
        m1 = BODIES[body1][2][0]
        [x2, y2, z2] = BODIES[body2][0]
        #p2 = BODIES[body2][0]
        v2 = BODIES[body2][1]
        m2 = BODIES[body2][2][0]
        #(dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
        #print(vec_deltas(np.array(p1),np.array(p2)))
        [dx, dy, dz] =  vec_deltas(np.array([x1, y1, z1]),np.array([x2, y2, z2]))
        mag = dt * (pow((dx * dx + dy * dy + dz * dz), -1.5))
        b_m1 = m1 * mag
        b_m2 = m2 * mag
        v1[0] -= dx * b_m2 #compute_b(m2, dt, dx, dy, dz)
        v1[1] -= dy * b_m2#compute_b(m2, dt, dx, dy, dz)
        v1[2] -= dz * b_m2#compute_b(m2, dt, dx, dy, dz)
        v2[0] += dx * b_m1#compute_b(m1, dt, dx, dy, dz)
        v2[1] += dy * b_m1#compute_b(m1, dt, dx, dy, dz)
        v2[2] += dz * b_m1#compute_b(m1, dt, dx, dy, dz)
        #update_vs(v1, v2, dt, dx, dy, dz, m1, m2, mag)

    for body in range(len(BODIES)):
        r = BODIES[body][0]
        [vx, vy, vz] = BODIES[body][1]
        m = BODIES[body][2][0]
        r[0] += dt * vx
        r[1] += dt * vy
        r[2] += dt * vz

@jit('float64(float64[:,:,:])', nopython = True)    
def report_energy(BODIES):
    '''
        compute the energy and return it so that it can be printed
    '''        
    e=0.0
    for (body1, body2) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        [x1, y1, z1] = BODIES[body1][0]
        v1 = BODIES[body1][1]
        m1 = BODIES[body1][2][0]
        [x2, y2, z2] = BODIES[body2][0]
        v2 = BODIES[body2][1]
        m2 = BODIES[body2][2][0]
        [dx, dy, dz] =  vec_deltas(np.array([x1, y1, z1]),np.array([x2, y2, z2]))
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)

    for body in range(len(BODIES)):
        r = BODIES[body][0]
        [vx, vy, vz] = BODIES[body][1]
        m = BODIES[body][2][0]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

@jit('void(float64[:,:], float64[:,:,:])', nopython = True)
def offset_momentum(ref, BODIES):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    px=0.0
    py=0.0
    pz=0.0
    for body in range(len(BODIES)):
        r = BODIES[body][0]
        [vx, vy, vz] = BODIES[body][1]
        m = BODIES[body][2][0]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    r = ref[0]
    v = ref[1]
    m = ref[2][0]
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


@jit('void(int32, char[:], int32)')
def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''

    PI = 3.14159265358979323
    SOLAR_MASS = 4 * PI * PI
    DAYS_PER_YEAR = 365.24

    BODIES = np.array([
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [SOLAR_MASS,0.0,0.0]),
    ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
                [9.54791938424326609e-04 * SOLAR_MASS,0.0,0.0]),
   ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR],
               [2.85885980666130812e-04 * SOLAR_MASS,0.0,0.0]),
     ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR],
               [4.36624404335156298e-05 * SOLAR_MASS,0.0,0.0]),
    ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
                [5.15138902046611451e-05 * SOLAR_MASS,0.0,0.0])])

    
    if reference == 'sun':
         ref = BODIES[0]
    elif reference == 'jupiter':
         ref = BODIES[1]   
    elif reference == 'saturn':
         ref = BODIES[2]
    elif reference == 'uranus':
         ref = BODIES[3]
    else: 
         ref = BODIES[4]
   
    	
    # Set up global state
    offset_momentum(ref, BODIES)

    for _ in range(loops):
        for _ in range(iterations):
            advance(0.01, BODIES)
        print(report_energy(BODIES))

if __name__ == '__main__':
    import timeit
    #print('Time taken :' + str(timeit.timeit(nbody(100, 'sun', 20000))))
