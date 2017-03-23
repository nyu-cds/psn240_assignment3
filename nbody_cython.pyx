"""
    N-body simulation.
    Changed all the varaibles to cdef
    Changed the function arguments to C type
    Used numpy array with efficient indexing
    Time reduced from 40sec to 20 sec
    Realtive Speed R = 40/20 = 2
"""

from itertools import combinations
import numpy as np

'''
Created numpy arrays with efficient indexing
'''
cdef advance(double dt, System* BODIES):
    '''
        advance the system one timestep
    '''
    #cdef double m, m1, m2, dx, dy, dz, mag, b_m1, b_m2
    for (body1, body2) in list(combinations(range(5),2)):
        [x1, y1, z1] = BODIES[body1].position
        v1 = BODIES[body1].velocity
        m1 = BODIES[body1].mass
        [x2, y2, z2] = BODIES[body2].position
        v2 = BODIES[body2].velocity
        m2 = BODIES[body2].mass
        #(dx, dy, dz) = compute_deltas(x1, x2, y1, y2, z1, z2)
        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
        mag = dt * (pow((dx * dx + dy * dy + dz * dz), -1.5))
        b_m1 = m1 * mag
        b_m2 = m2 * mag
        v1[0] -= dx * b_m2 
        v1[1] -= dy * b_m2
        v1[2] -= dz * b_m2
        v2[0] += dx * b_m1
        v2[1] += dy * b_m1
        v2[2] += dz * b_m1


    for body in range(sizeof(BODIES)):
        r = BODIES[body].position
        [vx,vy,vz] = BODIES[body].velocity
        m = BODIES[body].mass
        r[0] += dt * vx
        r[1] += dt * vy
        r[2] += dt * vz
        #update_rs(r, dt, vx, vy, vz)


    
cdef report_energy(System* BODIES, double e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''        
    #cdef double m, m1, m2
    for (body1, body2) in list(combinations(range(5),2)):
        [x1, y1, z1] = BODIES[body1].position
        v1 = BODIES[body1].velocity
        m1 = BODIES[body1].mass
        [x2, y2, z2] = BODIES[body2].position
        v2 = BODIES[body2].velocity
        m2 = BODIES[body2].mass
        #(dx, dy, dz) = compute_deltas(x1, x2, y1, y2, z1, z2)
        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
        #e -= compute_energy(m1, m2, dx, dy, dz)
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)

    for body in range(sizeof(BODIES)):
        [vx,vy,vz] = BODIES[body].velocity
        m = BODIES[body].mass
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e


cdef offset_momentum(System ref, System* BODIES, double px=0.0,double py=0.0,double pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    #cdef double m
    for body in range(sizeof(BODIES)):
        [vx,vy,vz] = BODIES[body].velocity
        m = BODIES[body].mass
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    v = ref.velocity
    m = ref.mass
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m
    
cdef double PI = 3.14159265358979323
cdef double SOLAR_MASS = 4 * PI * PI
cdef float DAYS_PER_YEAR = 365.24
    
'''
Created a new structure for Bodies object
'''
cdef struct System:
        #np.ndarray[np.float64_t, ndim=1] position, velocity
        float position[3]
        float velocity[3]
        float mass
        
'''
Changed all the variables to cdef, Created a new Bodies object with struct.
'''
def nbody(int loops, str reference, int iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    print("In the new changes 5 nbody from nbody_cython method")
       
    cdef double PI = 3.14159265358979323
    cdef double SOLAR_MASS = 4 * PI * PI
    cdef float DAYS_PER_YEAR = 365.24
    
    cdef System BODIES2[5] 
    
    BODIES2[0].position = np.array([0.0, 0.0, 0.0], dtype=np.float) #[0.0, 0.0, 0.0]#
    BODIES2[0].velocity = np.array([0.0, 0.0, 0.0], dtype=np.float)#[0.0, 0.0, 0.0]
    BODIES2[0].mass = SOLAR_MASS
    
    BODIES2[1].position = np.array([4.84143144246472090e+00,
                     -1.16032004402742839e+00,
                     -1.03622044471123109e-01], dtype=np.float)
    BODIES2[1].velocity = np.array([1.66007664274403694e-03 * DAYS_PER_YEAR,
                     7.69901118419740425e-03 * DAYS_PER_YEAR,
                     -6.90460016972063023e-05 * DAYS_PER_YEAR], dtype=np.float)
    BODIES2[1].mass = 9.54791938424326609e-04 * SOLAR_MASS
    
    BODIES2[2].position = np.array([8.34336671824457987e+00,
                    4.12479856412430479e+00,
                    -4.03523417114321381e-01], dtype=np.float)
    BODIES2[2].velocity = np.array([-2.76742510726862411e-03 * DAYS_PER_YEAR,
                    4.99852801234917238e-03 * DAYS_PER_YEAR,
                    2.30417297573763929e-05 * DAYS_PER_YEAR], dtype=np.float)
    BODIES2[2].mass = 2.85885980666130812e-04 * SOLAR_MASS
    
    BODIES2[3].position = np.array([1.28943695621391310e+01,
                    -1.51111514016986312e+01,
                    -2.23307578892655734e-01], dtype=np.float)
    BODIES2[3].velocity = np.array([2.96460137564761618e-03 * DAYS_PER_YEAR,
                    2.37847173959480950e-03 * DAYS_PER_YEAR,
                    -2.96589568540237556e-05 * DAYS_PER_YEAR], dtype=np.float)
    BODIES2[3].mass = 4.36624404335156298e-05 * SOLAR_MASS
    
    BODIES2[4].position = np.array([1.53796971148509165e+01,
                     -2.59193146099879641e+01,
                     1.79258772950371181e-01], dtype=np.float)
    BODIES2[4].velocity = np.array([2.68067772490389322e-03 * DAYS_PER_YEAR,
                     1.62824170038242295e-03 * DAYS_PER_YEAR,
                     -9.51592254519715870e-05 * DAYS_PER_YEAR], dtype=np.float)
    BODIES2[4].mass = 5.15138902046611451e-05 * SOLAR_MASS
    
    
    cdef System ref
    if reference == 'sun':
         ref = BODIES2[0]
    elif reference == 'jupiter':
         ref = BODIES2[1]   
    elif reference == 'saturn':
         ref = BODIES2[2]
    elif reference == 'uranus':
         ref = BODIES2[3]
    else: 
         ref = BODIES2[4]
    
    # Set up global state
    offset_momentum(ref, BODIES2)

    for _ in range(loops):
        for _ in range(iterations):
            advance(0.01, BODIES2)
        print(report_energy(BODIES2))