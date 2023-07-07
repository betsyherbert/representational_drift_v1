from modules import *

def get_path(folder_name, file_name):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    folder_path = os.path.join(grandparent_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    return file_path

def circular_gaussian(N, theta, amp=2, sigma=20, baseline=0):
    theta_y = np.linspace(0, 180, N)  # center of tuning curves 
    d = np.abs(theta - theta_y)       # distance to input theta
    d_plus = d + 180
    d_minus = d - 180
    y = amp * ( np.exp(-(d**2)/(2*sigma**2)) + np.exp(-(d_plus**2)/(2*sigma**2)) + np.exp(-(d_minus**2)/(2*sigma**2))) + baseline
    return y
    
def ori_matrix(N, var, diagonal=True):
    x = np.linspace(0, 180, N)
    matrix = np.zeros((N, N))
    for i in range(N):
        mean = x[i]
        matrix[:, i] = stats.norm.pdf(x, mean, var) + stats.norm.pdf(x, mean+180, var) + stats.norm.pdf(x, mean-180, var) 
        if diagonal==False: matrix[i, i] = 0
    return matrix

def ori_matrix_different_vars(N, vars, diagonal=True):
    x = np.linspace(0, 180, N)
    matrix = np.zeros((N, N))
    for i in range(N):
        mean = x[i]
        matrix[:, i] = stats.norm.pdf(x, mean, vars[i]) + stats.norm.pdf(x, mean+180, vars[i]) + stats.norm.pdf(x, mean-180, vars[i]) 
        if diagonal==False: matrix[i, i] = 0
    return matrix

def circular_distance(x, y):
    return np.minimum(np.abs(x-y), 180-np.abs(x-y))

def directional_circular_distance(x, y):
    difference = x-y
    difference[difference > 90] -= 180
    difference[difference < -90] += 180
    return difference

def convert_to_array(*args):
    for arg in args:
        arg = np.array(arg)
    return args

def hebbian(pre, post):
    return pre * post

def propensity_hebb(w, a):
    return np.tanh(a*w)
    # return np.tanh(a*w)
    # b = - np.sqrt(1/a)
    # val = - a * (w + b)**2 + 1
    # val[val<0] = 0
    # return val

def propensity_rand(w, b):
    return b*w
    # return np.tanh(b*w)
    # # y = b*w
    # # y[y>1] = 1
    # # return y

def get_preferred_orientation(N, W, n_angles):
    v = np.zeros(n_angles)
    for i, angle in enumerate(np.linspace(0, 180, n_angles)):
        u = circular_gaussian(N, angle, amp=1, sigma=5, baseline=0)
        v[i] = W.T.dot(u)
    return 180 * np.argmax(v) / (n_angles - 1 )

def get_preferred_orientations(N, W, n_angles):
    posts = np.zeros((N, n_angles))
    for i, angle in enumerate(np.linspace(0, 181, n_angles)):
        y = circular_gaussian(N, angle, amp=1, sigma=5, baseline=0)
        posts[:, i] = W.T.dot(y)
    return 180 * np.argmax(posts, axis=1) / (n_angles)

def gauss(x, p): # p[0]=mean, p[1]=stdev
    return 1.0 / (p[1] * np.sqrt(2*np.pi)) * np.exp(-(x-p[0])**2/(2*p[1]**2))

def get_tuning_curve(N, W, n_angles):
    posts = np.zeros((N, n_angles))
    for i, angle in enumerate(np.linspace(0, 181, n_angles)):
        y = circular_gaussian(N, angle, amp=1, sigma=5, baseline=0)
        posts[:, i] = W.T.dot(y)
    return posts

def FWHM(N, Y, n_angles):

    X = np.linspace(0, 180, n_angles)
    Y /= ((180)/N)*Y.sum()                      # Renormalize to a proper PDF
    mean = sum(X*Y)/N                  
    sigma = sum(Y*(X-mean)**2)/N    
    p0 = [mean, sigma]                           # Inital guess 
    errfunc = lambda p, x, y: gauss(x, p) - y    # Distance to the target function
    p1, success = opt.leastsq(errfunc, p0[:], args=(X, Y))
    fit_mu, fit_stdev = p1
    FWHM = 2*np.sqrt(2*np.log(2))*fit_stdev

    return FWHM


def initialise_W(N, vars):
    W = ori_matrix_different_vars(N, vars, diagonal=True) / N   # initialise network with tuning curves distributed across stimulis space 
    W /= np.sum(W, axis=0)
    return W

# def hebbian_component(N, W, n_thetas, theta_stim, type):    
#     if type == 'baseline': thetas = np.random.choice(180, size=n_thetas, replace=True)    # uniform sampling from stimulus space
#     if type == 'stripe_rearing': theta = theta_stim                                                              # permitted orientation
#     delta_ws = []
#     for i in range(n_thetas):
#         if type == 'baseline': theta = thetas[i]
#         u = circular_gaussian(N, theta, amp=1, sigma=60, baseline=0)       # pre-synaptic activity 
#         v = W.T.dot(u)                                                     # post-synaptic activity
#         if len(W.shape) > 1: delta_ws.append(np.outer(u, v))
#         else: delta_ws.append(u*v)
#     return np.sum(np.array(delta_ws), axis=0)


def hebbian_component(N, W, n_thetas, theta_stim, type):    
    if type == 'baseline' or type == 'test': thetas = np.random.choice(180, size=n_thetas, replace=False)    # uniform sampling from stimulus space
    # if type == 'test': thetas = np.random.choice(np.arange(0, 180), n_thetas, replace=False)  # thetas = np.linspace(0, 180, n_thetas, endpoint=False)
    if type == 'stripe_rearing': theta = theta_stim                                                              # permitted orientation
    delta_ws = []
    for i in range(n_thetas):
        if type == 'baseline' or type == 'test': theta = thetas[i]
        u = circular_gaussian(N, theta, amp=0.11, sigma=60, baseline=0)       # pre-synaptic activity 
        v = W.T.dot(u)                                                       # post-synaptic activity
        if len(W.shape) > 1: delta_ws.append(np.outer(u, v))
        else: delta_ws.append(u*v)
    return np.sum(np.array(delta_ws), axis=0)


def normalisation(W):
        W[W < 0] = 0    
        W /= np.sum(W, axis=0) + 1e-10      # divisive normalisation and rectification