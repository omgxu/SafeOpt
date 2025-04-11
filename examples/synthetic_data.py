from __future__ import print_function, division, absolute_import

import GPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import safeopt
import stageopt

mpl.rcParams['figure.figsize'] = (20.0, 10.0)
mpl.rcParams['font.size'] = 20
mpl.rcParams['lines.markersize'] = 20

# Set some constant values
num_samples = 20
num_iterations = 20
switch_time = 10

# Measurement noise
noise_var0 = 0.05 ** 2
noise_var1 = 1e-5
# noise_var2 = 1e-5
# noise_var3 = 1e-5

# Bounds on the inputs variable
bounds = [(0., 1.), (0., 1.)]

# Define Kernel
kernel0 = GPy.kern.Matern32(input_dim=len(bounds), variance=2., lengthscale=1.0, ARD=True)
kernel1 = kernel0.copy()
# kernel2 = kernel0.copy()
# kernel3 = kernel0.copy()

# Initial safe point
x0 = np.zeros((1, len(bounds)))

X = np.linspace(bounds[0][0], bounds[0][1], 25)
Y = np.linspace(bounds[1][0], bounds[1][1], 25)
X, Y = np.meshgrid(X, Y)

parameter_set = safeopt.linearly_spaced_combinations(bounds, 25)

# Generate function with safe initial point at x=0
def sample_safe_fun():
    fun_f = safeopt.sample_gp_function(kernel0, bounds, noise_var0, 25)
    while True:
        fun_s1 = safeopt.sample_gp_function(kernel1, bounds, noise_var1, 25)
        if fun_s1([0., 0.], noise=False) > 1:
            break
            
    def combined_fun(x, noise=True):
        return np.hstack([fun_f(x, noise), fun_s1(x, noise)])
    return combined_fun

funs = [sample_safe_fun() for _ in range(num_samples)]

# Storage for the results
reward_safe, reward_stage = [0.]*num_iterations, [0.]*num_iterations
regret_safe, regret_stage = [0.]*num_iterations, [0.]*num_iterations
safe_region_safe, safe_region_stage = [0.]*num_iterations, [0.]*num_iterations

for i in range(num_samples):

    # The statistical model of our objective function and safety constraint
    y0 = funs[i](x0)
    gp_safe_f = GPy.models.GPRegression(x0, y0[:, 0, None], kernel0, noise_var=noise_var0)
    gp_stage_f = GPy.models.GPRegression(x0, y0[:, 0, None], kernel0, noise_var=noise_var0)
    gp_safe_s1 = GPy.models.GPRegression(x0, y0[:, 1, None], kernel1, noise_var=noise_var1)
    gp_stage_s1 = GPy.models.GPRegression(x0, y0[:, 1, None], kernel1, noise_var=noise_var1)

    # The optimization routine
    # opt = safeopt.SafeOptSwarm(gp, 0., bounds=bounds, threshold=0.2)
    safe_opt = safeopt.SafeOpt([gp_safe_f, gp_safe_s1], parameter_set, [-np.inf, 0.], lipschitz=None, threshold=0.2)
    stage_opt = stageopt.StageOpt([gp_stage_f, gp_stage_s1], parameter_set, [-np.inf, 0.], lipschitz=None, threshold=0.2, switch_time=switch_time)

    Z = np.array([funs[i]([x, y], noise=False).flatten() for x, y in zip(X.ravel(), Y.ravel())])

    # Reshape Z for utility and safety components
    Z_f = Z[:, 0].reshape(X.shape)  # Utility function
    Z_s1 = Z[:, 1].reshape(X.shape)  # Safety function (1)

    # Calculate the maximum value of utility function in safe region
    safe_region = np.where(Z_s1 > 0)
    safe_utility = Z_f[safe_region]
    safe_max_utility = np.max(safe_utility)

    # First query point
    reward_safe[0] = (reward_safe[0] * i + safe_opt.get_maximum()[1]) / (i + 1)
    reward_stage[0] = (reward_stage[0] * i + stage_opt.get_maximum()[1]) / (i + 1)
    regret_safe[0] = (regret_safe[0] * i + (safe_max_utility - safe_opt.get_maximum()[1])) / (i + 1)
    regret_stage[0] = (regret_stage[0] * i + (safe_max_utility - stage_opt.get_maximum()[1])) / (i + 1)
    safe_region_safe[0] = (safe_region_safe[0] * i + np.sum(safe_opt.S)) / (i + 1)
    safe_region_stage[0] = (safe_region_stage[0] * i + np.sum(stage_opt.S)) / (i + 1)

    for j in range(num_iterations - 1):
        # Obtain next query point
        x_next = safe_opt.optimize()
        # Get a measurement from the real system
        y_meas = funs[i](x_next)
        # Add this to the GP model
        safe_opt.add_new_data_point(x_next, y_meas)

        x_next = stage_opt.optimize()[0]
        y_meas = funs[i](x_next)
        stage_opt.add_new_data_point(x_next, y_meas)
        
        reward_safe[j + 1] = (reward_safe[j + 1] * i + safe_opt.get_maximum()[1]) / (i + 1)
        reward_stage[j + 1] = (reward_stage[j + 1] * i + stage_opt.get_maximum()[1]) / (i + 1)
        regret_safe[j + 1] = (regret_safe[j + 1] * i + (safe_max_utility - safe_opt.get_maximum()[1])) / (i + 1)
        regret_stage[j + 1] = (regret_stage[j + 1] * i + (safe_max_utility - stage_opt.get_maximum()[1])) / (i + 1)
        safe_region_safe[j + 1] = (safe_region_safe[j + 1] * i + np.sum(safe_opt.S)) / (i + 1)
        safe_region_stage[j + 1] = (safe_region_stage[j + 1] * i + np.sum(stage_opt.S)) / (i + 1)

    if (i + 1) % 5 == 0:
        print("Sample: ", i + 1)

# Plot reward
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(reward_safe) + 1), reward_safe, label='SafeOpt', color='blue')
plt.plot(range(1, len(reward_stage) + 1), reward_stage, label='StageOpt', color='orange')
plt.xticks(range(1, len(regret_safe) + 1), rotation=45)  # 设置横坐标刻度和旋转角度
plt.xlabel('Time step')
plt.ylabel('Best Reward')
plt.title('Best Reward vs Time step')
plt.legend()
plt.grid()
plt.savefig('reward.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot regret
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(regret_safe) + 1), regret_safe, label='SafeOpt', color='blue')
plt.plot(range(1, len(regret_stage) + 1), regret_stage, label='StageOpt', color='orange')
plt.xticks(range(1, len(regret_safe) + 1), rotation=45)  # 设置横坐标刻度和旋转角度
plt.xlabel('Time step')
plt.ylabel('Regret')
plt.title('Regret vs Time step')
plt.legend()
plt.grid()
plt.savefig('regret.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot safe region
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(safe_region_safe) + 1), safe_region_safe, label='SafeOpt', color='blue')
plt.plot(range(1, len(safe_region_stage) + 1), safe_region_stage, label='StageOpt', color='orange')
plt.xticks(range(1, len(safe_region_safe) + 1), rotation=45)  # 设置横坐标刻度和旋转角度
plt.xlabel('Time step')
plt.ylabel('Safe Region')
plt.title('Safe Region vs Time step')
plt.legend()
plt.grid()
plt.savefig('safe_region.png', dpi=300, bbox_inches='tight')
plt.show()