import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os

taus = [0.1, 0.2, 0.3]
gammas = [0.7, 0.5]
temps = [0.25, 0.5, 0.75]
epsilons = [0.1, 0.2]


#2D for tau
# tau = 0.2
gamma = 0.5
T = 0.25
eps = 0.1

score = []
steps = []

for tau in taus:
    if os.path.isfile("tau_" + str(tau) + "_gamma_" + str(gamma) + "_temp_" + str(T) + "_epsilon_" + str(eps) + ".txt"):
        f = open("tau_" + str(tau) + "_gamma_" + str(gamma) + "_temp_" + str(T) + "_epsilon_" + str(eps) + ".txt", "r")
        out = f.read().split()
        numbers_ls = [list(map(int, x.split('_'))) for x in out]
        numbers = np.array(numbers_ls)
        score.append(np.mean(numbers[:,0]))
        steps.append(np.mean(numbers[:,1]))


# plt.plot(taus, score, 'ro', label = 'score')
# plt.plot(taus, steps, 'bo', label = 'steps')
plt.bar(taus, score, width = 0.045)
# plt.legend()
plt.title('Score')
plt.xlabel('$ \\tau $')
# plt.plot(taus, steps)
plt.show()

plt.bar(taus, steps, width = 0.045, color = 'orange')
plt.title('Steps')
plt.xlabel('$ \\tau $')
# plt.plot(taus, steps)
plt.show()

#


#2D for gamma

tau = 0.1
T = 0.25
eps = 0.1

score = []
steps = []

for gamma in gammas:
    f = open("tau_" + str(tau) + "_gamma_" + str(gamma) + "_temp_" + str(T) + "_epsilon_" + str(eps) + ".txt", "r")
    out = f.read().split()
    numbers_ls = [list(map(int, x.split('_'))) for x in out]
    numbers = np.array(numbers_ls)
    score.append(np.mean(numbers[:,0]))
    steps.append(np.mean(numbers[:,1]))



# plt.plot(gammas, steps, 'bo', label = 'steps')
plt.bar(gammas, score, width = 0.045, label = 'score')
# plt.legend()
plt.title('Score')
plt.xlabel('$ \gamma $')
# plt.plot(taus, steps)
plt.show()

plt.bar(gammas, steps, width = 0.045, color = 'orange')
plt.title('Steps')
plt.xlabel('$ \gamma $')
# plt.plot(taus, steps)
plt.show()

#2D for T and epss

score = []
steps =[]

gamma = 0.7
tau = 0.1
eps = 0.1
for T in temps:
    f = open("tau_" + str(tau) + "_gamma_" + str(gamma) + "_temp_" + str(T) + "_epsilon_" + str(eps) + ".txt", "r")
    out = f.read().split()
    numbers_ls = [list(map(int, x.split('_'))) for x in out]
    numbers = np.array(numbers_ls)
    score.append(np.mean(numbers[:,0]))
    steps.append(np.mean(numbers[:,1]))


plt.bar(temps, score, width = 0.045, label = 'score')
# plt.legend()
plt.title('Score')
plt.xlabel('T')
# plt.plot(taus, steps)
plt.show()

plt.bar(temps, steps, width = 0.045, color = 'orange')
plt.title('Steps')
plt.xlabel('T')
# plt.plot(taus, steps)
plt.show()


score = []
steps = []

gamma = 0.7
T = 0.25
tau = 0.1
for eps in epsilons:
    f = open("tau_" + str(tau) + "_gamma_" + str(gamma) + "_temp_" + str(T) + "_epsilon_" + str(eps) + ".txt", "r")
    out = f.read().split()
    numbers_ls = [list(map(int, x.split('_'))) for x in out]
    numbers = np.array(numbers_ls)
    score.append(np.mean(numbers[:,0]))
    steps.append(np.mean(numbers[:,1]))


plt.bar(epsilons, score, width = 0.045, label = 'score')
# plt.legend()
plt.title('Score')
plt.xlabel('$ \epsilon$')
# plt.plot(taus, steps)
plt.show()

plt.bar(epsilons, steps, width = 0.045, color = 'orange')
plt.title('Steps')
plt.xlabel('$ \epsilon$')
# plt.plot(taus, steps)
plt.show()



#Now 3D for T and eps

score = []
steps =[]

tau = 0.1
gamma = 0.7


for eps in epsilons:
    for T in temps:
        f = open("tau_" + str(tau) + "_gamma_" + str(gamma) + "_temp_" + str(T) + "_epsilon_" + str(eps) + ".txt", "r")
        out = f.read().split()
        numbers_ls = [list(map(int, x.split('_'))) for x in out]
        numbers = np.array(numbers_ls)
        score.append(np.mean(numbers[:,0]))
        steps.append(np.mean(numbers[:,1]))
        # print(eps, T, score[-1])

score = np.array(score)
steps = np.array(steps)

# Colormesh plots
# plt.pcolormesh(epsilons, temps, score.reshape(len(epsilons), len(temps)), cmap='seismic')
# cbar=plt.colorbar()
# cbar.set_label('Score', rotation= 0,labelpad=12)
# plt.xlabel('$\epsilon$')
# plt.ylabel('$Temp$')
# # plt.scatter(0.1,0.2,color='black',label='Real Parameters')
# plt.legend()
# plt.show()

# surface plot
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(epsilons, temps, score.reshape(len(epsilons), len(temps)), cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()



#Scatter plot


# Fixing random state for reproducibility


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(epsilons, temps, score, 'o')
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# import random
#
# epss = [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
# tempss = temps * 3
# print(epss)
# print(tempss)
# print(score)
#
# fig = plt.figure()
# ax = Axes3D(fig)
#
# # sequence_containing_x_vals = list(range(0, 100))
# # sequence_containing_y_vals = list(range(0, 100))
# # sequence_containing_z_vals = list(range(0, 100))
#
#
# # print(len(epss), len(tempss), len(score))
# # ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
# ax.scatter(epss, tempss, score)
#
# plt.show()


# 3D bar plot

epss = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
tempss = temps * 2

fig = plt.figure()
ax = fig.gca(projection = '3d')
data = score
Xi = epss
Yi = tempss
Zi = np.zeros(data.size)

dx = .025 * np.ones(data.size)
dy = .025 * np.ones(data.size)
dz = data.flatten()

ax.set_xlabel('$\epsilon$')
ax.set_ylabel('T')
ax.set_title('Score')
ax.bar3d(epss, tempss, Zi, dx, dy, dz, color = 'b', shade=True)

plt.show()

fig = plt.figure()
ax = fig.gca(projection = '3d')
data = steps
Zi = np.zeros(data.size)

dx = .025 * np.ones(data.size)
dy = .025 * np.ones(data.size)
dz = data.flatten()

ax.set_xlabel('$\epsilon$')
ax.set_ylabel('T')
ax.set_title('Steps')
ax.bar3d(epss, tempss, Zi, dx, dy, dz, color = 'orange', shade=True)

plt.show()
