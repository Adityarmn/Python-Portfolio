import numpy as np
import matplotlib.pyplot as plt
from Simulasi import simulasi
from Objective import objective_function
from MFD import MFD
from csv import writer

time_start = 8
time_end = 10
runs = 1


max_iteration = 3
error_max = 0.01
# Nilai c1,c2, dan W divariasikan hingga menghasilkan hasil yang paling optimal
c1 = 0.1  # cognitif parameter
c2 = 0.1  # social coefficient
W = 0.8  # weight
jumlah_partikel = 10

# random x1 x2 sesuai jumlah partikel
x1 = 5 * np.random.rand(1, jumlah_partikel)[0]
x2 = 5 * np.random.rand(1, jumlah_partikel)[0]

# random vx1 vx2 sesuai jumlah partikel
vx1 = np.random.rand(1, jumlah_partikel)[0]
vx2 = np.random.rand(1, jumlah_partikel)[0]

# Cari Objective Function
objective = [0 for i in range(jumlah_partikel)]
index_terbaik = np.argmax(objective)

pbest = np.array([x1, x2])
gbest = np.array(
    [[pbest[0, index_terbaik]], [pbest[1, index_terbaik]]]
)  # ambil nilai komponen kolom ke index terbaik dan seluruh baris
# print(index_terbaik)
# print(pbest)
# print(gbest)

v = np.array([vx1, vx2])
x = np.array([x1, x2])
r1 = np.random.rand()
r2 = np.random.rand()

objective_next = [0 for i in range(jumlah_partikel)]

iter = 0
while True:
    v_next = W * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    x_next = x + v_next
    x1_next = x_next[0, :]
    x2_next = x_next[1, :]
    for i in range(jumlah_partikel):
        # objective_next[i] = validasi(beta=0.5, density=0.1, time_start=x1_next[i], time_end=x2_next[i])
        simulasi(
            beta=x1_next[i],
            density=x2_next[i],
            time_start=time_start,
            time_end=time_end,
        )  # nanti x1 x2 diganti beta sama density
        Qpeak, Kpeak, Qgridlock, Kgridlock = MFD(iter, i)
        objective_next[i] = objective_function(Qpeak, Kpeak, Qgridlock, Kgridlock)
        data_list = [
            iter,
            i,
            Qpeak,
            Kpeak,
            Qgridlock,
            Kgridlock,
            objective_next[i],
            x1_next[i],
            x2_next[i],
        ]
        with open("data.csv", "a") as f_object:
            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object)

            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(data_list)

            # Close the file object
            f_object.close()

    # objective_next = (x1_next-3.14)**2 + (x2_next-2.72)**2 + np.sin(3*x1_next+1.41) + np.sin(4*x2_next-1.73)

    for j in range(jumlah_partikel):
        if objective_next[j] > objective[j]:
            pbest[0, j] = x1_next[j]
            pbest[1, j] = x2_next[j]
        else:
            None

    objective = objective_next
    index_terbaik = np.argmax(objective_next)
    gbest = np.array(
        [[pbest[0, index_terbaik]], [pbest[1, index_terbaik]]]
    )  # ambil nilai komponen kolom ke index terbaik dan seluruh baris

    v = v_next
    x = np.array([x1_next, x2_next])

    error = pbest - gbest
    status = np.linalg.norm(error, axis=0) < error_max

    if status.all() or iter > max_iteration:
        break

    iter = iter + 1


# def f(x,y):
#     "Objective function"
#     return J

# # Compute and plot the function in 3D within [0,5]x[0,5]
# x_mesh, y_mesh = np.array(np.meshgrid(np.linspace(0,5,100), np.linspace(0,5,100)))
# z_mesh = f(x_mesh, y_mesh)

# # Find the global minimum
# x_min = x_mesh.ravel()[z_mesh.argmin()]
# y_min = y_mesh.ravel()[z_mesh.argmin()]


# # Set up base figure: The contour map
# fig, ax = plt.subplots(figsize=(8,6))
# fig.set_tight_layout(True)
# img = ax.imshow(z_mesh, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.4)
# fig.colorbar(img, ax=ax)
# ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
# contours = ax.contour(x_mesh, y_mesh, z_mesh, 10, colors='black', alpha=0.3)
# ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
# ax.set_xlim([0,5])
# ax.set_ylim([0,5])
# pbest_plot = ax.scatter(pbest[0,:], pbest[1,:], marker='o', color='black', alpha=0.4)
# p_plot = ax.scatter(x[0,:], x[1,:], marker='o', color='blue', alpha=0.5)
# p_arrow = ax.quiver(x[0,:], x[1,:], v[0,:], v[1,:], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
# gbest_plot = plt.scatter([gbest[0,:]], [gbest[1,:]], marker='*', s=100, color='black', alpha=0.4)

# print(objective_next)
# print(gbest)
# print(iter)
