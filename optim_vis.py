from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.optim as  optim
from adabound import AdaBound
# plt.switch_backend('agg')

def create_optimizer(model_params, optim_name, lr=None, final_lr=None,
        momentum=None, beta1=None, beta2=None, gamma=None, weight_decay=5e-4):
    if optim_name == 'sgd':
        return optim.SGD(model_params, lr, momentum=momentum,
                         weight_decay=weight_decay)
    elif optim_name == 'adagrad':
        return optim.Adagrad(model_params, lr, weight_decay=weight_decay)
    elif optim_name == 'adam':
        return optim.Adam(model_params, lr, betas=(beta1, beta2),
                          weight_decay=weight_decay)
    elif optim_name == 'amsgrad':
        return optim.Adam(model_params, lr, betas=(beta1, beta2),
                          weight_decay=weight_decay, amsgrad=True)
    elif optim_name == 'adabound':
        return AdaBound(model_params, lr, betas=(beta1, beta2),
                        final_lr=final_lr, gamma=gamma,
                        weight_decay=weight_decay)
    else:
        assert optim_name == 'amsbound'
        return AdaBound(model_params, lr, betas=(beta1, beta2),
                        final_lr=final_lr, gamma=gamma,
                        weight_decay=weight_decay, amsbound=True)


# pyplot settings
plt.ion()
fig = plt.figure(figsize=(3, 2), dpi=300)
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
params = {'legend.fontsize': 3,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.axis('off')


def __f2(x, y, x_mean, y_mean, x_sig, y_sig):
    normalizing = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = (-1 * (x - x_mean)**2) / (2 * (x_sig**2))
    y_exp = (-1 * (y - y_mean)**2) / (2 * (y_sig**2))
    return normalizing * torch.exp(x_exp + y_exp)

def cost_func(x=None, y=None):
    z = -1 * __f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.35, y_sig=0.35)
    z -= __f2(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= __f2(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= __f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
    return x, y, z

def draw_surface():
    x_val = y_val = np.arange(-1.5, 1.5, 0.005, dtype=np.float32)
    x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
    x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
    y_val_mesh_flat = y_val_mesh.reshape([-1, 1])
    x = torch.tensor(x_val_mesh_flat)
    y = torch.tensor(y_val_mesh_flat)

    x, y, z = cost_func(x=x, y=y)
    z_val_mesh = z.numpy().reshape(x_val_mesh.shape)
    levels = np.arange(-10, 1, 0.05)
    ax.plot_surface(x_val_mesh, y_val_mesh, z_val_mesh, alpha=.4, cmap=cm.coolwarm)
    plt.draw()

    # plt.savefig('test.png')

def draw_route():
    xlm = ax.get_xlim3d()
    ylm = ax.get_ylim3d()
    zlm = ax.get_zlim3d()
    ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
    ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
    ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
    azm = ax.azim
    ele = ax.elev + 40
    ax.view_init(elev=ele, azim=azm)

    optim_names = ['sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound']
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    params = [
        {'lr': 0.1, 'momentum': 0.9},
        {'lr': 0.01},
        {'lr': 0.001, 'beta1': 0.99, 'beta2': 0.999},
        {'lr': 0.001, 'beta1': 0.99, 'beta2': 0.999},
        {'lr': 0.001, 'beta1': 0.99, 'beta2': 0.999, 'final_lr': 0.1, 'gamma': 0.001},
        {'lr': 0.001, 'beta1': 0.99, 'beta2': 0.999, 'final_lr': 0.1, 'gamma': 0.001}
    ]

    last_x = []
    last_y = []
    last_z = []

    x_init = 0.75
    y_init = 1.0
    x = []
    y = []
    optims = []
    plot_cache = [None for _ in range(len(optim_names))]
    for i, optim in enumerate(optim_names):
        x.append(torch.tensor(x_init, requires_grad=True))
        y.append(torch.tensor(y_init, requires_grad=True))
        optim = create_optimizer({x[i], y[i]}, optim, **params[i])
        optims.append(optim)

    steps = 1000
    optim_index = list(range(6))
    for iter in range(steps):
        # for i, optim in enumerate(optims):
        for ii, i in enumerate(optim_index):
            optim = optim_names[i]
            _x, _y, _z = cost_func(x=x[i], y=y[i])
            if plot_cache[i]:
                plot_cache[i].remove()
            # print(_x.detach().tolist())
            # print(_y.detach().tolist())
            # print(_z.detach().tolist())
            plot_cache[i] = ax.scatter(_x.detach().numpy(), _y.detach().numpy(), _z.detach().numpy(), s=1, depthshade=True, label=optim_names[i], color=colors[i])
            _z.backward()
            optims[i].zero_grad()
            optims[i].step()
            if iter == 0:
                last_z.append(_z.detach().tolist())
                last_x.append(x_init)
                last_y.append(y_init)
            ax.plot([last_x[ii], _x.detach().tolist()], [last_y[ii], _y.detach().tolist()], [last_z[ii], _z.detach().tolist()], linewidth=0.5, color=colors[i])
            last_x[ii] = _x.detach().tolist()
            last_y[ii] = _y.detach().tolist()
            last_z[ii] = _z.detach().tolist()

        if iter == 0:
            legend = np.vstack((optim_names, [ p['lr'] for p in params ])).transpose()
            plt.legend(plot_cache, legend)
        plt.savefig('test1/' + str(iter) + '.png')

draw_surface()
draw_route()
