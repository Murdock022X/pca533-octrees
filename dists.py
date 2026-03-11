#!/usr/bin/env python
# coding: utf-8

# ### Generate Different Initial Distributions of Interest
# 
# - Uniform
# - Gaussian
# - Stretched along 1 or 2 dimensions

# In[1]:

from dist_helpers import *

# In[3]:


x,y,z = uniform_initial(10000)
plot_3d(x,y,z)


# In[4]:


x,y,z = normal_initial(10000)
plot_3d(x,y,z)


# In[5]:


x,y,z = rectangular_initial(10000)
plot_3d(x,y,z)

# In[8]:


ix,iy,iz = uniform_initial(1000)
px,py,pz = perturb(len(ix),0.1)
plot_3d(px,py,pz)


# ### Save Distributions to HDF5

# In[9]:


f = h5.File('test.h5', 'w')

names = ["test"]
dist = [uniform_initial(int(1e3))]
pt = [perturb(len(dist[0][0]), 0.1)]

for name, (ix,iy,iz), (px,py,pz) in zip(names, dist, pt):
    g = f.create_group(name)
    g.create_dataset("ix", data=ix)
    g.create_dataset("iy", data=iy)
    g.create_dataset("iz", data=iz)
    g.create_dataset("px", data=px)
    g.create_dataset("py", data=py)
    g.create_dataset("pz", data=pz)

f.close()


# In[10]:


f = h5.File('particles.h5', 'w')

dist = [
        # ("leaf_order", (np.random.uniform(-1, 1, int(1e5)), np.random.uniform(-1, 1, int(1e5)), np.random.uniform(-1,1,int(1e5))*0.01), perturb(int(1e5), 0.001)),
        # ("square_em3_n1k", uniform_initial(int(1e3)), perturb(int(1e3), 0.001)), 
        # ("square_em3_n10k", uniform_initial(int(1e4)), perturb(int(1e4), 0.001)), 
        # ("square_em3_n100k", uniform_initial(int(1e5)), perturb(int(1e5), 0.001)), 
        # ("square_em3_n1m", uniform_initial(int(1e6)), perturb(int(1e6), 0.001)), 
        # ("square_em3_n10m", uniform_initial(int(1e7)), perturb(int(1e7), 0.001)),
        # ("square_em2_n1k", uniform_initial(int(1e3)), perturb(int(1e3), 0.01)), 
        # ("square_em2_n10k", uniform_initial(int(1e4)), perturb(int(1e4), 0.01)), 
        # ("square_em2_n100k", uniform_initial(int(1e5)), perturb(int(1e5), 0.01)), 
        # ("square_em2_n1m", uniform_initial(int(1e6)), perturb(int(1e6), 0.01)), 
        # ("square_em2_n10m", uniform_initial(int(1e7)), perturb(int(1e7), 0.01)),
        # ("square_em1_n1k", uniform_initial(int(1e3)), perturb(int(1e3), 0.1)), 
    ("square_em1_n10k", uniform_initial(int(1e4)), perturb(int(1e4), 0.1)), 
    ("square_em1_n100k", uniform_initial(int(1e5)), perturb(int(1e5), 0.1)), 
    ("square_em1_n1m", uniform_initial(int(1e6)), perturb(int(1e6), 0.1)), 
    ("square_em1_n10m", uniform_initial(int(1e7)), perturb(int(1e7), 0.1)),
    ("square_em1_n100m", uniform_initial(int(1e8)), perturb(int(1e8), 0.1)),
    # ("cylinder_em1_n1k", rectangular_initial(int(1e3)), perturb(int(1e3), 0.1)),
    # ("cylinder_em1_n10k", rectangular_initial(int(1e4)), perturb(int(1e4), 0.1)),
    # ("cylinder_em1_n100k", rectangular_initial(int(1e5)), perturb(int(1e5), 0.1)),
    # ("cylinder_em1_n1m", rectangular_initial(int(1e6)), perturb(int(1e6), 0.1)),
    # ("cylinder_em1_n10m", rectangular_initial(int(1e7)), perturb(int(1e7), 0.1)),
    # ("cylinder_em1_n100m", rectangular_initial(int(1e8)), perturb(int(1e8), 0.1)),
    # ("sphere_em1_n1k", normal_initial(int(1e3)), perturb(int(1e3), 0.1)),
    # ("sphere_em1_n10k", normal_initial(int(1e4)), perturb(int(1e4), 0.1)),
    # ("sphere_em1_n100k", normal_initial(int(1e5)), perturb(int(1e5), 0.1)),
    # ("sphere_em1_n1m", normal_initial(int(1e6)), perturb(int(1e6), 0.1)),
    # ("sphere_em1_n10m", normal_initial(int(1e7)), perturb(int(1e7), 0.1)),
    # ("sphere_em1_n100m", normal_initial(int(1e8)), perturb(int(1e8), 0.1))
]

for name, (ix,iy,iz), (px,py,pz) in dist:
    g = f.create_group(name)
    g.create_dataset("ix", data=ix)
    g.create_dataset("iy", data=iy)
    g.create_dataset("iz", data=iz)
    g.create_dataset("px", data=px)
    g.create_dataset("py", data=py)
    g.create_dataset("pz", data=pz)

f.close()

