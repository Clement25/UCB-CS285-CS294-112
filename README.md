# UCB-RL

This repository contains my own solution on the homework of UCB's reinforcement learning course.

## Common Issues

I faced many issues when running the code so I listed here just to save your time.

### 1. gcc compilation error when compiling mujoco-py
To run code of this repository, your gcc version must no less than 7. I use gcc 5.4.0 at the first time I met the error, later on I update to gcc 7.3.0 and there's no problem again.  
  
To upgrade gcc, use the commands below.
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-7 g++-7
``` 
Here you just installed new version gcc but haven't changed the default version to it. To achieve this, modify the soft link of gcc in folder /usr/bin
```
cd /usr/bin
sudo ln -snf gcc-7 gcc
```

### 2. Value error: numpy.ufunc size changed
This happens due to numpy version is too old, upgrade numpy to 1.16.0 and then it will disappear
```
pip install numpy==1.16.0
```