# UCB-RL

This repository contains my own solution to the homework of UCB's reinforcement learning course.  

Now I have finished hw2 and hw3 beacause they are related to my current project. The remaining part may be achieved in the future.

## Common Issues

I faced many issues when running the code so I listed here just to save your time.

### 1. gcc compilation error when compiling mujoco-py
To run the codes in this repository, your gcc version must be no less than 7. Beforehand I have used gcc 5.4.0 and then I met the error, later on I update to gcc 7.3.0 and there's no problem again.  
  
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
This happens due to numpy version is too old, upgrade numpy to 1.16.0 and the issue will be eliminated.
```
pip install numpy==1.16.0
```
