# CS106
requirement  
<!-- start install -->

### Install SUMO latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

### Install SUMO-RL

Stable release version is available through pip
```bash
pip install sumo-rl
```


### Run Qlearning
with CLI example
'''bash
!python Qlearning.py --net "/content/sumo-rl/nets/2x2grid/2x2.net.xml" --route "/content/sumo-rl/nets/2x2grid/2x2.rou.xml" --output "./" --train_time 1000 --play_time 1000
'''

