## Flappy Bird RL environment with Segmentation
<figure class="half">
    <img src="./images/demo.gif" width="70%">
</figure>

### Installation
```bash
git clone https://github.com/cxy1997/Flappy-Bird-with-Segmentation.git
cd Flappy-Bird-with-Segmentation/
python -m pip install -e . || python setup.py install
```

### Usage
```python
from Flappy_Bird_with_Segmentation.environment import FlappyBirdSegEnv

env = FlappyBirdSegEnv()
obs, info = env.reset()
obs, reward, terminal, info = env.step(0)  # 1-flap the bird; 0-do nothing
```

Have fun!
