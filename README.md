## Flappy Bird RL environment with Segmentation and Detection
<figure class="half">
    <img src="./images/demo.gif" width="70%">
</figure>

### Installation
```bash
pip install git+git://github.com/cxy1997/Flappy-Bird-with-Segmentation.git
```

### Usage
```python
from Flappy_Bird_with_Segmentation import FlappyBirdSegEnv

env = FlappyBirdSegEnv()
obs, info = env.reset()
obs, reward, terminal, info = env.step(0)  # 1-flap the bird; 0-do nothing
```

Have fun!
