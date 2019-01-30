# accurate-head-pose

We release the code of the [Hybrid Coarse-fine Classification for Head Pose Estimation](https://arxiv.org/abs/1901.06778), built on top of the [deep-head-pose](https://github.com/natanielruiz/deep-head-pose).

### Pretrained model
We provide pretrained model to reproduce the same result shown in the paper.  

[AFLW2000](https://pan.baidu.com/s/1y9q0JmnA-QxaORyn5fhPKQ), password: drmz  

[AFLW](https://pan.baidu.com/s/1rj2xLINrabaqiIzvSKlGEg), password: yym5  

[BIWI](https://pan.baidu.com/s/1bZXMdGiycX4T4u0VVofQXQ), password: 8qpc  


### Testing
Training and testing lists can be found in /tools, you need download corresonding dataset and update the path.
```bash
python test_hopenet.py --gpu 0 --data_dir directory-path-for-dataset --filename_list filename-list --snapshot model --dataset dataset-name 
```


### TODO:
Instructions for scripts  
Better and better models  
Videos and example demo  

### Acknowledgement
Our hybrid classification network is plug-and-play on top of the [deep-head-pose](https://github.com/natanielruiz/deep-head-pose), but it could be extended to other classification tasks easily. We thank Nataniel Ruiz for releasing deep-head-pose-Pytorch codebase. 
