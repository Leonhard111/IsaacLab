held_assets是轴
fixed_assets是孔
两个都在Peginertion中被定义了
env_origins是每个环境在世界坐标系中的基准偏移，是interactiveScene环境克隆过程的返回值
self.cfg_task.fixed_asset_cfg.height：螺栓头部高度
self.cfg_task.fixed_asset_cfg.base_height：螺栓杆身高度

[ x ] 夹爪的控制器可能没写好，怎么控都没用



TODO:
修改forgeenv的

[ x ] _get_observations

_get_rewards：
目前看来可能不需要大改，先不管了，后面再说


IK或是其他设置初始的位姿的步骤有点问题


train:

CUDA_VISIBLE_DEVICES=3 