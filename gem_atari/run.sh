python train.py --tag='clipped double dqn (lb) alien' --env=Alien --alg=cddqn --learn=atari_lb --save_rews=True
python train.py --tag='double dqn (lb) alien' --env=Alien --alg=dqn --double=True --learn=atari_lb --save_rews=True
python train.py --tag='clipped double dqn alien' --env=Alien --alg=cddqn --learn=atari --save_rews=True
python train.py --tag='double dqn alien' --env=Alien --alg=dqn --double=True --learn=atari --save_rews=True
python train.py --tag='maxmin dqn alien' --env=Alien --alg=mmdqn --learn=atari --save_rews=True
python train.py --tag='dqn alien' --env=Alien --alg=dqn --learn=atari --save_rews=True
