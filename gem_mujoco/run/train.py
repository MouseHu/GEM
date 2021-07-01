import os
import time

import tensorflow as tf

from run.run_util import parse_args, create_action_noise, create_env, save_args
from stable_baselines import logger
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.ddpg import DDPG
from stable_baselines.sac import SAC
from stable_baselines.td3 import TD3
from stable_baselines.td3.td3_doubletwin import TD3DoubleTwin
from stable_baselines.td3.td3_mem_gem import TD3MemGEM
from stable_baselines.td3.td3_n_step import TD3NSTEP
from stable_baselines.td3.td3_redq import TD3REDQ
from stable_baselines.td3.td3_sil import TD3SIL
from stable_baselines.td3.td3_mem_backprop import TD3MemBackProp

def run(env_id, seed, layer_norm, evaluation, agent, delay_step, gamma=0.99, **kwargs):
    # Create envs.
    env = create_env(env_id, delay_step, str(0))
    print(env.observation_space, env.action_space)
    if evaluation:
        eval_env = create_env(env_id, delay_step, "eval_env")
    else:
        eval_env = None

    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()

    policy = 'MlpPolicy'
    td3_variants = {
        "TD3": TD3,
        "TD3SIL": TD3SIL,
        "TD3NSTEP": TD3NSTEP,
        "TD3REDQ": TD3REDQ,
        "TD3DoubleTwin": TD3DoubleTwin,
    }
    if td3_variants.get(agent, None):
        model_func = td3_variants[agent]
        model = model_func(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=128,
                           tau=0.005, policy_delay=2, learning_starts=25000,
                           action_noise=create_action_noise(env, "normal_0.1"), buffer_size=100000, verbose=2,
                           n_cpu_tf_sess=10,
                           policy_kwargs={"layers": [400, 300]})
    elif agent == "DDPG":
        model = DDPG(policy=policy, env=env, eval_env=eval_env, gamma=gamma, nb_eval_steps=5, batch_size=100,
                     nb_train_steps=100, nb_rollout_steps=100, learning_starts=10000,
                     actor_lr=1e-3, critic_lr=1e-3, critic_l2_reg=0,
                     tau=0.005, normalize_observations=False,
                     action_noise=create_action_noise(env, "normal_0.1"), buffer_size=int(1e6),
                     verbose=2, n_cpu_tf_sess=10,
                     policy_kwargs={"layers": [400, 300]})
    elif agent == "SAC":
        model = SAC(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=256,
                    action_noise=create_action_noise(env, "normal_0.1"), buffer_size=int(1e6), verbose=2,
                    n_cpu_tf_sess=10, learning_starts=10000,
                    policy_kwargs={"layers": [256, 256]})
    elif agent == "GEM":
        policy = 'TD3LnMlpPolicy'
        model = TD3MemGEM(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=128,
                          tau=0.005, policy_delay=2, learning_starts=25000,
                          action_noise=create_action_noise(env, "normal_0.1"), buffer_size=100000, verbose=2,
                          n_cpu_tf_sess=10,
                          alpha=0.5, beta=-1, iterative_q=-1,
                          num_q=4, gradient_steps=200, max_step=kwargs['max_steps'], reward_scale=1., nb_eval_steps=10,
                          policy_kwargs={"layers": [400, 300]})
    elif agent == "BP":
        policy = 'TD3LnMlpPolicy'
        model = TD3MemBackProp(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=128,
                          tau=0.005, policy_delay=2, learning_starts=25000,
                          action_noise=create_action_noise(env, "normal_0.1"), buffer_size=100000, verbose=2,
                          n_cpu_tf_sess=10,
                          alpha=0.5, beta=-1, gradient_steps=200, max_step=kwargs['max_steps'], reward_scale=1., nb_eval_steps=10,
                          policy_kwargs={"layers": [400, 300]})
    else:
        raise NotImplementedError

    print("model building finished")
    model.learn(total_timesteps=kwargs['num_timesteps'])

    env.close()
    if eval_env is not None:
        eval_env.close()

    logger.info('total runtime: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_LOGDIR"] = os.path.join(os.getenv("OPENAI_LOGDIR"), args["comment"])
    save_args(args)
    logger.configure()
    # Run actual script.
    run(**args)
