import gym
import retro
import dreamerv2.api as dv2

config = dv2.defaults.update({
    'logdir': '~/logdir/streetfighter',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
#env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
dv2.train(env, config)
