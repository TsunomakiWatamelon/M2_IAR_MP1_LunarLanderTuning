# Hyperparamètres principaux ajustés pour LunarLanderContinuous-v2
BATCH_SIZE = 128  # Augmenter la taille du batch pour plus de stabilité
TAU_TARGET = 0.005  # Mise à jour douce des cibles pour une meilleure stabilité
DISCOUNT_FACTOR = 0.99  # Horizon temporel plus long pour l'environnement LunarLander
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
ACTOR_HIDDEN_SIZE = [400, 300]  # Taille des couches cachées de l'acteur
CRITIC_HIDDEN_SIZE = [400, 300]  # Taille des couches cachées des critiques
ACTION_NOISE = 0.1  # Conserver le bruit d'action pour favoriser l'exploration
MAX_GRAD_NORM = 0.5
LEARNING_STARTS = 100  # Commencer l'apprentissage après 10 000 pas
BUFFER_SIZE = int(1e6)  # Augmenter la taille du replay buffer à 1 million pour LunarLander
N_STEPS = 1  # Nombre de pas avant mise à jour
NB_EVALS = 10 # Nombre d'environements pour le parallel gym agent (pas compris)


def get_td3_params():
    # Initialisation des paramètres avec les hyperparamètres ajustés
    params = {
        "save_best": False,
        "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
        "collect_stats": True,
        "plot_agents": True,
        "algorithm": {
            "seed": 1,
            "max_grad_norm": MAX_GRAD_NORM,
            "n_envs": 1,
            "n_steps": N_STEPS,
            "nb_evals": NB_EVALS, # Nombre d'envs pour le parallel gym agent (pas compris)
            "discount_factor": DISCOUNT_FACTOR,  # Facteur d'actualisation ajusté
            "buffer_size": BUFFER_SIZE,  # Buffer augmenté
            "batch_size": BATCH_SIZE,  # Batch augmenté
            "tau_target": TAU_TARGET,  # Mise à jour douce ajustée
            "eval_interval": 1000, # Log tous les n pas
            "max_epochs": 100_000, # Nombre d'epoch, total time steps = max_epochs * n_steps
            "learning_starts": LEARNING_STARTS,
            "action_noise": ACTION_NOISE,
            "architecture": {
                "actor_hidden_size": ACTOR_HIDDEN_SIZE,
                "critic_hidden_size": CRITIC_HIDDEN_SIZE,
            },
        },
        "gym_env": {
            "env_name": "LunarLanderContinuous-v2",  #  LunarLanderContinuous-v2 - 
        },
        "actor_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": LEARNING_RATE_ACTOR,  # Taux d'apprentissage pour l'acteur
        },
        "critic_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": LEARNING_RATE_CRITIC,  # Taux d'apprentissage pour les critiques
        },
    }

    return params