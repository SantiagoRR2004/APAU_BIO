import gymnasium as gym

# Retrieve all environment IDs
env_ids = list(gym.envs.registry.keys())

# Print the list of environment IDs
for env_id in env_ids:
    # Get environment specification
    env_spec = gym.spec(env_id)

    # Extract reward threshold (if available)
    threshold = (
        env_spec.reward_threshold
        if env_spec.reward_threshold is not None
        else float("inf")
    )

    print(f"Threshold for {env_id}: {threshold}")
