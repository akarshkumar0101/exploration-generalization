# exploration-generalization

Does exploration generalizes better than RL?


In gymnasium and envpool,
the environments LostLuggage and Skiing only have 9 actions even when full_action_space=True.
Very weird.
In envpool,
the environments Frogger and KingKong do not move until the fire button is pressed (when fire_reset=False).
Very weird.
For these reason, these 4 environments are in atari_games_ignore.txt.




