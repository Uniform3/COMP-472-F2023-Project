To start our version of AI Wargame from the terminal, use the following commands after making sure the terminal is in the correct directory of the file:

"python ai_wargame_D2.py" to launch game without changing the game options

To change any of the following options, append this to the above default command:

"-h" or "--help" to see the options in the terminal

"--max_depth MAX_DEPTH" where MAX_DEPTH is an int value to change the maximum search depth from the default of 4

"--max_time MAX_TIME" where MAX_TIME is a float value to change the maximum search time from the default of 5.0

"--game_type GAME_TYPE" where GAME_TYPE is one of the following strings ('attacker', 'defender', 'auto', or 'manual') from the default value of 'manual'. 
'attacker' will set the user as the attacker player against an AI as the defender
'defender' will set the user as the defender player against an AI as the attacker
'auto' will set both players as AI to play against each other
'manual' will set the user as both attacker and defender players.

"--broker BROKER" where BROKER is a string to set the game to play via a broker from the default of None

"--alpha_beta ALPHA_BETA" where ALPHA_BETA is a boolean value where True indicates the use of the Alpha-Beta algorithm and False indicates the use of the Minimax algorithm. The default value is False.

"--attack_h ATTACK_H" where ATTACK_H is an integer value of 0, 1, or 2 which indicates which heuristic the attacker player will use if it is controlled by an AI. The default value is 0.

"--defend_h DEFEND_H" where DEFEND_H is an integer value of 0, 1, or 2 which indicates which heuristic the defender player will use if it is controlled by an AI. The default value is 0.