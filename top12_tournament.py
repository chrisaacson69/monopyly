"""
Top-12 tournament: Only the strongest AIs compete.
Tests whether top AIs win by bullying weak bots or by genuine strategy.
"""
from monopyly import *

# Top 12 from the full tournament (by wins)
TOP_12 = {
    "Edmund", "Baldrick", "LordMelchitt", "DarkRedLight",
    "SimpleMind", "Percy", "Queenie", "RedLight",
    "Catbert", "Ratbert", "DarkGreenLight", "MarkovTradeAI",
}

# Load all AIs and filter to top 12
all_ais = load_ais()
ais = [ai for ai in all_ais if ai.get_name() in TOP_12]

found = {ai.get_name() for ai in ais}
missing = TOP_12 - found
if missing:
    print(f"WARNING: Missing AIs: {missing}")
print(f"Loaded {len(ais)} AIs: {[ai.get_name() for ai in ais]}")

# Set up logging
Logger.add_handler(ConsoleLogHandler(Logger.INFO_PLUS))
Logger.add_handler(FileLogHandler("top12_tournament.log", Logger.INFO_PLUS))
Logger.log("Top-12 Tournament: {0} AIs".format(len(ais)), Logger.INFO_PLUS)

# With 12 AIs in groups of 4:
# P(12,4) = 11,880 permutations per sub-round
# 100 rounds × 2 sub-rounds × 1 player-count = 200 sub-rounds
# max_games_per_round = 35000 / 200 = 175
# So each sub-round samples 175 from 11,880 = good coverage
tournament = Tournament(
    player_ais=ais,
    min_players_per_game=4,
    max_players_per_game=4,
    number_of_rounds=30,
    maximum_games=10500,
    permutations_or_combinations=Tournament.PERMUTATIONS)

tournament.play()
tournament.log_results()
