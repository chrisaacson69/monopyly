"""
Diagnostic test: Run MarkovTradeAI in focused 1v3 games and analyze
monopoly acquisition patterns, deal flow, and loss modes.
"""
import sys
import os
import random
import logging

sys.path.insert(0, os.path.dirname(__file__))
from monopyly import *

# Set up diagnostic logging
logger = logging.getLogger('markov_diag')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

# Load all AIs (returns instances)
all_ai_instances = load_ais()
markov_class = None
other_classes = []
for ai in all_ai_instances:
    cls = type(ai)
    if ai.get_name() == "MarkovTradeAI":
        markov_class = cls
    else:
        other_classes.append(cls)

if markov_class is None:
    print("ERROR: MarkovTradeAI not found!")
    sys.exit(1)

print(f"Loaded {len(other_classes)} opponents")

N_GAMES = 50
wins = 0
losses = 0

# Aggregate diagnostics
agg = {
    'wins_with_monopoly': 0,
    'wins_without_monopoly': 0,
    'losses_with_monopoly': 0,
    'losses_without_monopoly': 0,
    'loss_never_monopoly': 0,
    'loss_had_monopoly': 0,
    'total_deals_we_proposed': 0,
    'total_deals_we_accepted': 0,
    'total_deals_we_completed': 0,
    'total_deals_others_completed': 0,
    'total_other_monopoly_deals': 0,
    'games_others_traded_we_didnt': 0,
    'loss_opponents': {},  # opponent name -> loss count
    'win_monopoly_sets': {},  # set -> count in wins
    'loss_monopoly_sets': {},  # set -> count in losses
    'turn_first_monopoly_wins': [],
    'turn_first_monopoly_losses': [],
    # Color group with timing
    'win_set_turns': [],   # list of (set_name, turn_first_monopoly) for wins
    'loss_set_turns': [],  # list of (set_name, turn_first_monopoly) for losses
    # New: deal flow breakdown
    'propose_results': {},  # DealInfo name -> count (as proposer)
    'respond_results': {},  # DealInfo name -> count (as proposee)
    'our_reject_reasons': {},  # why WE rejected incoming
    'propose_targets': {},  # who we propose to
    'propose_types': {'mutual_monopoly': 0, 'setup': 0},
    'propose_cash_pay': [],  # fc values when we're paying
    'propose_cash_ask': [],  # fc values when we're asking
}

for g in range(N_GAMES):
    opponent_classes = random.sample(other_classes, 3)
    player_classes = [markov_class] + opponent_classes
    random.shuffle(player_classes)

    game = Game()
    for cls in player_classes:
        game.add_player(cls())

    game.play_game()

    # Find our AI in the game (could be in players or bankrupt_players)
    markov_player = None
    all_players = list(game.state.players) + list(game.state.bankrupt_players)
    for p in all_players:
        if p.ai.get_name() == "MarkovTradeAI":
            markov_player = p
            break

    won = game.winner is not None and game.winner.ai.get_name() == "MarkovTradeAI"
    winner_name = game.winner.ai.get_name() if game.winner else "DRAW"

    if won:
        wins += 1
    else:
        losses += 1

    # Pull diagnostics from the AI instance
    d = markov_player.ai._diag

    # Aggregate
    agg['total_deals_we_proposed'] += d['deals_we_proposed']
    agg['total_deals_we_accepted'] += d['deals_we_accepted']
    agg['total_deals_we_completed'] += d['deals_we_completed']
    agg['total_deals_others_completed'] += d['deals_others_completed']
    agg['total_other_monopoly_deals'] += len(d['other_monopoly_deals'])

    had_monopoly = d['ever_held_monopoly']

    if won:
        if had_monopoly:
            agg['wins_with_monopoly'] += 1
            for s in d['monopoly_sets']:
                k = str(s)
                agg['win_monopoly_sets'][k] = agg['win_monopoly_sets'].get(k, 0) + 1
                agg['win_set_turns'].append((k, d['turn_first_monopoly']))
        else:
            agg['wins_without_monopoly'] += 1
        if d['turn_first_monopoly']:
            agg['turn_first_monopoly_wins'].append(d['turn_first_monopoly'])
    else:
        if had_monopoly:
            agg['losses_with_monopoly'] += 1
            agg['loss_had_monopoly'] += 1
            for s in d['monopoly_sets']:
                k = str(s)
                agg['loss_monopoly_sets'][k] = agg['loss_monopoly_sets'].get(k, 0) + 1
                agg['loss_set_turns'].append((k, d['turn_first_monopoly']))
        else:
            agg['losses_without_monopoly'] += 1
            agg['loss_never_monopoly'] += 1
        agg['loss_opponents'][winner_name] = agg['loss_opponents'].get(winner_name, 0) + 1
        if d['turn_first_monopoly']:
            agg['turn_first_monopoly_losses'].append(d['turn_first_monopoly'])

    if d['deals_others_completed'] > 0 and d['deals_we_completed'] == 0:
        agg['games_others_traded_we_didnt'] += 1

    # Aggregate new deal flow data
    for k, v in d['propose_results'].items():
        agg['propose_results'][k] = agg['propose_results'].get(k, 0) + v
    for k, v in d['respond_results'].items():
        agg['respond_results'][k] = agg['respond_results'].get(k, 0) + v
    for k, v in d['our_reject_reasons'].items():
        agg['our_reject_reasons'][k] = agg['our_reject_reasons'].get(k, 0) + v
    for k, v in d['propose_targets'].items():
        agg['propose_targets'][k] = agg['propose_targets'].get(k, 0) + v
    for k, v in d['propose_types'].items():
        agg['propose_types'][k] = agg['propose_types'].get(k, 0) + v
    for entry in d.get('propose_cash_terms', []):
        if entry[0] == 'pay':
            agg['propose_cash_pay'].append(entry[2])  # actual offer
        elif entry[0] == 'ask':
            agg['propose_cash_ask'].append(entry[2])  # actual ask

    if (g + 1) % 10 == 0:
        print(f"\n--- After {g+1} games: {wins}/{g+1} ({100*wins/(g+1):.1f}%) ---")

# Final report
print("\n" + "=" * 60)
print(f"DIAGNOSTIC REPORT: {N_GAMES} games")
print("=" * 60)
print(f"\nWin rate: {wins}/{N_GAMES} ({100*wins/N_GAMES:.1f}%)")

print(f"\n--- MONOPOLY ACQUISITION ---")
print(f"Wins WITH monopoly:    {agg['wins_with_monopoly']}")
print(f"Wins WITHOUT monopoly: {agg['wins_without_monopoly']}")
print(f"Losses WITH monopoly:  {agg['losses_with_monopoly']}")
print(f"Losses WITHOUT monopoly (NEVER got one): {agg['losses_without_monopoly']}")
pct_loss_no_mono = 100 * agg['losses_without_monopoly'] / max(1, losses)
print(f"  -> {pct_loss_no_mono:.0f}% of losses were monopoly-less")

print(f"\n--- DEAL FLOW ---")
print(f"Deals we proposed (total): {agg['total_deals_we_proposed']}")
print(f"Deals we accepted (total): {agg['total_deals_we_accepted']}")
print(f"Deals WE completed:        {agg['total_deals_we_completed']}")
print(f"Deals OTHERS completed:    {agg['total_deals_others_completed']}")
print(f"Other monopoly-completing deals: {agg['total_other_monopoly_deals']}")
print(f"Games where others traded but we didn't: {agg['games_others_traded_we_didnt']}")

if agg['turn_first_monopoly_wins']:
    avg_w = sum(agg['turn_first_monopoly_wins']) / len(agg['turn_first_monopoly_wins'])
    print(f"\nAvg turn of first monopoly (wins):   {avg_w:.0f}")
if agg['turn_first_monopoly_losses']:
    avg_l = sum(agg['turn_first_monopoly_losses']) / len(agg['turn_first_monopoly_losses'])
    print(f"Avg turn of first monopoly (losses): {avg_l:.0f}")

print(f"\n--- MONOPOLY SETS (in wins) ---")
for k, v in sorted(agg['win_monopoly_sets'].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\n--- MONOPOLY SETS (in losses) ---")
for k, v in sorted(agg['loss_monopoly_sets'].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\n--- PROPOSAL OUTCOMES (as proposer) ---")
for k, v in sorted(agg['propose_results'].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\n--- RESPONSE OUTCOMES (as proposee) ---")
for k, v in sorted(agg['respond_results'].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\n--- WHY WE REJECT INCOMING DEALS ---")
for k, v in sorted(agg['our_reject_reasons'].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\n--- DEAL STRUCTURE ---")
for k, v in agg['propose_types'].items():
    print(f"  {k}: {v}")

print(f"\n--- CASH TERMS WHEN PROPOSING ---")
if agg['propose_cash_pay']:
    avg_pay = sum(agg['propose_cash_pay']) / len(agg['propose_cash_pay'])
    print(f"  We pay: {len(agg['propose_cash_pay'])} deals, avg offer={avg_pay:.0f}")
if agg['propose_cash_ask']:
    avg_ask = sum(agg['propose_cash_ask']) / len(agg['propose_cash_ask'])
    print(f"  They pay: {len(agg['propose_cash_ask'])} deals, avg ask={avg_ask:.0f}")
no_cash = agg['propose_types']['mutual_monopoly'] - len(agg['propose_cash_pay']) - len(agg['propose_cash_ask'])
if no_cash > 0:
    print(f"  Even (no cash): {no_cash} deals")

print(f"\n--- WHO WE PROPOSE TO ---")
for k, v in sorted(agg['propose_targets'].items(), key=lambda x: -x[1])[:15]:
    print(f"  {k}: {v}")

print(f"\n--- COLOR GROUP × TIMING (wins) ---")
print(f"  {'Set':<12} {'Count':>5} {'Avg Turn':>9} {'Early(<100)':>12} {'Mid(100-300)':>13} {'Late(>300)':>11}")
win_by_set = {}
for s, t in agg['win_set_turns']:
    if s not in win_by_set:
        win_by_set[s] = []
    win_by_set[s].append(t)
for s in sorted(win_by_set, key=lambda x: sum(win_by_set[x])/len(win_by_set[x])):
    turns = win_by_set[s]
    avg = sum(turns) / len(turns)
    early = sum(1 for t in turns if t < 100)
    mid = sum(1 for t in turns if 100 <= t <= 300)
    late = sum(1 for t in turns if t > 300)
    print(f"  {s:<12} {len(turns):>5} {avg:>9.0f} {early:>12} {mid:>13} {late:>11}")

print(f"\n--- COLOR GROUP × TIMING (losses) ---")
print(f"  {'Set':<12} {'Count':>5} {'Avg Turn':>9} {'Early(<100)':>12} {'Mid(100-300)':>13} {'Late(>300)':>11}")
loss_by_set = {}
for s, t in agg['loss_set_turns']:
    if s not in loss_by_set:
        loss_by_set[s] = []
    loss_by_set[s].append(t)
for s in sorted(loss_by_set, key=lambda x: sum(loss_by_set[x])/len(loss_by_set[x])):
    turns = loss_by_set[s]
    avg = sum(turns) / len(turns)
    early = sum(1 for t in turns if t < 100)
    mid = sum(1 for t in turns if 100 <= t <= 300)
    late = sum(1 for t in turns if t > 300)
    print(f"  {s:<12} {len(turns):>5} {avg:>9.0f} {early:>12} {mid:>13} {late:>11}")

# Win rate by timing bucket
print(f"\n--- WIN RATE BY TIMING OF FIRST MONOPOLY ---")
all_turns = [(t, True) for t in agg['turn_first_monopoly_wins']] + \
            [(t, False) for t in agg['turn_first_monopoly_losses']]
buckets = [('< 50', 0, 50), ('50-100', 50, 100), ('100-200', 100, 200),
           ('200-500', 200, 500), ('500+', 500, 99999)]
for label, lo, hi in buckets:
    in_bucket = [(t, w) for t, w in all_turns if lo <= t < hi]
    if in_bucket:
        wins_b = sum(1 for _, w in in_bucket if w)
        print(f"  {label:>8}: {wins_b}/{len(in_bucket)} ({100*wins_b/len(in_bucket):.0f}%)")

print(f"\n--- LOSS OPPONENTS ---")
for k, v in sorted(agg['loss_opponents'].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
