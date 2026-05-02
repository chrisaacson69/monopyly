"""
Deal Analysis: Why are MarkovTradeAI's proposals rejected?
Breaks down by board state, proposee state, deal type, and outcomes.
"""
import sys
import random
import logging
from collections import defaultdict
from monopyly import *
from monopyly.game.game import Game
from monopyly.game.deal_response import DealResponse
from monopyly.game.deal_result import DealResult
from monopyly.squares.street import Street

Logger.add_handler(ConsoleLogHandler(Logger.WARNING))
logging.getLogger('markov_diag').setLevel(logging.WARNING)

TOP_12 = {'Edmund', 'Baldrick', 'LordMelchitt', 'DarkRedLight',
           'SimpleMind', 'Percy', 'Queenie', 'RedLight',
           'Catbert', 'Ratbert', 'DarkGreenLight', 'MarkovTradeAI'}
all_ais = load_ais()
top_ais = [ai for ai in all_ais if ai.get_name() in TOP_12]
markov_class = None
other_classes = []
for ai in top_ais:
    cls = type(ai)
    if ai.get_name() == 'MarkovTradeAI':
        markov_class = cls
    else:
        other_classes.append(cls)

proposal_log = []

class InstrumentedGame(Game):
    def _make_deal(self, current_player):
        proposal = current_player.call_ai(
            current_player.ai.propose_deal, self.state, current_player)

        if not proposal or not proposal.propose_to_player:
            if proposal and not proposal.propose_to_player:
                current_player.call_ai(current_player.ai.deal_result,
                    PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            return

        proposer_name = current_player.ai.get_name()
        proposed_to = proposal.propose_to_player
        proposee_name = proposed_to.ai.get_name()

        if current_player is proposed_to:
            current_player.call_ai(current_player.ai.deal_result,
                PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            return

        # Capture context for MarkovTradeAI proposals
        record = None
        if proposer_name == 'MarkovTradeAI':
            board = self.state.board
            all_props = [sq for sq in board.squares if hasattr(sq, 'owner')]
            unowned = [p for p in all_props if p.owner is None]
            proposee_has_monopoly = len(proposed_to.state.owned_unmortgaged_sets) > 0
            proposee_has_houses = any(
                isinstance(p, Street) and p.number_of_houses > 0
                for p in proposed_to.state.properties)

            # Classify deal type
            offered_names = [p.name for p in proposal.properties_offered]
            wanted_names = [p.name for p in proposal.properties_wanted]

            # Would this complete a set for either side?
            completes_us = False
            completes_them = False
            for p in proposal.properties_wanted:
                if hasattr(p, 'property_set'):
                    set_props = list(p.property_set.properties)
                    my_owned = sum(1 for sp in set_props if sp.owner == current_player)
                    my_gaining = sum(1 for sp in set_props if sp in proposal.properties_wanted)
                    my_losing = sum(1 for sp in set_props if sp in proposal.properties_offered)
                    if my_owned + my_gaining - my_losing == len(set_props):
                        completes_us = True
            for p in proposal.properties_offered:
                if hasattr(p, 'property_set'):
                    set_props = list(p.property_set.properties)
                    their_owned = sum(1 for sp in set_props if sp.owner == proposed_to)
                    their_gaining = sum(1 for sp in set_props if sp in proposal.properties_offered)
                    their_losing = sum(1 for sp in set_props if sp in proposal.properties_wanted)
                    if their_owned + their_gaining - their_losing == len(set_props):
                        completes_them = True

            is_setup = (len(proposal.properties_offered) > 0 and
                       len(proposal.properties_wanted) > 0 and
                       not completes_us and not completes_them)

            record = {
                'proposee': proposee_name,
                'unowned_count': len(unowned),
                'all_sold': len(unowned) == 0,
                'proposee_has_monopoly': proposee_has_monopoly,
                'proposee_has_houses': proposee_has_houses,
                'offered': offered_names,
                'wanted': wanted_names,
                'cash_offered': proposal.maximum_cash_offered,
                'cash_wanted': proposal.minimum_cash_wanted,
                'is_setup': is_setup,
                'completes_us': completes_us,
                'completes_them': completes_them,
                'mutual_completion': completes_us and completes_them,
            }

        # Validate
        def validate_properties(player, properties):
            for prop in properties:
                if not player.owns_properties([prop]):
                    return False
                if isinstance(prop, Street) and prop.number_of_houses > 0:
                    if not set(properties).issuperset(set(prop.property_set.properties)):
                        return False
            return True

        if not validate_properties(current_player, proposal.properties_offered):
            current_player.call_ai(current_player.ai.deal_result,
                PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            if record:
                record['outcome'] = 'INVALID'
                proposal_log.append(record)
            return
        if not validate_properties(proposed_to, proposal.properties_wanted):
            current_player.call_ai(current_player.ai.deal_result,
                PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            if record:
                record['outcome'] = 'INVALID'
                proposal_log.append(record)
            return

        # Redact cash, ask proposee
        max_cash = proposal.maximum_cash_offered
        min_cash = proposal.minimum_cash_wanted
        proposal.maximum_cash_offered = 0
        proposal.minimum_cash_wanted = 0
        proposal.proposed_by_player = current_player

        response = proposed_to.call_ai(
            proposed_to.ai.deal_proposed, self.state, proposed_to, proposal)

        proposal.maximum_cash_offered = max_cash
        proposal.minimum_cash_wanted = min_cash

        outcome = 'REJECTED'
        cash_transfer = 0

        if response and response.action == DealResponse.Action.ACCEPT:
            if min_cash > 0:
                if response.maximum_cash_offered < min_cash:
                    outcome = 'ASKED_TOO_MUCH'
                else:
                    cash_transfer = -int((min_cash + response.maximum_cash_offered) / 2)
                    outcome = 'SUCCEEDED'
            elif max_cash > 0:
                if response.minimum_cash_wanted > max_cash:
                    outcome = 'OFFERED_TOO_LITTLE'
                else:
                    cash_transfer = int((max_cash + response.minimum_cash_wanted) / 2)
                    outcome = 'SUCCEEDED'
            else:
                outcome = 'SUCCEEDED'

            if outcome == 'SUCCEEDED':
                if cash_transfer > 0 and current_player.state.cash < cash_transfer:
                    outcome = 'CANT_AFFORD'
                elif cash_transfer < 0 and proposed_to.state.cash < -cash_transfer:
                    outcome = 'CANT_AFFORD'
                else:
                    for prop in proposal.properties_offered:
                        prop.owner = proposed_to
                        current_player.state.properties.discard(prop)
                        proposed_to.state.properties.add(prop)
                    for prop in proposal.properties_wanted:
                        prop.owner = current_player
                        proposed_to.state.properties.discard(prop)
                        current_player.state.properties.add(prop)
                    if cash_transfer != 0:
                        current_player.state.cash -= cash_transfer
                        proposed_to.state.cash += cash_transfer
                    self._update_sets()

        if record:
            record['outcome'] = outcome
            proposal_log.append(record)

        # Notifications
        if outcome == 'SUCCEEDED':
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.SUCCEEDED)
            proposed_to.call_ai(proposed_to.ai.deal_result, PlayerAIBase.DealInfo.SUCCEEDED)
            dr = DealResult()
            dr.proposer = current_player
            dr.proposee = proposed_to
            dr.properties_transferred_to_proposee = proposal.properties_offered
            dr.properties_transferred_to_proposer = proposal.properties_wanted
            dr.cash_transferred_from_proposer_to_proposee = cash_transfer
            for p in self.state.players:
                p.call_ai(p.ai.deal_completed, dr)
        elif outcome == 'REJECTED':
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.DEAL_REJECTED)
            proposed_to.call_ai(proposed_to.ai.deal_result, PlayerAIBase.DealInfo.DEAL_REJECTED)
        elif outcome == 'ASKED_TOO_MUCH':
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.ASKED_FOR_TOO_MUCH_MONEY)
            proposed_to.call_ai(proposed_to.ai.deal_result, PlayerAIBase.DealInfo.OFFERED_TOO_LITTLE_MONEY)
        elif outcome == 'OFFERED_TOO_LITTLE':
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.OFFERED_TOO_LITTLE_MONEY)
            proposed_to.call_ai(proposed_to.ai.deal_result, PlayerAIBase.DealInfo.ASKED_FOR_TOO_MUCH_MONEY)
        else:
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.PLAYER_DID_NOT_HAVE_ENOUGH_MONEY)
            proposed_to.call_ai(proposed_to.ai.deal_result, PlayerAIBase.DealInfo.PLAYER_DID_NOT_HAVE_ENOUGH_MONEY)

    def _make_deals(self, current_player):
        self._in_make_deals = True
        for i in range(3):
            self._make_deal(current_player)
        self._in_make_deals = False


# ---- Run games ----
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
wins = 0
games_we_complete = 0
games_others_complete_not_us = 0
completed_deal_details = []  # track what deals DO succeed (for all players)

for g in range(N):
    opponents = random.sample(other_classes, 3)
    players = [markov_class] + opponents
    random.shuffle(players)
    game = InstrumentedGame()
    game.eminent_domain = False
    for cls in players:
        game.add_player(cls())
    game.play_game()

    mp = None
    for p in list(game.state.players) + list(game.state.bankrupt_players):
        if p.ai.get_name() == 'MarkovTradeAI':
            mp = p
            break
    won = game.winner and game.winner.ai.get_name() == 'MarkovTradeAI'
    if won:
        wins += 1
    d = mp.ai._diag
    if d['deals_we_completed'] > 0:
        games_we_complete += 1
    if d['deals_others_completed'] > 0 and d['deals_we_completed'] == 0:
        games_others_complete_not_us += 1
    if (g + 1) % 25 == 0:
        print(f"  {g+1}/{N}...")

# ---- Analysis ----
print()
print("=" * 70)
print(f"DEAL ANALYSIS — {len(proposal_log)} MarkovTradeAI proposals, {N} games")
print(f"Win rate: {wins}/{N} ({100*wins/N:.1f}%)")
print("=" * 70)

# Q1: Strip out "they won't trade" cases
before_sold = [p for p in proposal_log if not p['all_sold']]
after_sold = [p for p in proposal_log if p['all_sold']]

print(f"\n1. PROPOSALS BY BOARD STATE")
print(f"   Before all sold: {len(before_sold)}")
print(f"   After all sold:  {len(after_sold)}")

for label, subset in [("BEFORE all sold", before_sold), ("AFTER all sold", after_sold)]:
    if not subset:
        continue
    outcomes = defaultdict(int)
    for p in subset:
        outcomes[p['outcome']] += 1
    total = len(subset)
    print(f"\n   [{label}] — {total} proposals:")
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"     {k}: {v} ({v/total*100:.1f}%)")

    # By proposee state
    has_houses = [p for p in subset if p['proposee_has_houses']]
    has_mono_no_houses = [p for p in subset if p['proposee_has_monopoly'] and not p['proposee_has_houses']]
    no_mono = [p for p in subset if not p['proposee_has_monopoly']]

    print(f"\n   [{label}] By proposee state:")
    for state_label, state_sub in [("already has houses", has_houses),
                                     ("has monopoly, no houses yet", has_mono_no_houses),
                                     ("no monopoly", no_mono)]:
        if not state_sub:
            continue
        oc = defaultdict(int)
        for p in state_sub:
            oc[p['outcome']] += 1
        t = len(state_sub)
        parts = ", ".join(f"{k}={v}({v/t*100:.0f}%)" for k, v in sorted(oc.items(), key=lambda x: -x[1]))
        print(f"     {state_label} [{t}]: {parts}")

# Q1b: Rejection by proposee AI
print(f"\n2. REJECTIONS BY OPPONENT AI (after all sold, proposee no monopoly)")
interesting = [p for p in after_sold if not p['proposee_has_monopoly']]
by_ai = defaultdict(lambda: defaultdict(int))
for p in interesting:
    by_ai[p['proposee']][p['outcome']] += 1
for ai_name in sorted(by_ai.keys()):
    oc = by_ai[ai_name]
    total = sum(oc.values())
    parts = ", ".join(f"{k}={v}" for k, v in sorted(oc.items(), key=lambda x: -x[1]))
    print(f"   {ai_name:<16} [{total:>4}]: {parts}")

# Q2: Setup trades
print(f"\n3. DEAL TYPE BREAKDOWN")
setups = [p for p in proposal_log if p['is_setup']]
mutual = [p for p in proposal_log if p['mutual_completion']]
one_sided_us = [p for p in proposal_log if p['completes_us'] and not p['completes_them']]
one_sided_them = [p for p in proposal_log if p['completes_them'] and not p['completes_us']]
other = [p for p in proposal_log if not p['is_setup'] and not p['completes_us'] and not p['completes_them']]

for label, subset in [("Setup (non-completion)", setups),
                       ("Mutual completion", mutual),
                       ("Completes US only", one_sided_us),
                       ("Completes THEM only", one_sided_them),
                       ("Other/unclassified", other)]:
    if not subset:
        continue
    oc = defaultdict(int)
    for p in subset:
        oc[p['outcome']] += 1
    total = len(subset)
    parts = ", ".join(f"{k}={v}({v/total*100:.1f}%)" for k, v in sorted(oc.items(), key=lambda x: -x[1]))
    print(f"   {label} [{total}]: {parts}")

# Successful deals detail
print(f"\n4. SUCCESSFUL DEALS (all types)")
successes = [p for p in proposal_log if p['outcome'] == 'SUCCEEDED']
for p in successes:
    dtype = "SETUP" if p['is_setup'] else "MUTUAL" if p['mutual_completion'] else "ONE-SIDED-US" if p['completes_us'] else "ONE-SIDED-THEM" if p['completes_them'] else "OTHER"
    print(f"   [{dtype}] to={p['proposee']}: "
          f"offered={p['offered']}, wanted={p['wanted']}, "
          f"cash_offer={p['cash_offered']}, cash_want={p['cash_wanted']}")

# Q4
print(f"\n5. DEAL COMPLETION RATES")
print(f"   Games we completed a deal:            {games_we_complete}/{N} ({100*games_we_complete/N:.0f}%)")
print(f"   Games others traded without us:       {games_others_complete_not_us}/{N} ({100*games_others_complete_not_us/N:.0f}%)")
print(f"   Games nobody completed any deal:      {N - games_we_complete - games_others_complete_not_us}/{N}")
