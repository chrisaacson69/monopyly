"""
Deal Analysis v2: Full AI field.
Tests whether MarkovTradeAI's deal quality is the problem, or opponent composition.
Tracks ALL deals in each game (not just ours) for comparison.
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

# For suppressing noisy AI print statements during game play
import io, os

# Full field: every AI
all_ais = load_ais()
markov_class = None
other_classes = []
for ai in all_ais:
    cls = type(ai)
    if ai.get_name() == 'MarkovTradeAI':
        markov_class = cls
    else:
        other_classes.append(cls)

print(f"Loaded {len(other_classes)} opponent AIs + MarkovTradeAI")

# Logs
markov_proposals = []   # our proposals (detailed)
all_completed = []      # ALL successful deals in every game

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

        # Board state (shared)
        board = self.state.board
        all_props = [sq for sq in board.squares if hasattr(sq, 'owner')]
        unowned = [p for p in all_props if p.owner is None]
        all_sold = len(unowned) == 0

        # Classify deal completion
        offered_names = [p.name for p in proposal.properties_offered]
        wanted_names = [p.name for p in proposal.properties_wanted]

        completes_proposer = False
        completes_proposee = False
        for p in proposal.properties_wanted:
            if hasattr(p, 'property_set'):
                set_props = list(p.property_set.properties)
                owned = sum(1 for sp in set_props if sp.owner == current_player)
                gaining = sum(1 for sp in set_props if sp in proposal.properties_wanted)
                losing = sum(1 for sp in set_props if sp in proposal.properties_offered)
                if owned + gaining - losing == len(set_props):
                    completes_proposer = True
        for p in proposal.properties_offered:
            if hasattr(p, 'property_set'):
                set_props = list(p.property_set.properties)
                owned = sum(1 for sp in set_props if sp.owner == proposed_to)
                gaining = sum(1 for sp in set_props if sp in proposal.properties_offered)
                losing = sum(1 for sp in set_props if sp in proposal.properties_wanted)
                if owned + gaining - losing == len(set_props):
                    completes_proposee = True

        is_setup = (len(proposal.properties_offered) > 0 and
                   len(proposal.properties_wanted) > 0 and
                   not completes_proposer and not completes_proposee)

        # MarkovTradeAI-specific record
        markov_record = None
        if proposer_name == 'MarkovTradeAI':
            proposee_has_monopoly = len(proposed_to.state.owned_unmortgaged_sets) > 0
            proposee_has_houses = any(
                isinstance(p, Street) and p.number_of_houses > 0
                for p in proposed_to.state.properties)
            markov_record = {
                'proposee': proposee_name,
                'all_sold': all_sold,
                'proposee_has_monopoly': proposee_has_monopoly,
                'proposee_has_houses': proposee_has_houses,
                'offered': offered_names,
                'wanted': wanted_names,
                'cash_offered': proposal.maximum_cash_offered,
                'cash_wanted': proposal.minimum_cash_wanted,
                'is_setup': is_setup,
                'completes_us': completes_proposer,
                'completes_them': completes_proposee,
                'mutual_completion': completes_proposer and completes_proposee,
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
            if markov_record:
                markov_record['outcome'] = 'INVALID'
                markov_proposals.append(markov_record)
            return
        if not validate_properties(proposed_to, proposal.properties_wanted):
            current_player.call_ai(current_player.ai.deal_result,
                PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            if markov_record:
                markov_record['outcome'] = 'INVALID'
                markov_proposals.append(markov_record)
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

        # Log MarkovTradeAI proposals
        if markov_record:
            markov_record['outcome'] = outcome
            markov_proposals.append(markov_record)

        # Log ALL successful deals
        if outcome == 'SUCCEEDED':
            all_completed.append({
                'proposer': proposer_name,
                'proposee': proposee_name,
                'offered': offered_names,
                'wanted': wanted_names,
                'cash_transfer': cash_transfer,
                'is_setup': is_setup,
                'completes_proposer': completes_proposer,
                'completes_proposee': completes_proposee,
                'all_sold': all_sold,
            })

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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
wins = 0

crashed = 0
for g in range(N):
    opponents = random.sample(other_classes, 3)
    players = [markov_class] + opponents
    random.shuffle(players)
    game = InstrumentedGame()
    game.eminent_domain = False
    for cls in players:
        game.add_player(cls())
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        game.play_game()
        sys.stdout = old_stdout
    except Exception:
        sys.stdout = old_stdout
        crashed += 1
        continue

    won = game.winner and game.winner.ai.get_name() == 'MarkovTradeAI'
    if won:
        wins += 1
    if (g + 1) % 50 == 0:
        print(f"  {g+1}/{N}...")

# ---- Analysis ----
print()
print("=" * 70)
print(f"FULL-FIELD DEAL ANALYSIS — {N} games, {len(markov_proposals)} MarkovTradeAI proposals")
completed = N - crashed
print(f"Completed: {completed}/{N} (crashed: {crashed})")
print(f"Win rate: {wins}/{completed} ({100*wins/completed:.1f}%)" if completed > 0 else "No games completed")
print(f"All completed deals in field: {len(all_completed)}")
print("=" * 70)

# 1. Our acceptance rate
total_p = len(markov_proposals)
if total_p > 0:
    by_outcome = defaultdict(int)
    for p in markov_proposals:
        by_outcome[p['outcome']] += 1
    print(f"\n1. MARKOV PROPOSAL OUTCOMES ({total_p} total)")
    for k, v in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"   {k}: {v} ({v/total_p*100:.1f}%)")

# 2. Acceptance rate by opponent AI
print(f"\n2. MARKOV ACCEPTANCE RATE BY OPPONENT")
by_ai = defaultdict(lambda: {'total': 0, 'accepted': 0, 'asked_too_much': 0, 'offered_too_little': 0})
for p in markov_proposals:
    d = by_ai[p['proposee']]
    d['total'] += 1
    if p['outcome'] == 'SUCCEEDED':
        d['accepted'] += 1
    elif p['outcome'] == 'ASKED_TOO_MUCH':
        d['asked_too_much'] += 1
    elif p['outcome'] == 'OFFERED_TOO_LITTLE':
        d['offered_too_little'] += 1
# Sort by acceptance rate desc
sorted_ais = sorted(by_ai.items(), key=lambda x: -x[1]['accepted']/max(x[1]['total'],1))
for name, d in sorted_ais:
    rate = 100*d['accepted']/d['total'] if d['total'] > 0 else 0
    cash_issues = d['asked_too_much'] + d['offered_too_little']
    extra = f"  (cash_mismatch={cash_issues})" if cash_issues > 0 else ""
    print(f"   {name:<30} [{d['total']:>4}]: accepted={d['accepted']} ({rate:.0f}%){extra}")

# 3. Who else is trading? (all completed deals, any proposer)
print(f"\n3. ALL COMPLETED DEALS — WHO TRADES WITH WHOM")
by_proposer = defaultdict(int)
by_proposee = defaultdict(int)
for d in all_completed:
    by_proposer[d['proposer']] += 1
    by_proposee[d['proposee']] += 1

print(f"   Top proposers (closed deals):")
for name, count in sorted(by_proposer.items(), key=lambda x: -x[1])[:15]:
    print(f"     {name:<30} {count}")
print(f"   Top acceptors (accepted deals):")
for name, count in sorted(by_proposee.items(), key=lambda x: -x[1])[:15]:
    print(f"     {name:<30} {count}")

# 4. Overlap: AIs that accept others but reject us
print(f"\n4. AIs THAT ACCEPT OTHERS BUT REJECT US")
our_accepted_by = set(p['proposee'] for p in markov_proposals if p['outcome'] == 'SUCCEEDED')
our_rejected_by = set(p['proposee'] for p in markov_proposals if p['outcome'] != 'SUCCEEDED') - our_accepted_by
others_accepted_by = set(d['proposee'] for d in all_completed if d['proposer'] != 'MarkovTradeAI')

accept_others_reject_us = others_accepted_by & our_rejected_by
if accept_others_reject_us:
    for name in sorted(accept_others_reject_us):
        others_count = sum(1 for d in all_completed if d['proposee'] == name and d['proposer'] != 'MarkovTradeAI')
        our_count = sum(1 for p in markov_proposals if p['proposee'] == name)
        our_success = sum(1 for p in markov_proposals if p['proposee'] == name and p['outcome'] == 'SUCCEEDED')
        print(f"   {name:<30} accepts others: {others_count}, our proposals: {our_count}, our accepted: {our_success}")
else:
    print("   (none found)")

# 5. Cash terms: rejected vs accepted
print(f"\n5. CASH TERMS — REJECTED vs ACCEPTED")
accepted = [p for p in markov_proposals if p['outcome'] == 'SUCCEEDED']
rejected = [p for p in markov_proposals if p['outcome'] == 'REJECTED']
cash_mismatch = [p for p in markov_proposals if p['outcome'] in ('ASKED_TOO_MUCH', 'OFFERED_TOO_LITTLE')]

def cash_summary(subset, label):
    if not subset:
        print(f"   {label}: (none)")
        return
    offering = [p for p in subset if p['cash_offered'] > 0]
    wanting = [p for p in subset if p['cash_wanted'] > 0]
    even = [p for p in subset if p['cash_offered'] == 0 and p['cash_wanted'] == 0]
    print(f"   {label} [{len(subset)}]:")
    print(f"     We offer cash: {len(offering)}" +
          (f"  avg=${sum(p['cash_offered'] for p in offering)/len(offering):.0f}" if offering else ""))
    print(f"     We want cash:  {len(wanting)}" +
          (f"  avg=${sum(p['cash_wanted'] for p in wanting)/len(wanting):.0f}" if wanting else ""))
    print(f"     Even swap:     {len(even)}")

cash_summary(accepted, "ACCEPTED")
cash_summary(rejected, "REJECTED")
cash_summary(cash_mismatch, "CASH MISMATCH (they accepted properties but cash didn't align)")

# 6. Deal type breakdown
print(f"\n6. DEAL TYPE BREAKDOWN")
for label, filt in [("Setup", lambda p: p['is_setup']),
                     ("Mutual completion", lambda p: p['mutual_completion']),
                     ("Completes US only", lambda p: p['completes_us'] and not p['completes_them']),
                     ("Completes THEM only", lambda p: p['completes_them'] and not p['completes_us']),
                     ]:
    subset = [p for p in markov_proposals if filt(p)]
    if not subset:
        continue
    succ = sum(1 for p in subset if p['outcome'] == 'SUCCEEDED')
    rej = sum(1 for p in subset if p['outcome'] == 'REJECTED')
    cash_miss = sum(1 for p in subset if p['outcome'] in ('ASKED_TOO_MUCH', 'OFFERED_TOO_LITTLE'))
    print(f"   {label} [{len(subset)}]: accepted={succ} ({100*succ/len(subset):.1f}%), rejected={rej}, cash_mismatch={cash_miss}")

# 7. Successful deals detail
print(f"\n7. OUR SUCCESSFUL DEALS")
for p in accepted:
    dtype = "SETUP" if p['is_setup'] else "MUTUAL" if p['mutual_completion'] else "ONE-SIDED-US" if p['completes_us'] else "ONE-SIDED-THEM" if p['completes_them'] else "OTHER"
    print(f"   [{dtype}] to={p['proposee']}: "
          f"offered={p['offered']}, wanted={p['wanted']}, "
          f"cash_offer={p['cash_offered']}, cash_want={p['cash_wanted']}")

# 8. Others' successful deals (sample)
print(f"\n8. OTHERS' SUCCESSFUL DEALS (sample, up to 30)")
others_deals = [d for d in all_completed if d['proposer'] != 'MarkovTradeAI']
for d in others_deals[:30]:
    dtype = "SETUP" if d['is_setup'] else "MUTUAL" if d['completes_proposer'] and d['completes_proposee'] else "ONE-SIDED" if d['completes_proposer'] or d['completes_proposee'] else "OTHER"
    print(f"   [{dtype}] {d['proposer']} → {d['proposee']}: "
          f"gave={d['offered']}, got={d['wanted']}, cash={d['cash_transfer']}")
