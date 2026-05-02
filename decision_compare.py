"""
Decision Comparison Harness
===========================
Runs games with top AIs, and at each decision point queries a "shadow"
MarkovTradeAI instance with the same game state.  Logs both decisions
side-by-side for analysis.

Only considers non-eminent-domain games.

Usage:
    python decision_compare.py [num_games]   (default 50)
"""
import sys
import json
import copy
from collections import defaultdict
from monopyly import *
from monopyly.game.game import Game
from monopyly.game.deal_proposal import DealProposal
from monopyly.game.deal_response import DealResponse
from monopyly.squares.street import Street

# ---------------------------------------------------------------------------
# Load AIs
# ---------------------------------------------------------------------------
all_ais = load_ais()
ai_by_name = {ai.get_name(): ai for ai in all_ais}

# The shadow AI — a fresh MarkovTradeAI instance used for comparison
SHADOW_NAME = "MarkovTradeAI"
shadow_ai = ai_by_name[SHADOW_NAME]

# Top AIs to watch (exclude MarkovTradeAI — it IS the shadow)
WATCH_AIS = {"Baldrick", "Edmund", "Percy", "Queenie", "LordMelchitt", "DarkRedLight"}

# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------
decisions = []

def log_decision(decision_type, turn, player_name, context, real_decision, shadow_decision):
    """Append a decision record."""
    decisions.append({
        "type": decision_type,
        "turn": turn,
        "player": player_name,
        "context": context,
        "real": real_decision,
        "shadow": shadow_decision,
    })

# ---------------------------------------------------------------------------
# Helper: describe a property for logging
# ---------------------------------------------------------------------------
def prop_name(prop):
    if prop is None:
        return "None"
    return prop.name

def prop_list(props):
    return [prop_name(p) for p in props] if props else []

def describe_deal_proposal(proposal):
    if proposal is None:
        return None
    return {
        "to": proposal.propose_to_player.ai.get_name() if proposal.propose_to_player else None,
        "offered": prop_list(proposal.properties_offered),
        "wanted": prop_list(proposal.properties_wanted),
        "max_cash_offered": proposal.maximum_cash_offered,
        "min_cash_wanted": proposal.minimum_cash_wanted,
    }

def describe_deal_response(response):
    if response is None:
        return "REJECT"
    if response.action == DealResponse.Action.ACCEPT:
        return {
            "action": "ACCEPT",
            "max_cash_offered": response.maximum_cash_offered,
            "min_cash_wanted": response.minimum_cash_wanted,
        }
    return "REJECT"

def describe_build(instructions):
    if not instructions:
        return []
    return [(prop_name(street), n) for street, n in instructions]

def describe_board_state(game_state, player):
    """Compact snapshot of the player's position."""
    props = sorted([p.name for p in player.state.properties])
    sets = [str(s) for s in player.state.owned_unmortgaged_sets]
    return {
        "cash": player.state.cash,
        "properties": props,
        "sets": sets,
        "net_worth": player.net_worth,
    }

# ---------------------------------------------------------------------------
# Instrumented Game subclass
# ---------------------------------------------------------------------------
class CompareGame(Game):
    """
    Runs a normal game but intercepts decision points for watched AIs,
    queries the shadow AI with the same state, and logs both decisions.
    """

    def __init__(self):
        super().__init__()
        self.eminent_domain = False
        self._turn_counter = 0
        self._shadow = None  # will be set before play

    def set_shadow(self, shadow):
        self._shadow = shadow

    def _is_watched(self, player):
        return player.ai.get_name() in WATCH_AIS

    # --- Notifications: forward to shadow so it tracks game state ---

    def _notify_shadow_start(self):
        self._shadow.start_of_game()

    def _notify_shadow_turn(self, game_state, current_player):
        self._shadow.start_of_turn(game_state, current_player)

    def _notify_shadow_deal_completed(self, deal_result):
        self._shadow.deal_completed(deal_result)

    def _notify_shadow_auction_result(self, status, prop, player, amount):
        self._shadow.auction_result(status, prop, player, amount)

    def _notify_shadow_bankrupt(self, player):
        self._shadow.player_went_bankrupt(player)

    def _notify_shadow_money_will_be_taken(self, player, amount):
        self._shadow.money_will_be_taken(player, amount)

    # --- Override play_game to hook start ---

    def play_game(self):
        self._notify_shadow_start()
        return super().play_game()

    # --- Override play_one_turn to hook turn notifications ---

    def play_one_turn(self, current_player):
        self._turn_counter += 1

        # Forward start_of_turn to shadow for ALL players (mirrors game.py:191)
        for player in self.state.players:
            self._notify_shadow_turn(self.state, current_player)

        # Now run the real turn
        super().play_one_turn(current_player)

    # --- Override _make_deal to intercept propose_deal ---

    def _make_deal(self, current_player):
        # Get the real proposal
        proposal = current_player.call_ai(
            current_player.ai.propose_deal,
            self.state,
            current_player)

        # If this is a watched AI, also query shadow
        if self._is_watched(current_player):
            shadow_proposal = self._shadow.propose_deal(self.state, current_player)
            ctx = describe_board_state(self.state, current_player)
            log_decision(
                "propose_deal",
                self._turn_counter,
                current_player.ai.get_name(),
                ctx,
                describe_deal_proposal(proposal),
                describe_deal_proposal(shadow_proposal),
            )

        if not proposal:
            return

        # --- Validate and execute the deal (replicate game.py logic) ---
        if proposal.propose_to_player is None:
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            return
        proposed_to_player = proposal.propose_to_player

        if current_player is proposed_to_player:
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            return

        # Validate ownership
        def validate_properties(player, properties):
            for prop in properties:
                if player.owns_properties([prop]) is False:
                    return False
                if isinstance(prop, Street) and prop.number_of_houses > 0:
                    props_in_set = prop.property_set.properties
                    if not set(properties).issuperset(set(props_in_set)):
                        return False
            return True

        if not validate_properties(current_player, proposal.properties_offered):
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            return
        if not validate_properties(proposed_to_player, proposal.properties_wanted):
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.INVALID_DEAL_PROPOSED)
            return

        # Redact cash and ask proposee
        max_cash = proposal.maximum_cash_offered
        min_cash = proposal.minimum_cash_wanted
        proposal.maximum_cash_offered = 0
        proposal.minimum_cash_wanted = 0
        proposal.proposed_by_player = current_player

        response = proposed_to_player.call_ai(
            proposed_to_player.ai.deal_proposed,
            self.state,
            proposed_to_player,
            proposal)

        # If proposee is watched, also query shadow for what it would decide
        if self._is_watched(proposed_to_player):
            shadow_response = self._shadow.deal_proposed(self.state, proposed_to_player, proposal)
            ctx = {
                **describe_board_state(self.state, proposed_to_player),
                "from": current_player.ai.get_name(),
                "offered": prop_list(proposal.properties_offered),
                "wanted": prop_list(proposal.properties_wanted),
            }
            log_decision(
                "deal_proposed",
                self._turn_counter,
                proposed_to_player.ai.get_name(),
                ctx,
                describe_deal_response(response),
                describe_deal_response(shadow_response),
            )

        # Restore cash fields for deal execution
        proposal.maximum_cash_offered = max_cash
        proposal.minimum_cash_wanted = min_cash

        if (response is None) or (response.action == DealResponse.Action.REJECT):
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.DEAL_REJECTED)
            proposed_to_player.call_ai(proposed_to_player.ai.deal_result, PlayerAIBase.DealInfo.DEAL_REJECTED)
            return

        # Cash negotiation (replicate game.py logic)
        cash_transfer = 0
        if min_cash > 0:
            if response.maximum_cash_offered < min_cash:
                current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.ASKED_FOR_TOO_MUCH_MONEY)
                proposed_to_player.call_ai(proposed_to_player.ai.deal_result, PlayerAIBase.DealInfo.OFFERED_TOO_LITTLE_MONEY)
                return
            cash_transfer = -int((min_cash + response.maximum_cash_offered) / 2)
        elif max_cash > 0:
            if response.minimum_cash_wanted > max_cash:
                current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.OFFERED_TOO_LITTLE_MONEY)
                proposed_to_player.call_ai(proposed_to_player.ai.deal_result, PlayerAIBase.DealInfo.ASKED_FOR_TOO_MUCH_MONEY)
                return
            cash_transfer = int((max_cash + response.minimum_cash_wanted) / 2)

        # Check proposer can afford
        if cash_transfer > 0 and current_player.state.cash < cash_transfer:
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.PLAYER_DID_NOT_HAVE_ENOUGH_MONEY)
            proposed_to_player.call_ai(proposed_to_player.ai.deal_result, PlayerAIBase.DealInfo.PLAYER_DID_NOT_HAVE_ENOUGH_MONEY)
            return
        if cash_transfer < 0 and proposed_to_player.state.cash < -cash_transfer:
            current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.PLAYER_DID_NOT_HAVE_ENOUGH_MONEY)
            proposed_to_player.call_ai(proposed_to_player.ai.deal_result, PlayerAIBase.DealInfo.PLAYER_DID_NOT_HAVE_ENOUGH_MONEY)
            return

        # Execute the deal
        # Transfer properties
        for prop in proposal.properties_offered:
            prop.owner = proposed_to_player
            current_player.state.properties.discard(prop)
            proposed_to_player.state.properties.add(prop)
        for prop in proposal.properties_wanted:
            prop.owner = current_player
            proposed_to_player.state.properties.discard(prop)
            current_player.state.properties.add(prop)

        # Transfer cash
        if cash_transfer > 0:
            current_player.state.cash -= cash_transfer
            proposed_to_player.state.cash += cash_transfer
        elif cash_transfer < 0:
            current_player.state.cash -= cash_transfer
            proposed_to_player.state.cash += cash_transfer

        self._update_sets()

        # Notify deal success
        current_player.call_ai(current_player.ai.deal_result, PlayerAIBase.DealInfo.SUCCEEDED)
        proposed_to_player.call_ai(proposed_to_player.ai.deal_result, PlayerAIBase.DealInfo.SUCCEEDED)

        # Build a deal_result-like object for deal_completed notifications
        class DealResult:
            pass
        dr = DealResult()
        dr.proposer = current_player
        dr.proposee = proposed_to_player
        dr.properties_offered = proposal.properties_offered
        dr.properties_wanted = proposal.properties_wanted
        dr.cash_transfer = cash_transfer

        for player in self.state.players:
            player.call_ai(player.ai.deal_completed, dr)
        self._notify_shadow_deal_completed(dr)

    # --- Override _make_deals to use our _make_deal ---

    def _make_deals(self, current_player):
        self._in_make_deals = True
        for i in range(3):
            self._make_deal(current_player)
        self._in_make_deals = False

    # --- Override _build_houses to intercept ---

    def _build_houses(self, current_player):
        if not current_player.state.owned_unmortgaged_sets:
            return

        build_instructions = current_player.call_ai(
            current_player.ai.build_houses, self.state, current_player)

        if self._is_watched(current_player) and current_player.state.owned_unmortgaged_sets:
            shadow_build = self._shadow.build_houses(self.state, current_player)
            ctx = describe_board_state(self.state, current_player)
            log_decision(
                "build_houses",
                self._turn_counter,
                current_player.ai.get_name(),
                ctx,
                describe_build(build_instructions),
                describe_build(shadow_build),
            )

        if not build_instructions:
            return

        # Validate and execute (replicate game.py)
        for (street, n) in build_instructions:
            if (n < 0) or (street.number_of_houses + n > 5):
                return

        self._build_houses_and_take_money(current_player, build_instructions)

        if current_player.state.cash < 0:
            self._roll_back_house_building(current_player, build_instructions)
            return

        for (street, n) in build_instructions:
            if n < 0 or street.number_of_houses > 5:
                self._roll_back_house_building(current_player, build_instructions)
                return
            if street.property_set not in current_player.state.owned_unmortgaged_sets:
                self._roll_back_house_building(current_player, build_instructions)
                return
            if not self._set_has_balanced_houses(street.property_set):
                self._roll_back_house_building(current_player, build_instructions)
                return

    # --- Override _offer_property_for_auction to intercept bids ---

    def _offer_property_for_auction(self, square):
        bids = []
        for player in self.state.players:
            bid = player.call_ai(
                player.ai.property_offered_for_auction,
                self.state, player, square)

            if self._is_watched(player):
                shadow_bid = self._shadow.property_offered_for_auction(
                    self.state, player, square)
                ctx = {
                    **describe_board_state(self.state, player),
                    "property": prop_name(square),
                    "property_price": square.price if hasattr(square, 'price') else None,
                }
                log_decision(
                    "auction_bid",
                    self._turn_counter,
                    player.ai.get_name(),
                    ctx,
                    bid,
                    shadow_bid,
                )

            if bid > 0:
                bids.append((player, bid))

        # Execute auction (replicate game.py)
        bids.sort(key=lambda x: x[1], reverse=True)

        status = PlayerAIBase.Action.AUCTION_FAILED
        player_who_won = None
        selling_price = 0

        for i in range(len(bids)):
            player_who_won = bids[i][0]
            next_bid = bids[i+1][1] if i+1 < len(bids) else 0
            selling_price = next_bid + 1

            self.take_money_from_player(player_who_won, selling_price)
            if player_who_won.state.cash < 0:
                self.give_money_to_player(player_who_won, selling_price)
            else:
                self.give_property_to_player(player_who_won, square.name)
                status = PlayerAIBase.Action.AUCTION_SUCCEEDED
                break

        for player in self.state.players:
            player.call_ai(player.ai.auction_result, status, square, player_who_won, selling_price)
        self._notify_shadow_auction_result(status, square, player_who_won, selling_price)

    # --- Override _check_for_bankrupt_players to notify shadow ---

    def _check_for_bankrupt_players(self):
        # Snapshot players before check
        before = set(self.state.players)
        super()._check_for_bankrupt_players()
        after = set(self.state.players)
        for gone in before - after:
            self._notify_shadow_bankrupt(gone)


# ---------------------------------------------------------------------------
# Run games and collect decisions
# ---------------------------------------------------------------------------
def run_comparison(num_games=50):
    """Run num_games 4-player games with top AIs, collect shadow comparisons."""

    # Pick the top 6 watched AIs + MarkovTradeAI's shadow observes
    watched_list = [ai for ai in all_ais if ai.get_name() in WATCH_AIS]
    if len(watched_list) < 4:
        print(f"ERROR: Only found {len(watched_list)} watched AIs")
        return

    import random
    for game_num in range(num_games):
        # Pick 4 random watched AIs per game
        players = random.sample(watched_list, min(4, len(watched_list)))

        game = CompareGame()
        game.set_shadow(shadow_ai)

        for ai in players:
            game.add_player(ai)

        game.play_game()

        if (game_num + 1) % 10 == 0:
            print(f"  Game {game_num + 1}/{num_games} done, {len(decisions)} decisions logged")

    return decisions


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def summarize(decisions):
    """Print a summary of decision agreement/disagreement."""

    by_type = defaultdict(list)
    for d in decisions:
        by_type[d["type"]].append(d)

    print(f"\n{'='*70}")
    print(f"DECISION COMPARISON SUMMARY — {len(decisions)} total decisions")
    print(f"{'='*70}\n")

    for dtype, dlist in sorted(by_type.items()):
        agree = 0
        disagree = 0
        for d in dlist:
            r = d["real"]
            s = d["shadow"]

            if dtype == "propose_deal":
                # Both propose or both pass?
                r_proposes = r is not None
                s_proposes = s is not None
                if r_proposes == s_proposes:
                    agree += 1
                else:
                    disagree += 1

            elif dtype == "deal_proposed":
                r_accept = isinstance(r, dict) and r.get("action") == "ACCEPT"
                s_accept = isinstance(s, dict) and s.get("action") == "ACCEPT"
                if r_accept == s_accept:
                    agree += 1
                else:
                    disagree += 1

            elif dtype == "auction_bid":
                # Within 20% of each other = agreement
                r_val = r if isinstance(r, (int, float)) else 0
                s_val = s if isinstance(s, (int, float)) else 0
                if r_val == 0 and s_val == 0:
                    agree += 1
                elif r_val == 0 or s_val == 0:
                    disagree += 1
                elif abs(r_val - s_val) / max(r_val, s_val) < 0.20:
                    agree += 1
                else:
                    disagree += 1

            elif dtype == "build_houses":
                # Both build or both don't?
                r_builds = len(r) > 0 if r else False
                s_builds = len(s) > 0 if s else False
                if r_builds == s_builds:
                    agree += 1
                else:
                    disagree += 1

        total = agree + disagree
        pct = 100 * agree / total if total else 0
        print(f"  {dtype:<20}  {total:>5} decisions  "
              f"{agree:>5} agree ({pct:5.1f}%)  {disagree:>5} disagree")

    # Detailed disagreements
    print(f"\n{'='*70}")
    print("NOTABLE DISAGREEMENTS (first 30)")
    print(f"{'='*70}\n")

    shown = 0
    for d in decisions:
        r, s = d["real"], d["shadow"]
        is_disagree = False

        if d["type"] == "propose_deal":
            is_disagree = (r is not None) != (s is not None)
        elif d["type"] == "deal_proposed":
            r_acc = isinstance(r, dict) and r.get("action") == "ACCEPT"
            s_acc = isinstance(s, dict) and s.get("action") == "ACCEPT"
            is_disagree = r_acc != s_acc
        elif d["type"] == "auction_bid":
            rv = r if isinstance(r, (int, float)) else 0
            sv = s if isinstance(s, (int, float)) else 0
            if rv == 0 and sv == 0:
                pass
            elif rv == 0 or sv == 0:
                is_disagree = True
            elif abs(rv - sv) / max(rv, sv) >= 0.20:
                is_disagree = True
        elif d["type"] == "build_houses":
            rb = len(r) > 0 if r else False
            sb = len(s) > 0 if s else False
            is_disagree = rb != sb

        if is_disagree and shown < 30:
            shown += 1
            print(f"  [{d['type']}] Turn {d['turn']}, {d['player']}")
            print(f"    State: cash={d['context'].get('cash')}, "
                  f"sets={d['context'].get('sets', [])}")
            if d["type"] == "auction_bid":
                print(f"    Property: {d['context'].get('property')}")
            if d["type"] in ("deal_proposed",):
                print(f"    From: {d['context'].get('from')}, "
                      f"offered={d['context'].get('offered')}, "
                      f"wanted={d['context'].get('wanted')}")
            print(f"    Real:   {r}")
            print(f"    Shadow: {s}")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    Logger.add_handler(ConsoleLogHandler(Logger.WARNING))

    print(f"Running {num_games} games for decision comparison...")
    print(f"Shadow AI: {SHADOW_NAME}")
    print(f"Watched AIs: {WATCH_AIS}")
    print()

    run_comparison(num_games)

    summarize(decisions)

    # Save raw data
    outfile = "decision_compare_results.json"
    with open(outfile, "w") as f:
        json.dump(decisions, f, indent=2, default=str)
    print(f"\nRaw data saved to {outfile} ({len(decisions)} records)")
