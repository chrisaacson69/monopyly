"""
MarkovTradeAI - Ported from Chris Isaacson's Monopoly AI engine.

Uses Markov-chain property valuation, bilateral trajectory simulation,
and Nash bargaining for trade evaluation.
"""

import copy
import logging
from monopyly import *
from monopyly.squares.street import Street
from monopyly.squares.station import Station
from monopyly.squares.utility import Utility

# Diagnostic logger — silent by default, enable externally
diag_log = logging.getLogger('markov_diag')

# ==========================================================================
# Constants
# ==========================================================================

# Steady-state landing probabilities (leave-jail strategy)
MARKOV_PROBS = {
    0: 0.031394, 1: 0.020543, 2: 0.019199, 3: 0.021062, 4: 0.023112,
    5: 0.029213, 6: 0.022712, 7: 0.008533, 8: 0.023489, 9: 0.023180,
    10: 0.0,     11: 0.026934, 12: 0.024483, 13: 0.023963, 14: 0.023562,
    15: 0.029808, 16: 0.027269, 17: 0.026840, 18: 0.028820, 19: 0.031808,
    20: 0.027986, 21: 0.028527, 22: 0.010032, 23: 0.027160, 24: 0.032123,
    25: 0.030263, 26: 0.027225, 27: 0.026247, 28: 0.027936, 29: 0.025225,
    30: 0.0,     31: 0.026318, 32: 0.027119, 33: 0.023184, 34: 0.025719,
    35: 0.023707, 36: 0.008625, 37: 0.020885, 38: 0.022180, 39: 0.025425,
}

DICE_EPT = 38  # ~35 Go salary + ~3 cards per turn
PROJECTION_HORIZON = 62
RAILROAD_RENT = {1: 25, 2: 50, 3: 100, 4: 200}
UTILITY_MULT = {1: 4, 2: 10}  # multiplied by avg dice roll of 7

STATION_POSITIONS = [5, 15, 25, 35]
UTILITY_POSITIONS = [12, 28]

# Board position -> PropertySet enum (streets only)
SET_POSITIONS = {
    PropertySet.BROWN:      [1, 3],
    PropertySet.LIGHT_BLUE: [6, 8, 9],
    PropertySet.PURPLE:     [11, 13, 14],
    PropertySet.ORANGE:     [16, 18, 19],
    PropertySet.RED:        [21, 23, 24],
    PropertySet.YELLOW:     [26, 27, 29],
    PropertySet.GREEN:      [31, 32, 34],
    PropertySet.DARK_BLUE:  [37, 39],
}

# CAPM-derived monopoly quality (ROI at 3 houses, 3 opponents)
# Orange=10.26%, Red=9.32%, DarkBlue=9.01%, Yellow=8.96%,
# Green=8.14%, Purple=7.77%, LightBlue=7.57%, Brown=4.03%
GROUP_QUALITY = {
    PropertySet.ORANGE:     1.30,
    PropertySet.RED:        1.18,
    PropertySet.DARK_BLUE:  1.14,
    PropertySet.YELLOW:     1.13,
    PropertySet.GREEN:      1.03,
    PropertySet.PURPLE:     0.98,
    PropertySet.LIGHT_BLUE: 0.96,
    PropertySet.BROWN:      0.51,
}

# Security Market Line — full investment profile per color group
# ROI = (ply_EPT@3h × 3 opponents) / (acquisition + development cost)
# Alpha = ROI - market average (0.0813)
# ept_ply_3h = per-ply EPT at 3 houses (Markov × rent@3h, summed over set)
MARKET_AVG_ROI = 0.0813
SML = {
    PropertySet.BROWN: {
        'size': 2, 'roi': 0.0403, 'alpha': -0.0410, 'quality': 0.51,
        'acq_cost': 120, 'dev_cost': 300, 'total_cost': 420,
        'ept_ply_3h': 5.64, 'house_price': 50,
    },
    PropertySet.LIGHT_BLUE: {
        'size': 3, 'roi': 0.0757, 'alpha': -0.0056, 'quality': 0.96,
        'acq_cost': 320, 'dev_cost': 450, 'total_cost': 770,
        'ept_ply_3h': 19.43, 'house_price': 50,
    },
    PropertySet.PURPLE: {
        'size': 3, 'roi': 0.0776, 'alpha': -0.0037, 'quality': 0.98,
        'acq_cost': 440, 'dev_cost': 900, 'total_cost': 1340,
        'ept_ply_3h': 34.68, 'house_price': 100,
    },
    PropertySet.ORANGE: {
        'size': 3, 'roi': 0.1026, 'alpha': 0.0213, 'quality': 1.30,
        'acq_cost': 560, 'dev_cost': 900, 'total_cost': 1460,
        'ept_ply_3h': 49.93, 'house_price': 100,
    },
    PropertySet.RED: {
        'size': 3, 'roi': 0.0932, 'alpha': 0.0119, 'quality': 1.18,
        'acq_cost': 680, 'dev_cost': 1350, 'total_cost': 2030,
        'ept_ply_3h': 63.07, 'house_price': 150,
    },
    PropertySet.YELLOW: {
        'size': 3, 'roi': 0.0896, 'alpha': 0.0083, 'quality': 1.13,
        'acq_cost': 800, 'dev_cost': 1350, 'total_cost': 2150,
        'ept_ply_3h': 64.22, 'house_price': 150,
    },
    PropertySet.GREEN: {
        'size': 3, 'roi': 0.0814, 'alpha': 0.0001, 'quality': 1.03,
        'acq_cost': 920, 'dev_cost': 1800, 'total_cost': 2720,
        'ept_ply_3h': 73.81, 'house_price': 200,
    },
    PropertySet.DARK_BLUE: {
        'size': 2, 'roi': 0.0901, 'alpha': 0.0088, 'quality': 1.14,
        'acq_cost': 750, 'dev_cost': 1200, 'total_cost': 1950,
        'ept_ply_3h': 58.57, 'house_price': 200,
    },
}

ABSOLUTE_MIN_CASH = 75

# Development Frontier: EPT (×3 opponents) at each house level, and marginal cost
# Level 0 = monopoly with 0 houses (2× base rent). Levels 1-5 = houses/hotel.
# cost[level] = house_price × set_size (cash to advance FROM level-1 TO level)
# Derived from MARKOV_PROBS × rent tables. Positions match SET_POSITIONS.
DEV_FRONTIER = {
    PropertySet.BROWN:      {'ept': [0.75, 1.88, 5.64, 16.92, 30.08, 43.84],
                             'cost': [0, 100, 100, 100, 100, 100]},
    PropertySet.LIGHT_BLUE: {'ept': [2.78, 6.94, 19.43, 58.28, 86.73, 117.96],
                             'cost': [0, 150, 150, 150, 150, 150]},
    PropertySet.PURPLE:     {'ept': [4.75, 11.88, 35.63, 104.05, 144.91, 178.14],
                             'cost': [0, 300, 300, 300, 300, 300]},
    PropertySet.ORANGE:     {'ept': [7.77, 19.41, 54.65, 149.80, 202.54, 255.28],
                             'cost': [0, 300, 300, 300, 300, 300]},
    PropertySet.RED:        {'ept': [9.87, 24.67, 70.68, 189.22, 235.32, 281.42],
                             'cost': [0, 450, 450, 450, 450, 450]},
    PropertySet.YELLOW:     {'ept': [10.39, 26.73, 80.18, 192.66, 233.97, 275.29],
                             'cost': [0, 450, 450, 450, 450, 450]},
    PropertySet.GREEN:      {'ept': [12.66, 32.41, 97.24, 221.44, 268.93, 312.42],
                             'cost': [0, 600, 600, 600, 600, 600]},
    PropertySet.DARK_BLUE:  {'ept': [12.01, 26.22, 77.09, 175.71, 211.12, 246.53],
                             'cost': [0, 400, 400, 400, 400, 400]},
}

# Base rents by position (for singleton EPT — not in a monopoly)
# position → base rent from property card
_BASE_RENT = {
    1: 2, 3: 4,                      # Brown
    6: 6, 8: 6, 9: 8,                # Light Blue
    11: 10, 13: 10, 14: 12,          # Purple
    16: 14, 18: 14, 19: 16,          # Orange
    21: 18, 23: 18, 24: 20,          # Red
    26: 22, 27: 22, 29: 22,          # Yellow
    31: 26, 32: 26, 34: 28,          # Green
    37: 35, 39: 50,                   # Dark Blue
}
# Pre-multiply by Markov probability and 3 opponents for quick lookup
BASE_EPT = {pos: MARKOV_PROBS.get(pos, 0) * rent * 3
            for pos, rent in _BASE_RENT.items()}


# ==========================================================================
# Bilateral Trajectory Simulation Engine
# ==========================================================================

def _compute_ept(sp, opponents):
    """Expected per-turn rental income from all property types."""
    total = 0.0
    for g in sp['groups']:
        for i, pos in enumerate(g['positions']):
            h = g['houses'][i]
            prob = MARKOV_PROBS.get(pos, 0.025)
            rent = g['rents'][i][0] * 2 if h == 0 else g['rents'][i][h]
            total += prob * rent * opponents
    rr = sp['rr_count']
    if rr > 0:
        rr_rent = RAILROAD_RENT[rr]
        for pos in sp['rr_positions']:
            total += MARKOV_PROBS.get(pos, 0.025) * rr_rent * opponents
    uc = sp['util_count']
    if uc > 0:
        u_rent = UTILITY_MULT[uc] * 7
        for pos in sp['util_positions']:
            total += MARKOV_PROBS.get(pos, 0.025) * u_rent * opponents
    return total


def _compute_position(sp):
    """Total net worth: cash + property values + house investment."""
    val = sp['cash']
    for g in sp['groups']:
        for i in range(len(g['positions'])):
            val += g['prices'][i]
            val += g['houses'][i] * g['house_price']
    val += len(sp['rr_positions']) * 200
    val += len(sp['util_positions']) * 150
    return val


def _try_build(sp):
    """Build one house with best marginal ROI. Returns True if built."""
    best_roi = 0
    best_g = None
    best_i = -1
    for g in sp['groups']:
        if not g['houses']:
            continue
        min_h = min(g['houses'])
        hp = g['house_price']
        if hp <= 0 or hp > sp['cash']:
            continue
        for i, pos in enumerate(g['positions']):
            h = g['houses'][i]
            if h >= 5 or h > min_h:
                continue
            prob = MARKOV_PROBS.get(pos, 0.025)
            cur = g['rents'][i][0] * 2 if h == 0 else g['rents'][i][h]
            nxt = g['rents'][i][h + 1]
            roi = prob * (nxt - cur) / hp
            if roi > best_roi:
                best_roi = roi
                best_g = g
                best_i = i
    if best_g is not None and best_roi > 0:
        sp['cash'] -= best_g['house_price']
        best_g['houses'][best_i] += 1
        return True
    return False


def _try_sell_house(sp):
    """Sell one house with worst marginal ROI. Returns True if sold."""
    worst_roi = float('inf')
    worst_g = None
    worst_i = -1
    for g in sp['groups']:
        if not g['houses'] or max(g['houses']) <= 0:
            continue
        max_h = max(g['houses'])
        hp = g['house_price']
        if hp <= 0:
            continue
        for i, pos in enumerate(g['positions']):
            h = g['houses'][i]
            if h <= 0 or h < max_h:
                continue
            prob = MARKOV_PROBS.get(pos, 0.025)
            cur = g['rents'][i][h]
            prev = g['rents'][i][0] * 2 if h == 1 else g['rents'][i][h - 1]
            roi = prob * (cur - prev) / hp
            if roi < worst_roi:
                worst_roi = roi
                worst_g = g
                worst_i = i
    if worst_g is not None:
        sp['cash'] += worst_g['house_price'] // 2
        worst_g['houses'][worst_i] -= 1
        return True
    return False


def _bilateral_sim(me_state, them_state, num_others):
    """Simulate 62 turns of bilateral growth.

    Returns (my_trajectory, their_trajectory) as lists of position values.
    """
    me = copy.deepcopy(me_state)
    them = copy.deepcopy(them_state)
    my_opp = 1 + num_others
    their_opp = 1 + num_others

    my_traj = [_compute_position(me)]
    their_traj = [_compute_position(them)]

    for _ in range(PROJECTION_HORIZON):
        my_ept = _compute_ept(me, my_opp)
        their_ept = _compute_ept(them, their_opp)

        me['cash'] += DICE_EPT + my_ept - their_ept / my_opp
        them['cash'] += DICE_EPT + their_ept - my_ept / their_opp

        while me['cash'] < 0 and _try_sell_house(me):
            pass
        me['cash'] = max(0, me['cash'])
        while them['cash'] < 0 and _try_sell_house(them):
            pass
        them['cash'] = max(0, them['cash'])

        while _try_build(me):
            pass
        while _try_build(them):
            pass

        my_traj.append(_compute_position(me))
        their_traj.append(_compute_position(them))

    return my_traj, their_traj


# ==========================================================================
# MarkovTradeAI
# ==========================================================================

class MarkovTradeAI(PlayerAIBase):

    def __init__(self):
        self._pending_debt = 0
        self._proposed_deals = set()  # keys of deals we've already proposed
        self._deals_this_turn = 0
        self._current_turn = -1
        # Diagnostics
        self._diag = {
            'ever_held_monopoly': False,
            'monopoly_sets': set(),
            'turn_first_monopoly': None,
            'deals_we_proposed': 0,
            'deals_we_accepted': 0,
            'deals_we_rejected': 0,
            'deals_others_completed': 0,
            'deals_we_completed': 0,
            'other_monopoly_deals': [],
            'our_monopoly_deals': [],
            'turn': 0,
            # deal_result tracking (as proposer)
            'propose_results': {},  # DealInfo name -> count
            # deal_result tracking (as proposee)
            'respond_results': {},
            # rejection reasons when WE reject incoming deals
            'our_reject_reasons': {},  # reason -> count
            # who we propose to
            'propose_targets': {},  # AI name -> count
            # deal structure tracking
            'propose_types': {'mutual_monopoly': 0, 'setup': 0},
            # cash terms when we propose
            'propose_cash_terms': [],  # list of (fc, offered/asked)
            # cash terms when we accept incoming
            'accept_cash_terms': [],  # list of (fc, offered/asked)
        }
        self._last_proposed_to = None  # track if we were proposer or proposee

    def get_name(self):
        return "MarkovTradeAI"

    def start_of_game(self):
        self._pending_debt = 0
        self._proposed_deals = set()
        self._deals_this_turn = 0
        self._current_turn = -1
        self._diag = {
            'ever_held_monopoly': False,
            'monopoly_sets': set(),
            'turn_first_monopoly': None,
            'deals_we_proposed': 0,
            'deals_we_accepted': 0,
            'deals_we_rejected': 0,
            'deals_others_completed': 0,
            'deals_we_completed': 0,
            'other_monopoly_deals': [],
            'our_monopoly_deals': [],
            'turn': 0,
            'propose_results': {},
            'respond_results': {},
            'our_reject_reasons': {},
            'propose_targets': {},
            'propose_types': {'mutual_monopoly': 0, 'setup': 0},
            'propose_cash_terms': [],
            'accept_cash_terms': [],
        }
        self._last_proposed_to = None

    def start_of_turn(self, game_state, player):
        self._diag['turn'] += 1
        # Check if we currently hold any monopolies
        if player.state.owned_unmortgaged_sets:
            for ps in player.state.owned_unmortgaged_sets:
                se = ps.set_enum
                if se not in (PropertySet.STATION, PropertySet.UTILITY):
                    if not self._diag['ever_held_monopoly']:
                        self._diag['ever_held_monopoly'] = True
                        self._diag['turn_first_monopoly'] = self._diag['turn']
                    self._diag['monopoly_sets'].add(se)

    def deal_completed(self, deal_result):
        """Board state changed — clear proposed deals to re-evaluate.
        Also track all completed deals for diagnostics."""
        self._proposed_deals = set()

        # Track deal diagnostics
        proposer = deal_result.proposer
        proposee = deal_result.proposee
        props_to_proposer = deal_result.properties_transferred_to_proposer
        props_to_proposee = deal_result.properties_transferred_to_proposee
        we_involved = (proposer.ai.get_name() == "MarkovTradeAI" or
                       proposee.ai.get_name() == "MarkovTradeAI")

        if we_involved:
            self._diag['deals_we_completed'] += 1
        else:
            self._diag['deals_others_completed'] += 1

        # Check if any monopolies were completed by this deal
        all_transferred = list(props_to_proposer) + list(props_to_proposee)
        sets_completed = []
        for prop in all_transferred:
            if isinstance(prop, Street):
                se = prop.property_set.set_enum
                ps = prop.property_set
                if ps.owner is not None:
                    sets_completed.append((ps.owner.ai.get_name(), str(se)))

        if sets_completed:
            entry = {
                'turn': self._diag['turn'],
                'proposer': proposer.ai.get_name(),
                'proposee': proposee.ai.get_name(),
                'sets': sets_completed,
            }
            if we_involved:
                self._diag['our_monopoly_deals'].append(entry)
            else:
                self._diag['other_monopoly_deals'].append(entry)

    def game_over(self, winner, maximum_rounds_played):
        d = self._diag
        won = winner is not None and winner.ai.get_name() == "MarkovTradeAI"
        winner_name = winner.ai.get_name() if winner else "DRAW"

        diag_log.info("=" * 60)
        diag_log.info(f"GAME OVER: {'WIN' if won else 'LOSS'} (winner: {winner_name})")
        diag_log.info(f"  Turns played: {d['turn']}, max rounds: {maximum_rounds_played}")
        diag_log.info(f"  Ever held monopoly: {d['ever_held_monopoly']}")
        if d['ever_held_monopoly']:
            diag_log.info(f"    First monopoly turn: {d['turn_first_monopoly']}")
            diag_log.info(f"    Sets held: {[str(s) for s in d['monopoly_sets']]}")
        diag_log.info(f"  Deals WE completed: {d['deals_we_completed']}")
        diag_log.info(f"  Deals OTHERS completed (without us): {d['deals_others_completed']}")
        if d['our_monopoly_deals']:
            diag_log.info(f"  Our monopoly deals:")
            for e in d['our_monopoly_deals']:
                diag_log.info(f"    Turn {e['turn']}: {e['proposer']} <-> {e['proposee']} => {e['sets']}")
        if d['other_monopoly_deals']:
            diag_log.info(f"  Other monopoly deals:")
            for e in d['other_monopoly_deals']:
                diag_log.info(f"    Turn {e['turn']}: {e['proposer']} <-> {e['proposee']} => {e['sets']}")
        diag_log.info(f"  --- Proposal outcomes (as proposer) ---")
        for k, v in sorted(d['propose_results'].items(), key=lambda x: -x[1]):
            diag_log.info(f"    {k}: {v}")
        diag_log.info(f"  --- Response outcomes (as proposee) ---")
        for k, v in sorted(d['respond_results'].items(), key=lambda x: -x[1]):
            diag_log.info(f"    {k}: {v}")
        diag_log.info(f"  --- Our rejection reasons ---")
        for k, v in sorted(d['our_reject_reasons'].items(), key=lambda x: -x[1]):
            diag_log.info(f"    {k}: {v}")
        diag_log.info(f"  --- Propose targets ---")
        for k, v in sorted(d['propose_targets'].items(), key=lambda x: -x[1]):
            diag_log.info(f"    {k}: {v}")
        diag_log.info(f"  --- Propose types ---")
        for k, v in d['propose_types'].items():
            diag_log.info(f"    {k}: {v}")
        diag_log.info("")

    def deal_result(self, deal_info):
        """Track outcome of deals we're involved in."""
        info_names = {
            0: 'SUCCEEDED', 1: 'INVALID', 2: 'ASKED_TOO_MUCH',
            3: 'OFFERED_TOO_LITTLE', 4: 'NOT_ENOUGH_MONEY', 5: 'REJECTED',
        }
        name = info_names.get(deal_info, str(deal_info))

        if self._last_proposed_to is not None:
            # We were the proposer
            d = self._diag['propose_results']
            d[name] = d.get(name, 0) + 1
            self._last_proposed_to = None
        else:
            # We were the proposee
            d = self._diag['respond_results']
            d[name] = d.get(name, 0) + 1

    # ------------------------------------------------------------------
    # Board state helpers
    # ------------------------------------------------------------------

    def _snapshot(self, board):
        """Lightweight snapshot of property ownership and development."""
        snap = {}
        for idx in range(40):
            sq = board.squares[idx]
            if isinstance(sq, Street):
                snap[idx] = {
                    'owner': sq.owner,
                    'houses': sq.number_of_houses,
                    'mortgaged': sq.is_mortgaged,
                    'price': sq.price,
                    'rents': sq.rents,
                    'house_price': sq.house_price,
                }
            elif isinstance(sq, (Station, Utility)):
                snap[idx] = {
                    'owner': sq.owner,
                    'mortgaged': sq.is_mortgaged,
                    'price': sq.price,
                }
        return snap

    def _get_monopolies(self, snap, player):
        """Street groups where player owns all properties (unmortgaged)."""
        result = []
        for se, positions in SET_POSITIONS.items():
            if all(pos in snap and snap[pos]['owner'] is player
                   for pos in positions):
                result.append(se)
        return result

    # ------------------------------------------------------------------
    # EPT Strategic Planner
    # ------------------------------------------------------------------

    def _get_monopoly_levels(self, player):
        """Return {PropertySet: min_house_level} for owned unmortgaged monopolies."""
        levels = {}
        for pset in player.state.owned_unmortgaged_sets:
            if pset not in DEV_FRONTIER:
                continue  # skip stations/utilities
            min_houses = min(p.number_of_houses for p in pset.properties)
            levels[pset] = min_houses
        return levels

    def _achievable_ept(self, cash, monopoly_levels):
        """Given cash budget and {PropertySet: current_level}, return maximum
        achievable total EPT through optimal development spending.

        Uses greedy priority queue: always build the highest marginal-ROI
        step available. Respects sequential building (must do 1h before 2h).
        """
        import heapq

        # Current EPT at existing development levels
        total_ept = 0.0
        for pset, level in monopoly_levels.items():
            total_ept += DEV_FRONTIER[pset]['ept'][level]

        # Priority queue: next available step per monopoly, sorted by ROI
        heap = []
        for pset, level in monopoly_levels.items():
            if level < 5:
                nl = level + 1
                cost = DEV_FRONTIER[pset]['cost'][nl]
                marg = DEV_FRONTIER[pset]['ept'][nl] - DEV_FRONTIER[pset]['ept'][level]
                if cost > 0:
                    heapq.heappush(heap, (-marg / cost, cost, marg, str(pset), pset, nl))

        remaining = cash
        while heap:
            _, cost, marg, _, pset, nl = heapq.heappop(heap)
            if cost > remaining:
                continue
            remaining -= cost
            total_ept += marg
            if nl < 5:
                nnl = nl + 1
                ncost = DEV_FRONTIER[pset]['cost'][nnl]
                nmarg = DEV_FRONTIER[pset]['ept'][nnl] - DEV_FRONTIER[pset]['ept'][nl]
                if ncost > 0:
                    heapq.heappush(heap, (-nmarg / ncost, ncost, nmarg, str(pset), pset, nnl))

        return total_ept

    def _ept_of_positions(self, positions):
        """Sum of base EPT for a list of board positions (singleton rent, not monopoly)."""
        return sum(BASE_EPT.get(pos, 0) for pos in positions)

    # ------------------------------------------------------------------
    # Frontier Ownership Graph
    # ------------------------------------------------------------------

    def _build_ownership_graph(self, snap, players):
        """Build ownership graph from board snapshot.

        Returns dict: {PropertySet → {
            'size': int,
            'by_player': {player → [positions]},
            'counts': {player → int},
            'unowned': [positions],
            'all_sold': bool,
            'completable_by': {player → {
                'owned': [positions],
                'needed': [positions],
                'needed_from': {player → [positions]},
                'distance': int,
                'alpha': float,
            }},
        }}
        """
        graph = {}
        for se, positions in SET_POSITIONS.items():
            by_player = {}
            unowned = []
            for pos in positions:
                owner = snap.get(pos, {}).get('owner')
                if owner is not None:
                    if owner not in by_player:
                        by_player[owner] = []
                    by_player[owner].append(pos)
                else:
                    unowned.append(pos)

            counts = {p: len(ps) for p, ps in by_player.items()}
            all_sold = len(unowned) == 0
            alpha = SML[se]['alpha']

            # Completion analysis: who could complete this set via trades?
            completable_by = {}
            if all_sold:
                for p, owned in by_player.items():
                    if len(owned) < len(positions):
                        needed = [pos for pos in positions if pos not in owned]
                        needed_from = {}
                        for npos in needed:
                            holder = snap[npos]['owner']
                            if holder not in needed_from:
                                needed_from[holder] = []
                            needed_from[holder].append(npos)
                        completable_by[p] = {
                            'owned': owned,
                            'needed': needed,
                            'needed_from': needed_from,
                            'distance': len(needed),
                            'alpha': alpha,
                        }
                    elif len(owned) == len(positions):
                        # Already complete
                        completable_by[p] = {
                            'owned': owned,
                            'needed': [],
                            'needed_from': {},
                            'distance': 0,
                            'alpha': alpha,
                        }

            graph[se] = {
                'size': len(positions),
                'by_player': by_player,
                'counts': counts,
                'unowned': unowned,
                'all_sold': all_sold,
                'completable_by': completable_by,
            }
        return graph

    def _find_coalitions(self, graph, me, players):
        """Find effective coalitions between me and each opponent.

        Returns list of coalition dicts sorted by my_alpha descending:
        {
            'partner': Player,
            'my_completions': [{set_id, alpha, positions_acquired}],
            'their_completions': [{set_id, alpha, positions_acquired}],
            'my_alpha': float,
            'their_alpha': float,
            'combined_alpha': float,
            'i_give': [positions],
            'they_give': [positions],
            'type': 'mutual' | 'one_sided' | 'setup',
        }
        """
        coalitions = []
        opponents = [p for p in players if p is not me]

        for opp in opponents:
            # What can I complete if opp gives me pieces?
            my_possible = []
            for se, info in graph.items():
                cb = info['completable_by']
                if me in cb and cb[me]['distance'] > 0:
                    nf = cb[me]['needed_from']
                    if opp in nf:
                        my_possible.append({
                            'set_id': se,
                            'alpha': SML[se]['alpha'],
                            'positions_acquired': nf[opp],
                            'distance': cb[me]['distance'],
                            'fully_from_opp': len(nf[opp]) == cb[me]['distance'],
                        })

            # What can opp complete if I give them pieces?
            their_possible = []
            for se, info in graph.items():
                cb = info['completable_by']
                if opp in cb and cb[opp]['distance'] > 0:
                    nf = cb[opp]['needed_from']
                    if me in nf:
                        their_possible.append({
                            'set_id': se,
                            'alpha': SML[se]['alpha'],
                            'positions_acquired': nf[me],
                            'distance': cb[opp]['distance'],
                            'fully_from_me': len(nf[me]) == cb[opp]['distance'],
                        })

            # Build coalition candidates
            # Case 1: Mutual completion — I complete a set, they complete a set
            for mc in my_possible:
                if not mc['fully_from_opp']:
                    continue
                if mc['alpha'] < -0.01:
                    continue  # never complete deeply negative-alpha sets
                for tc in their_possible:
                    if not tc['fully_from_me']:
                        continue
                    if mc['set_id'] == tc['set_id']:
                        continue
                    coalitions.append({
                        'partner': opp,
                        'my_completions': [mc],
                        'their_completions': [tc],
                        'my_alpha': mc['alpha'],
                        'their_alpha': tc['alpha'],
                        'combined_alpha': mc['alpha'] + tc['alpha'],
                        'i_give': list(tc['positions_acquired']),
                        'they_give': list(mc['positions_acquired']),
                        'type': 'mutual',
                    })

            # Case 2: One-sided — I complete, give them expendable property
            for mc in my_possible:
                if not mc['fully_from_opp']:
                    continue
                if mc['alpha'] < -0.01:
                    continue  # never complete deeply negative-alpha sets
                # Find what I can give in return (non-critical singletons)
                for se, info in graph.items():
                    if se == mc['set_id']:
                        continue
                    my_in_set = info['by_player'].get(me, [])
                    if len(my_in_set) != 1:
                        continue
                    # Don't give away sole blockers of positive-alpha sets
                    opp_in_set = info['counts'].get(opp, 0)
                    if opp_in_set >= info['size'] - 1 and SML[se]['alpha'] > 0:
                        continue
                    coalitions.append({
                        'partner': opp,
                        'my_completions': [mc],
                        'their_completions': [{
                            'set_id': se,
                            'alpha': SML[se]['alpha'],
                            'positions_acquired': my_in_set,
                            'distance': info['size'] - info['counts'].get(opp, 0) - 1,
                            'fully_from_me': True,
                        }],
                        'my_alpha': mc['alpha'],
                        'their_alpha': 0.0,  # they don't complete
                        'combined_alpha': mc['alpha'],
                        'i_give': list(my_in_set),
                        'they_give': list(mc['positions_acquired']),
                        'type': 'one_sided',
                    })

            # Case 3: Setup — neither completes, but I advance toward
            # a positive-alpha set
            for se, info in graph.items():
                if SML[se]['alpha'] <= 0:
                    continue
                if not info['all_sold']:
                    continue
                my_count = info['counts'].get(me, 0)
                opp_count = info['counts'].get(opp, 0)
                if my_count == 0 or opp_count == 0:
                    continue
                if my_count + opp_count >= info['size']:
                    continue  # already covered by completion cases
                # I want one of opp's pieces in this set
                opp_pieces = info['by_player'].get(opp, [])
                # Find expendable singleton to give in return
                for se2, info2 in graph.items():
                    if se2 == se:
                        continue
                    my_in_se2 = info2['by_player'].get(me, [])
                    if len(my_in_se2) != 1:
                        continue
                    # Sole blocker check for positive-alpha sets
                    opp_in_se2 = info2['counts'].get(opp, 0)
                    if opp_in_se2 >= info2['size'] - 1 and SML[se2]['alpha'] > 0:
                        continue
                    # Also check all opponents for sole blocker
                    any_sole_block = False
                    for op2 in opponents:
                        if op2 is opp:
                            continue
                        op2_ct = info2['counts'].get(op2, 0)
                        if op2_ct >= info2['size'] - 1 and SML[se2]['alpha'] > 0:
                            any_sole_block = True
                            break
                    if any_sole_block:
                        continue

                    coalitions.append({
                        'partner': opp,
                        'my_completions': [{
                            'set_id': se,
                            'alpha': SML[se]['alpha'],
                            'positions_acquired': [opp_pieces[0]],
                            'distance': info['size'] - my_count - 1,
                            'fully_from_opp': False,
                        }],
                        'their_completions': [{
                            'set_id': se2,
                            'alpha': SML[se2]['alpha'],
                            'positions_acquired': my_in_se2,
                            'distance': info2['size'] - info2['counts'].get(opp, 0) - 1,
                            'fully_from_me': True,
                        }],
                        'my_alpha': SML[se]['alpha'] * (my_count + 1) / info['size'],
                        'their_alpha': 0.0,
                        'combined_alpha': SML[se]['alpha'] * (my_count + 1) / info['size'],
                        'i_give': list(my_in_se2),
                        'they_give': [opp_pieces[0]],
                        'type': 'setup',
                    })

        # Sort: mutual first, then by my_alpha descending
        type_order = {'mutual': 0, 'one_sided': 1, 'setup': 2}
        coalitions.sort(key=lambda c: (type_order[c['type']], -c['my_alpha']))
        return coalitions

    def _compute_property_values(self, graph, coalitions, me):
        """Compute effective value of each property owned by me.

        Returns dict: {position → {
            'set_id': PropertySet,
            'current_ept': float,
            'coalition_value': float,
            'blocking_value': float,
            'effective_value': float,
        }}
        """
        values = {}

        # Gather all properties I own
        for se, info in graph.items():
            for pos in info['by_player'].get(me, []):
                values[pos] = {
                    'set_id': se,
                    'current_ept': 0.0,
                    'coalition_value': 0.0,
                    'blocking_value': 0.0,
                    'effective_value': 0.0,
                }

        # Component 1: Current income EPT (base rent, no monopoly bonus)
        # Stored as mortgage_value as minimum floor
        for pos, val in values.items():
            se = val['set_id']
            price = SML[se]['acq_cost'] / SML[se]['size']
            val['current_ept'] = MARKOV_PROBS.get(pos, 0) * price * 0.05
            # Floor: mortgage value (half purchase price)
            val['mortgage_value'] = price / 2

        # Component 2: Coalition contribution
        # How many coalitions does this property participate in?
        for c in coalitions:
            for pos in c['i_give']:
                if pos in values:
                    # This property is a trade chip — its value =
                    # what it enables for me (my_alpha)
                    values[pos]['coalition_value'] = max(
                        values[pos]['coalition_value'],
                        c['my_alpha'] * SML[values[pos]['set_id']]['total_cost']
                    )
            for pos in c['they_give']:
                # These are properties I'd receive — not owned yet
                pass

        # Component 3: Blocking value
        for se, info in graph.items():
            my_pieces = info['by_player'].get(me, [])
            if not my_pieces:
                continue
            alpha = SML[se]['alpha']
            if alpha <= 0:
                continue  # blocking negative-alpha sets is worthless

            # Check each opponent's completion status
            for opp, opp_pieces in info['by_player'].items():
                if opp is me:
                    continue
                opp_needs = info['size'] - len(opp_pieces)
                if opp_needs == 0:
                    continue  # already complete
                # How many of the needed pieces do I hold?
                my_blocking = [p for p in my_pieces
                               if p not in opp_pieces]
                if not my_blocking:
                    continue
                # Am I the sole blocker? (only I hold pieces they need)
                sole = True
                for other_p, other_pieces in info['by_player'].items():
                    if other_p is me or other_p is opp:
                        continue
                    if any(p not in opp_pieces for p in other_pieces):
                        sole = False
                        break
                block_mult = 2.0 if sole else 0.5
                block_val = alpha * SML[se]['total_cost'] * block_mult
                for pos in my_blocking:
                    if pos in values:
                        values[pos]['blocking_value'] = max(
                            values[pos]['blocking_value'], block_val)

        # Component 4: Progress value for partially-sold sets
        # Properties in sets where not all are sold still have value
        # as a head start on future monopoly completion
        for se, info in graph.items():
            alpha = SML[se]['alpha']
            if alpha <= 0:
                continue
            my_pieces = info['by_player'].get(me, [])
            if not my_pieces:
                continue
            if info['all_sold']:
                continue  # handled by coalition/blocking above
            # I have pieces in a partially-sold positive-alpha set
            # Value = progress toward completion × alpha × cost
            progress = len(my_pieces) / info['size']
            progress_val = alpha * SML[se]['total_cost'] * progress * 2
            for pos in my_pieces:
                if pos in values:
                    values[pos]['progress_value'] = max(
                        values[pos].get('progress_value', 0), progress_val)

        # Combine
        for pos, val in values.items():
            val['effective_value'] = (
                val.get('mortgage_value', 0)
                + val['coalition_value']
                + val['blocking_value']
                + val.get('progress_value', 0)
            )

        return values

    def _sim_from_snap(self, snap, board, player, monops, cash):
        """Build bilateral sim state dict from a board snapshot."""
        groups = []
        for se in monops:
            positions = SET_POSITIONS[se]
            groups.append({
                'positions': positions,
                'house_price': board.get_property_set(se).house_price,
                'houses': [snap[pos].get('houses', 0) for pos in positions],
                'rents': [snap[pos]['rents'] for pos in positions],
                'prices': [snap[pos]['price'] for pos in positions],
            })
        rr = [p for p in STATION_POSITIONS
              if p in snap and snap[p]['owner'] is player
              and not snap[p].get('mortgaged', False)]
        ut = [p for p in UTILITY_POSITIONS
              if p in snap and snap[p]['owner'] is player
              and not snap[p].get('mortgaged', False)]
        return {
            'groups': groups,
            'rr_count': len(rr), 'util_count': len(ut),
            'rr_positions': rr, 'util_positions': ut,
            'cash': cash,
        }

    def _apply_trade(self, snap, me, them, my_give_pos, their_give_pos):
        """Create post-trade snapshot with ownership changes."""
        post = {k: dict(v) for k, v in snap.items()}
        for pos in my_give_pos:
            post[pos] = dict(snap[pos])
            post[pos]['owner'] = them
            if 'houses' in post[pos]:
                post[pos]['houses'] = 0
        for pos in their_give_pos:
            post[pos] = dict(snap[pos])
            post[pos]['owner'] = me
            if 'houses' in post[pos]:
                post[pos]['houses'] = 0
        return post

    # ------------------------------------------------------------------
    # Trade engine
    # ------------------------------------------------------------------

    def _find_trade_candidates(self, snap, me, them):
        """Find 1-for-1 monopoly completion swaps.

        Priority 1: Mutual completion — both sides complete a monopoly.
        Priority 2: One-sided — we complete, they get a useful property.
        All candidates are exactly 1-for-1 property exchanges.
        """
        # Sets where I need exactly 1 from them to complete
        my_completable = []  # (set_enum, their_position)
        for se, positions in SET_POSITIONS.items():
            my_count = sum(1 for p in positions
                          if snap.get(p, {}).get('owner') is me)
            their_count = sum(1 for p in positions
                             if snap.get(p, {}).get('owner') is them)
            if my_count + their_count == len(positions) and their_count == 1:
                pos = next(p for p in positions
                           if snap.get(p, {}).get('owner') is them)
                my_completable.append((se, pos))

        # Sets where they need exactly 1 from me to complete
        their_completable = []  # (set_enum, my_position)
        for se, positions in SET_POSITIONS.items():
            my_count = sum(1 for p in positions
                          if snap.get(p, {}).get('owner') is me)
            their_count = sum(1 for p in positions
                             if snap.get(p, {}).get('owner') is them)
            if my_count + their_count == len(positions) and my_count == 1:
                pos = next(p for p in positions
                           if snap.get(p, {}).get('owner') is me)
                their_completable.append((se, pos))

        candidates = []

        # PRIORITY 1: Mutual completion (1-for-1, both complete)
        for sa, want_pos in my_completable:
            for sb, give_pos in their_completable:
                if sa == sb:
                    continue
                candidates.append({
                    'i_want': [want_pos], 'i_give': [give_pos],
                    'my_set': sa, 'their_set': sb,
                })

        # PRIORITY 2: One-sided completion — I complete, give them a singleton
        # Only if no mutual candidates found with this opponent
        if not candidates:
            # Find singletons I can offer (sets I own exactly 1 of, can't self-complete)
            my_singletons = []
            for se, positions in SET_POSITIONS.items():
                my_count = sum(1 for p in positions
                              if snap.get(p, {}).get('owner') is me)
                their_count = sum(1 for p in positions
                                 if snap.get(p, {}).get('owner') is them)
                if my_count != 1:
                    continue
                # Don't offer if we could still self-complete this set
                unowned = sum(1 for p in positions
                              if snap.get(p, {}).get('owner') is None)
                if my_count + unowned >= len(positions):
                    continue
                # They should have at least 1 in this set for it to be useful
                if their_count == 0:
                    continue
                pos = next(p for p in positions
                           if snap.get(p, {}).get('owner') is me)
                my_singletons.append((se, pos))

            for sa, want_pos in my_completable:
                for sb, give_pos in my_singletons:
                    if sa == sb:
                        continue
                    candidates.append({
                        'i_want': [want_pos], 'i_give': [give_pos],
                        'my_set': sa, 'their_set': sb,
                    })

        # PRIORITY 3: Setup/accumulation — I get closer to a monopoly,
        # give them a singleton from a set I'm not pursuing.
        # Only when no completion candidates exist with this opponent.
        if not candidates:
            # What I want: pieces from sets I'm already collecting
            accum_targets = []
            for se, positions in SET_POSITIONS.items():
                my_count = sum(1 for p in positions
                              if snap.get(p, {}).get('owner') is me)
                their_count = sum(1 for p in positions
                                 if snap.get(p, {}).get('owner') is them)
                if my_count == 0 or their_count == 0:
                    continue
                # Skip if already completable (handled by priority 1/2)
                if my_count + their_count == len(positions):
                    continue
                their_pieces = [p for p in positions
                                if snap.get(p, {}).get('owner') is them]
                for tp in their_pieces:
                    accum_targets.append((se, tp))

            # What I can give: singletons in sets I'm not pursuing
            expendable = []
            for se, positions in SET_POSITIONS.items():
                my_count = sum(1 for p in positions
                              if snap.get(p, {}).get('owner') is me)
                if my_count != 1:
                    continue
                # Don't offer if we could self-complete this set
                unowned = sum(1 for p in positions
                              if snap.get(p, {}).get('owner') is None)
                if my_count + unowned >= len(positions):
                    continue
                # Don't offer sole blockers (any opponent has all but ours)
                other_counts = {}
                for p in positions:
                    o = snap.get(p, {}).get('owner')
                    if o is not None and o is not me:
                        key = id(o)
                        other_counts[key] = other_counts.get(key, 0) + 1
                if any(c >= len(positions) - 1 for c in other_counts.values()):
                    continue
                pos = next(p for p in positions
                           if snap.get(p, {}).get('owner') is me)
                expendable.append((se, pos))

            for sa, want_pos in accum_targets:
                for sb, give_pos in expendable:
                    if sa == sb:
                        continue
                    candidates.append({
                        'i_want': [want_pos], 'i_give': [give_pos],
                        'my_set': sa, 'their_set': sb,
                        'setup': True,
                    })

        return candidates

    def _find_convergence_cash(self, post_snap, board, me, them,
                                my_monops, their_monops,
                                me_cash, them_cash, num_others):
        """Find Nash bargaining cash via trajectory convergence search.

        Returns int: positive = I pay, negative = they pay.
        """
        best_cash = 0
        min_diff = float('inf')
        lo = max(int(-them_cash), -500)
        hi = min(int(me_cash), 500)

        for cash in range(lo, hi + 1, 25):
            mc, tc = me_cash - cash, them_cash + cash
            if mc < 0 or tc < 0:
                continue
            ms = self._sim_from_snap(post_snap, board, me, my_monops, mc)
            ts = self._sim_from_snap(post_snap, board, them, their_monops, tc)
            mt, tt = _bilateral_sim(ms, ts, num_others)

            # Find closest approach between trajectories
            min_gap = float('inf')
            conv = 0
            for t in range(len(mt)):
                g = abs(mt[t] - tt[t])
                if g < min_gap:
                    min_gap = g
                    conv = mt[t] - tt[t]
            if abs(conv) < min_diff:
                min_diff = abs(conv)
                best_cash = cash

        return best_cash

    def _evaluate_trade(self, snap, board, me, them, cand, game_state):
        """Score a trade candidate. Returns dict with improvement/cash, or None."""
        n = max(0, len(game_state.players) - 2)

        # Pre-trade trajectories
        pre_mm = self._get_monopolies(snap, me)
        pre_tm = self._get_monopolies(snap, them)
        pre_ms = self._sim_from_snap(snap, board, me, pre_mm, me.state.cash)
        pre_ts = self._sim_from_snap(snap, board, them, pre_tm, them.state.cash)
        pre_mt, pre_tt = _bilateral_sim(pre_ms, pre_ts, n)

        # Post-trade state
        post = self._apply_trade(snap, me, them, cand['i_give'], cand['i_want'])
        post_mm = self._get_monopolies(post, me)
        post_tm = self._get_monopolies(post, them)

        # Alpha-based quality filter: reject if they get much more alpha
        my_new_a = sum(SML[s]['alpha'] for s in post_mm
                       if s not in pre_mm and s in SML)
        their_new_a = sum(SML[s]['alpha'] for s in post_tm
                          if s not in pre_tm and s in SML)
        if their_new_a > 0 and my_new_a > 0 and their_new_a > my_new_a + 0.02:
            return None

        # Convergence cash (Nash bargaining point)
        fc = self._find_convergence_cash(
            post, board, me, them, post_mm, post_tm,
            me.state.cash, them.state.cash, n)

        # Evaluate at fair cash level
        ms = self._sim_from_snap(post, board, me, post_mm, me.state.cash - fc)
        ts = self._sim_from_snap(post, board, them, post_tm, them.state.cash + fc)
        mt, tt = _bilateral_sim(ms, ts, n)

        my_imp = sum(mt) - sum(pre_mt)
        their_imp = sum(tt) - sum(pre_tt)

        if my_imp <= 0:
            return None
        if their_imp > my_imp * 2.0:
            return None  # relaxed fairness filter

        return {'improvement': my_imp, 'fair_cash': fc}

    def _evaluate_setup_trade(self, snap, me, them, cand):
        """Score a setup/accumulation trade (no monopoly completion).

        Uses strategic heuristic rather than bilateral sim, because the
        sim only values completed monopolies — accumulation value is
        positional (getting to 2/3 enables future completion trades).
        """
        want_pos = cand['i_want'][0]
        give_pos = cand['i_give'][0]

        target_set = cand['my_set']
        target_positions = SET_POSITIONS[target_set]
        my_count = sum(1 for p in target_positions
                      if snap.get(p, {}).get('owner') is me)

        # Progress: how close does this get us to completion?
        # 1/3 → 2/3 in a 3-set (need 1 more) = high value
        # 0/2 → 1/2 in a 2-set = moderate
        pieces_after = my_count + 1
        pieces_needed = len(target_positions) - pieces_after
        if pieces_needed <= 0:
            return None  # would complete — should be priority 1/2

        progress = pieces_after / len(target_positions)
        target_quality = GROUP_QUALITY.get(target_set, 1.0)

        # What we give up
        give_set = cand['their_set']
        give_quality = GROUP_QUALITY.get(give_set, 1.0)

        # Score: value of advancing × quality - value of what we lose
        # Weigh progress heavily — 2/3 is worth much more than 1/3
        gain = progress * target_quality
        loss = (1.0 / len(SET_POSITIONS[give_set])) * give_quality
        improvement = gain - loss

        if improvement <= 0:
            return None

        # Fair cash: based on property value difference
        want_price = snap.get(want_pos, {}).get('price', 100)
        give_price = snap.get(give_pos, {}).get('price', 100)
        fair_cash = int(want_price - give_price)  # positive = I should pay

        return {'improvement': improvement, 'fair_cash': fair_cash}

    # ------------------------------------------------------------------
    # Auction helpers
    # ------------------------------------------------------------------

    def _indifference_bid(self, game_state, player, prop, pos, se):
        """Binary search for max cost where owning beats not owning."""
        board = game_state.board
        snap = self._snapshot(board)
        n = max(0, len(game_state.players) - 2)

        # Find biggest threat for this set
        threat = None
        best_tc = -1
        for p in game_state.players:
            if p is player:
                continue
            tc = sum(1 for pp in SET_POSITIONS[se]
                     if snap.get(pp, {}).get('owner') is p)
            if tc > best_tc:
                best_tc = tc
                threat = p
        if threat is None:
            return int(prop.price * 1.50)

        # Scenario A: I get it
        snap_a = {k: dict(v) for k, v in snap.items()}
        snap_a[pos] = dict(snap[pos])
        snap_a[pos]['owner'] = player
        mm_a = self._get_monopolies(snap_a, player)
        tm_a = self._get_monopolies(snap_a, threat)

        # Scenario B: threat gets it
        snap_b = {k: dict(v) for k, v in snap.items()}
        snap_b[pos] = dict(snap[pos])
        snap_b[pos]['owner'] = threat
        mm_b = self._get_monopolies(snap_b, player)
        tm_b = self._get_monopolies(snap_b, threat)

        # Baseline: my trajectory area if threat gets it
        ms_b = self._sim_from_snap(snap_b, board, player, mm_b, player.state.cash)
        ts_b = self._sim_from_snap(snap_b, board, threat, tm_b, threat.state.cash)
        mt_b, _ = _bilateral_sim(ms_b, ts_b, n)
        base_area = sum(mt_b)

        # Binary search: find max cost where owning is still better
        lo, hi = 0, player.state.cash
        while hi - lo > 25:
            mid = (lo + hi) // 2
            ms_a = self._sim_from_snap(snap_a, board, player, mm_a,
                                        player.state.cash - mid)
            ts_a = self._sim_from_snap(snap_a, board, threat, tm_a,
                                        threat.state.cash)
            mt_a, _ = _bilateral_sim(ms_a, ts_a, n)
            if sum(mt_a) > base_area:
                lo = mid
            else:
                hi = mid

        return max(prop.price, lo)

    # ------------------------------------------------------------------
    # PlayerAIBase overrides
    # ------------------------------------------------------------------

    def landed_on_unowned_property(self, game_state, player, property):
        if player.state.cash >= property.price:
            return PlayerAIBase.Action.BUY
        return PlayerAIBase.Action.DO_NOT_BUY

    def property_offered_for_auction(self, game_state, player, property):
        board = game_state.board
        pos = board.get_index(property.name)
        cash = player.state.cash
        base = int(property.price * 1.05)

        if not isinstance(property, Street):
            return min(base, cash)

        se = property.property_set.set_enum
        positions = SET_POSITIONS.get(se, [])
        my_count = sum(1 for p in positions
                       if board.squares[p].owner is player)

        if my_count == len(positions) - 1:
            # Completes my monopoly
            bid = self._indifference_bid(game_state, player, property, pos, se)
            return min(bid, cash)

        # Accumulation: already collecting this set — bid above face value
        if my_count > 0:
            base = max(base, int(property.price * 1.20))

        # Check if blocking an opponent's monopoly
        for other in game_state.players:
            if other is player:
                continue
            tc = sum(1 for p in positions
                     if board.squares[p].owner is other)
            if tc == len(positions) - 1:
                # They'd complete — check if sole blocker
                unowned = [p for p in positions
                           if board.squares[p].owner is None and p != pos]
                if not unowned:
                    return min(int(property.price * 1.30), cash)

        return min(base, cash)

    def build_houses(self, game_state, player):
        board = game_state.board
        result = {}  # pos -> additional houses
        n_others = max(1, len(game_state.players) - 1)

        # Phase 1: build with cash down to minimum reserve
        available = player.state.cash - ABSOLUTE_MIN_CASH
        available = self._greedy_build(board, player, result, available)

        # Phase 2: mortgage-funded builds
        # Calculate mortgage capital from non-monopoly singletons
        mortgage_cap = 0
        mortgage_ept_loss = 0
        for prop in player.state.properties:
            if prop.is_mortgaged:
                continue
            if isinstance(prop, Street) and prop.number_of_houses > 0:
                continue
            # Don't mortgage monopoly members
            if isinstance(prop, Street) and prop.property_set.owner is player:
                continue
            pos = board.get_index(prop.name)
            mortgage_cap += prop.mortgage_value
            # EPT we lose from mortgaging this property
            if isinstance(prop, Station):
                rr_count = sum(1 for p in STATION_POSITIONS
                               if board.squares[p].owner is player
                               and not board.squares[p].is_mortgaged)
                if rr_count > 0:
                    mortgage_ept_loss += (MARKOV_PROBS.get(pos, 0.025)
                                          * RAILROAD_RENT.get(rr_count, 25)
                                          * n_others)
            elif isinstance(prop, Utility):
                uc = sum(1 for p in UTILITY_POSITIONS
                         if board.squares[p].owner is player
                         and not board.squares[p].is_mortgaged)
                if uc > 0:
                    mortgage_ept_loss += (MARKOV_PROBS.get(pos, 0.025)
                                          * UTILITY_MULT.get(uc, 4) * 7
                                          * n_others)
            # Streets with no houses give base rent (unimproved)
            elif isinstance(prop, Street):
                mortgage_ept_loss += (MARKOV_PROBS.get(pos, 0.025)
                                      * prop.rents[0] * n_others)

        if mortgage_cap > 0:
            # Try building with mortgage capital, but only if net EPT positive
            extra = mortgage_cap + available  # available may be negative
            if extra > 0:
                saved_result = dict(result)
                saved_avail = available
                new_avail = self._greedy_build(board, player, result, extra)
                # Calculate EPT gain from new houses
                build_ept_gain = 0
                for pos, count in result.items():
                    old_count = saved_result.get(pos, 0)
                    if count <= old_count:
                        continue
                    st = board.squares[pos]
                    prob = MARKOV_PROBS.get(pos, 0.025)
                    base_h = st.number_of_houses + old_count
                    for h in range(base_h, st.number_of_houses + count):
                        prev = st.rents[0] * 2 if h == 0 else st.rents[h]
                        cur = st.rents[h + 1] if h + 1 <= 5 else st.rents[h]
                        build_ept_gain += prob * (cur - prev) * n_others

                if build_ept_gain <= mortgage_ept_loss:
                    # Not worth it — rollback to pre-mortgage builds
                    result = saved_result

        return [(board.squares[p], c) for p, c in result.items()]

    def _greedy_build(self, board, player, result, available):
        """Build houses greedily by ROI until budget exhausted. Returns remaining budget."""
        while available > 0:
            best_roi, best_pos, best_hp = 0, -1, 0
            for ps in player.state.owned_unmortgaged_sets:
                se = ps.set_enum
                if se in (PropertySet.STATION, PropertySet.UTILITY):
                    continue
                if not ps.can_build_houses:
                    continue
                hp = ps.house_price
                if hp > available:
                    continue
                positions = SET_POSITIONS.get(se, [])
                if not positions:
                    continue
                cur = [board.squares[p].number_of_houses + result.get(p, 0)
                       for p in positions]
                min_h = min(cur)
                for i, pos in enumerate(positions):
                    h = cur[i]
                    if h >= 5 or h > min_h:
                        continue
                    st = board.squares[pos]
                    prob = MARKOV_PROBS.get(pos, 0.025)
                    cr = st.rents[0] * 2 if h == 0 else st.rents[h]
                    nr = st.rents[h + 1]
                    roi = prob * (nr - cr) / hp
                    if roi > best_roi:
                        best_roi, best_pos, best_hp = roi, pos, hp
            if best_pos < 0:
                break
            result[best_pos] = result.get(best_pos, 0) + 1
            available -= best_hp
        return available

    def money_will_be_taken(self, player, amount):
        self._pending_debt = amount

    def sell_houses(self, game_state, player):
        board = game_state.board
        debt = getattr(self, '_pending_debt', 0)
        need = debt - player.state.cash
        if need <= 0:
            return []

        result = {}
        raised = 0

        while raised < need:
            worst_roi, worst_pos = float('inf'), -1
            for se, positions in SET_POSITIONS.items():
                cur = [board.squares[p].number_of_houses - result.get(p, 0)
                       for p in positions]
                if max(cur, default=0) <= 0:
                    continue
                max_h = max(cur)
                for i, pos in enumerate(positions):
                    h = cur[i]
                    if h <= 0 or h < max_h:
                        continue
                    st = board.squares[pos]
                    if st.owner is not player:
                        continue
                    hp = st.house_price
                    if hp <= 0:
                        continue
                    prob = MARKOV_PROBS.get(pos, 0.025)
                    cr = st.rents[h]
                    pr = st.rents[0] * 2 if h == 1 else st.rents[h - 1]
                    roi = prob * (cr - pr) / hp
                    if roi < worst_roi:
                        worst_roi, worst_pos = roi, pos
            if worst_pos < 0:
                break
            st = board.squares[worst_pos]
            result[worst_pos] = result.get(worst_pos, 0) + 1
            raised += st.house_price // 2

        self._pending_debt = 0
        return [(board.squares[p], c) for p, c in result.items()]

    def mortgage_properties(self, game_state, player):
        debt = getattr(self, '_pending_debt', 0)
        need = debt - player.state.cash
        if need <= 0:
            return []

        mortgageable = []
        for prop in player.state.properties:
            if prop.is_mortgaged:
                continue
            if isinstance(prop, Street) and prop.number_of_houses > 0:
                continue
            is_mono = (isinstance(prop, Street)
                       and prop.property_set.owner is player)
            mortgageable.append((prop, is_mono))
        # Non-monopoly first, cheapest first
        mortgageable.sort(key=lambda x: (x[1], x[0].price))

        result = []
        raised = 0
        for prop, _ in mortgageable:
            if raised >= need:
                break
            result.append(prop)
            raised += prop.mortgage_value
        return result

    def unmortgage_properties(self, game_state, player):
        board = game_state.board
        result = []
        available = player.state.cash

        mortgaged = []
        for prop in player.state.properties:
            if not prop.is_mortgaged:
                continue
            cost = int(prop.mortgage_value * 1.1)
            is_mono = False
            if isinstance(prop, Street):
                se = prop.property_set.set_enum
                positions = SET_POSITIONS.get(se, [])
                if positions and all(board.squares[p].owner is player
                                     for p in positions):
                    is_mono = True
            mortgaged.append((prop, cost, is_mono))

        # Monopoly first, then cheapest
        mortgaged.sort(key=lambda x: (-x[2], x[1]))
        for prop, cost, _ in mortgaged:
            if available - cost >= ABSOLUTE_MIN_CASH * 2:
                result.append(prop)
                available -= cost

        return result

    def propose_deal(self, game_state, player):
        board = game_state.board
        snap = self._snapshot(board)
        players = list(game_state.players)

        # EPT baseline: what we gain from just building with current cash
        current_levels = self._get_monopoly_levels(player)
        available_cash = max(0, player.state.cash - ABSOLUTE_MIN_CASH)
        build_ept = self._achievable_ept(available_cash, dict(current_levels))

        # Build frontier model
        graph = self._build_ownership_graph(snap, players)
        coalitions = self._find_coalitions(graph, player, players)

        # Score coalitions
        scored = []
        for c in coalitions:
            other = c['partner']
            key = (other.name,
                   tuple(sorted(c['i_give'])),
                   tuple(sorted(c['they_give'])))
            if key in self._proposed_deals:
                continue

            # Convert coalition to candidate format for existing evaluators
            cand = {
                'i_want': c['they_give'],
                'i_give': c['i_give'],
                'my_set': (c['my_completions'][0]['set_id']
                           if c['my_completions'] else None),
                'their_set': (c['their_completions'][0]['set_id']
                              if c['their_completions'] else None),
                'setup': c['type'] == 'setup',
                'coalition': c,
            }

            if c['type'] == 'setup':
                # Use alpha-based evaluation instead of old heuristic
                pv = self._compute_property_values(graph, coalitions, player)
                give_val = sum(pv.get(p, {}).get('effective_value', 0)
                               for p in c['i_give'])
                my_alpha = c['my_alpha']
                if my_alpha <= 0:
                    continue
                # Improvement: alpha-weighted progress minus what we give up
                improvement = my_alpha * 1000 - give_val * 0.001
                if improvement <= 0:
                    continue
                want_price = sum(snap.get(p, {}).get('price', 100)
                                  for p in c['they_give'])
                give_price = sum(snap.get(p, {}).get('price', 100)
                                  for p in c['i_give'])
                r = {'improvement': improvement,
                     'fair_cash': int(want_price - give_price)}
            else:
                r = self._evaluate_trade(snap, board, player, other,
                                          cand, game_state)

            if not r:
                continue

            # --- EPT gate: does this trade beat just building? ---
            post_levels = dict(current_levels)
            if c['my_completions']:
                new_set = c['my_completions'][0]['set_id']
                if new_set in DEV_FRONTIER and new_set not in post_levels:
                    post_levels[new_set] = 0

            trade_cash_cost = max(0, r.get('fair_cash', 0))
            remaining = available_cash - trade_cash_cost
            if remaining < 0:
                remaining = 0

            lost_ept = self._ept_of_positions(c['i_give'])
            gained_ept = self._ept_of_positions(c['they_give'])
            trade_ept = (self._achievable_ept(remaining, post_levels)
                         - lost_ept + gained_ept)

            ept_advantage = trade_ept - build_ept
            r['ept_advantage'] = ept_advantage

            # Trade must beat building (or be close for setup trades)
            if ept_advantage < -1.0 and not c.get('setup', cand.get('setup')):
                continue
            if ept_advantage < -5.0:
                continue  # even setup trades have a floor

            scored.append((cand, other, r, key))

        if not scored:
            return None

        # Pick best by EPT advantage first, then improvement as tiebreaker
        scored.sort(key=lambda x: (-x[2].get('ept_advantage', 0),
                                   -x[2]['improvement']))
        c, other, r, key = scored[0]

        # Mark as proposed so we don't re-propose
        self._proposed_deals.add(key)
        self._diag['deals_we_proposed'] += 1
        self._last_proposed_to = other.ai.get_name()
        tgt = self._diag['propose_targets']
        tgt[other.ai.get_name()] = tgt.get(other.ai.get_name(), 0) + 1
        is_setup = c.get('setup', False)
        self._diag['propose_types']['setup' if is_setup else 'mutual_monopoly'] += 1

        give = [board.squares[p] for p in c['i_give']]
        want = [board.squares[p] for p in c['i_want']]
        fc = r['fair_cash']

        diag_log.debug(f"  PROPOSE {'SETUP' if is_setup else 'COMPLETION'} "
                       f"to {other.ai.get_name()}: "
                       f"give={[board.squares[p].name for p in c['i_give']]} "
                       f"want={[board.squares[p].name for p in c['i_want']]} "
                       f"fc={fc} my_set={c['my_set']} their_set={c['their_set']}")

        # Cash terms: be generous on completion, fair on setup
        if fc > 0:
            # We pay
            if is_setup:
                offer = max(1, int(fc * 1.10))  # slight premium
            else:
                offer = max(1, int(fc * 1.20))  # generous for monopoly
            offer = min(offer, player.state.cash - ABSOLUTE_MIN_CASH)
            if offer <= 0:
                offer = max(1, abs(fc))
            self._diag['propose_cash_terms'].append(('pay', fc, offer))
            return DealProposal(
                propose_to_player=other,
                properties_offered=give,
                properties_wanted=want,
                maximum_cash_offered=offer)
        elif fc < 0:
            if is_setup:
                ask = max(1, int(-fc * 0.80))  # ask near fair
            else:
                ask = max(1, int(-fc * 0.50))  # discount for monopoly
            self._diag['propose_cash_terms'].append(('ask', fc, ask))
            return DealProposal(
                propose_to_player=other,
                properties_offered=give,
                properties_wanted=want,
                minimum_cash_wanted=ask)
        self._diag['propose_cash_terms'].append(('even', 0, 0))
        return DealProposal(
            propose_to_player=other,
            properties_offered=give,
            properties_wanted=want)

    def _property_value(self, prop, board, snap, player):
        """Estimate value of a single property including blocking value."""
        pos = board.get_index(prop.name)
        base = prop.mortgage_value  # face value floor

        if isinstance(prop, Street):
            se = prop.property_set.set_enum
            positions = SET_POSITIONS.get(se, [])
            # How close are opponents to completing this set?
            for other_info in prop.property_set.owners:
                other_player, count, frac = other_info
                if other_player is player:
                    continue
                if count == len(positions) - 1:
                    # We're the sole blocker — high value
                    base = max(base, int(prop.price * 2.0))
                elif count >= len(positions) - 2:
                    base = max(base, int(prop.price * 1.3))
        return base

    def _reject(self, reason):
        """Track rejection reason and return REJECT response."""
        d = self._diag['our_reject_reasons']
        d[reason] = d.get(reason, 0) + 1
        self._diag['deals_we_rejected'] += 1
        return DealResponse(DealResponse.Action.REJECT)

    def deal_proposed(self, game_state, player, deal_proposal):
        board = game_state.board
        proposer = deal_proposal.proposed_by_player
        my_give = deal_proposal.properties_wanted   # they want from me
        my_recv = deal_proposal.properties_offered  # they give to me

        if not my_give and not my_recv:
            return self._reject('empty_deal')

        # Quick reject: if we're giving properties and not receiving any,
        # only accept if we already have a monopoly to develop with the cash
        if my_give and not my_recv:
            has_monopoly = bool(player.state.owned_unmortgaged_sets)
            if not has_monopoly:
                return self._reject('sell_no_monopoly')

        snap = self._snapshot(board)
        gp = [board.get_index(p.name) for p in my_give]
        rp = [board.get_index(p.name) for p in my_recv]
        n = max(0, len(game_state.players) - 2)
        players = list(game_state.players)

        # --- EPT gate: does accepting beat just building? ---
        current_levels = self._get_monopoly_levels(player)
        available_cash = max(0, player.state.cash - ABSOLUTE_MIN_CASH)
        reject_ept = self._achievable_ept(available_cash, dict(current_levels))

        # Compute post-acceptance monopoly state
        post_levels = dict(current_levels)
        for prop in my_recv:  # properties we'd receive
            if isinstance(prop, Street) and prop.property_set in DEV_FRONTIER:
                pset = prop.property_set
                # Would we own the full set after receiving this?
                owners = set()
                for sp in pset.properties:
                    if sp in my_recv:
                        owners.add(player)  # we'd receive it
                    elif sp in my_give:
                        pass  # we're giving it away
                    elif sp.owner is player:
                        owners.add(player)
                    else:
                        owners.add(sp.owner)
                if owners == {player} and pset not in post_levels:
                    post_levels[pset] = 0

        # Estimate cash impact from deal (conservative: assume we pay something)
        deal_cash_cost = deal_proposal.minimum_cash_wanted
        post_cash = max(0, available_cash - deal_cash_cost)

        lost_ept = self._ept_of_positions(gp)
        gained_ept = self._ept_of_positions(rp)
        accept_ept = (self._achievable_ept(post_cash, post_levels)
                      - lost_ept + gained_ept)

        if accept_ept < reject_ept - 1.0:
            return self._reject('ept_building_better')

        # --- Existing valuation layers (bilateral sim, quality filters) ---

        # Graph-based property valuation
        graph = self._build_ownership_graph(snap, players)
        coalitions = self._find_coalitions(graph, player, players)
        pv = self._compute_property_values(graph, coalitions, player)

        give_value = sum(pv.get(p, {}).get('effective_value',
                         self._property_value(prop, board, snap, player))
                         for p, prop in zip(gp, my_give))
        recv_value = sum(snap.get(p, {}).get('price', 100) * 0.5
                         for p in rp)
        value_gap = give_value - recv_value  # positive = we're net giving

        # Pre-trade
        pre_mm = self._get_monopolies(snap, player)
        pre_tm = self._get_monopolies(snap, proposer)
        pre_ms = self._sim_from_snap(snap, board, player, pre_mm,
                                      player.state.cash)
        pre_ts = self._sim_from_snap(snap, board, proposer, pre_tm,
                                      proposer.state.cash)
        pre_mt, pre_tt = _bilateral_sim(pre_ms, pre_ts, n)

        # Post-trade
        post = self._apply_trade(snap, player, proposer, gp, rp)
        post_mm = self._get_monopolies(post, player)
        post_tm = self._get_monopolies(post, proposer)

        # Hard block: don't give them a monopoly if we don't get one
        my_new_monops = [s for s in post_mm if s not in pre_mm]
        their_new_monops = [s for s in post_tm if s not in pre_tm]
        if their_new_monops and not my_new_monops:
            return self._reject('they_get_monopoly_we_dont')

        # Never accept completing a deeply negative-alpha set (Brown trap)
        if my_new_monops:
            my_best_alpha = max(SML[s]['alpha'] for s in my_new_monops
                                if s in SML)
            if my_best_alpha < -0.01:
                return self._reject('negative_alpha_completion')

        # Alpha-based quality filter (SML)
        my_nq = sum(SML[s]['alpha'] for s in my_new_monops if s in SML)
        their_nq = sum(SML[s]['alpha'] for s in their_new_monops if s in SML)
        if their_new_monops and my_new_monops:
            # Reject if they get much more alpha than we do
            if their_nq > my_nq + 0.02:
                return self._reject('alpha_too_lopsided')

        # Check if this trade advances our set collection (accumulation)
        # Use SML alpha instead of GROUP_QUALITY
        accum_value = 0.0
        if not my_new_monops:
            for se, positions in SET_POSITIONS.items():
                if SML.get(se, {}).get('alpha', 0) <= 0:
                    continue  # only accumulate toward positive-alpha sets
                before = sum(1 for p in positions
                            if snap.get(p, {}).get('owner') is player)
                after = sum(1 for p in positions
                           if post.get(p, {}).get('owner') is player)
                if after > before and after < len(positions):
                    progress = after / len(positions)
                    quality = SML[se]['quality']
                    accum_value = max(accum_value, progress * quality)

        # Early reject: giving more value without getting monopoly or
        # meaningful accumulation
        if not my_new_monops and value_gap > 0 and accum_value < 0.5:
            return self._reject('pay_without_monopoly')

        # Find convergence cash
        fc = self._find_convergence_cash(
            post, board, player, proposer, post_mm, post_tm,
            player.state.cash, proposer.state.cash, n)

        # Evaluate at convergence cash
        ms = self._sim_from_snap(post, board, player, post_mm,
                                  player.state.cash - fc)
        ts = self._sim_from_snap(post, board, proposer, post_tm,
                                  proposer.state.cash + fc)
        mt, tt = _bilateral_sim(ms, ts, n)

        my_imp = sum(mt) - sum(pre_mt)
        their_imp = sum(tt) - sum(pre_tt)

        if my_imp <= 0 and accum_value < 0.5:
            return self._reject('negative_improvement')
        if my_imp > 0 and their_imp > my_imp * 2.5:
            return self._reject('too_unfair')

        # For accumulation trades with negative sim improvement,
        # only accept if the value gap is reasonable
        is_accum_accept = (my_imp <= 0 and accum_value >= 0.5)
        if is_accum_accept and value_gap > 100:
            return self._reject('accum_overpay')

        self._diag['deals_we_accepted'] += 1
        diag_log.debug(f"  ACCEPTING {'ACCUM' if is_accum_accept else ''} "
                       f"deal from {proposer.ai.get_name()}: "
                       f"give={[p.name for p in my_give]} recv={[p.name for p in my_recv]} "
                       f"my_imp={my_imp:.0f} their_imp={their_imp:.0f} fc={fc} "
                       f"accum={accum_value:.2f}")

        # Always use maximum_cash_offered (never minimum_cash_wanted).
        # This prevents ASKED_TOO_MUCH from cash direction mismatch.
        if my_new_monops:
            # MONOPOLY-COMPLETING — offer up to half our cash
            offer = max(1, player.state.cash // 2)
        elif is_accum_accept:
            # ACCUMULATION — offer based on property value difference
            offer = max(1, abs(value_gap) + 50) if value_gap < 0 else 0
        else:
            # Non-monopoly with positive sim
            offer = max(1, int(abs(fc) * 1.20)) if fc > 0 else 0

        offer = min(offer, player.state.cash - ABSOLUTE_MIN_CASH)
        if offer <= 0:
            return DealResponse(DealResponse.Action.ACCEPT)
        return DealResponse(DealResponse.Action.ACCEPT,
                            maximum_cash_offered=offer)

    def get_out_of_jail(self, game_state, player):
        board = game_state.board
        for idx in range(40):
            sq = board.squares[idx]
            if isinstance(sq, (Street, Station, Utility)) and sq.owner is None:
                if player.state.number_of_get_out_of_jail_free_cards > 0:
                    return PlayerAIBase.Action.PLAY_GET_OUT_OF_JAIL_FREE_CARD
                return PlayerAIBase.Action.BUY_WAY_OUT_OF_JAIL
        return PlayerAIBase.Action.STAY_IN_JAIL

    def players_birthday(self):
        return "Happy Birthday!"
