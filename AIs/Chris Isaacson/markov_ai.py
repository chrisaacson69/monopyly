"""
MarkovTradeAI - Ported from Chris Isaacson's Monopoly AI engine.

Uses Markov-chain property valuation, bilateral trajectory simulation,
and Nash bargaining for trade evaluation.
"""

import copy
from monopyly import *
from monopyly.squares.street import Street
from monopyly.squares.station import Station
from monopyly.squares.utility import Utility

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

# Empirical monopoly quality from tournament data
GROUP_QUALITY = {
    PropertySet.GREEN:      1.30,
    PropertySet.YELLOW:     1.20,
    PropertySet.DARK_BLUE:  1.15,
    PropertySet.RED:        1.05,
    PropertySet.ORANGE:     1.00,
    PropertySet.LIGHT_BLUE: 0.95,
    PropertySet.PURPLE:     0.95,
    PropertySet.BROWN:      0.85,
}

ABSOLUTE_MIN_CASH = 75


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

    def get_name(self):
        return "MarkovTradeAI"

    def start_of_game(self):
        self._pending_debt = 0
        self._proposed_deals = set()
        self._deals_this_turn = 0
        self._current_turn = -1

    def deal_completed(self, deal_result):
        """Board state changed — clear proposed deals to re-evaluate."""
        self._proposed_deals = set()

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
        """Find monopoly-completing trade candidates between two players."""
        candidates = []
        for sa, pa in SET_POSITIONS.items():
            my_a = [p for p in pa if snap.get(p, {}).get('owner') is me]
            their_a = [p for p in pa if snap.get(p, {}).get('owner') is them]
            if not my_a or not their_a:
                continue
            if len(my_a) + len(their_a) != len(pa):
                continue  # third party or unowned blocks completion

            # I can complete group A by getting their properties
            # Option 1: cash-only purchase
            candidates.append({
                'i_want': their_a, 'i_give': [],
                'my_set': sa, 'their_set': None,
            })

            # Option 2: mutual trade — find groups they can complete with my help
            for sb, pb in SET_POSITIONS.items():
                if sb == sa:
                    continue
                my_b = [p for p in pb if snap.get(p, {}).get('owner') is me]
                their_b = [p for p in pb if snap.get(p, {}).get('owner') is them]
                if not their_b or not my_b:
                    continue
                if len(my_b) + len(their_b) != len(pb):
                    continue
                candidates.append({
                    'i_want': their_a, 'i_give': my_b,
                    'my_set': sa, 'their_set': sb,
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

        # Quality filter: reject if they get much better monopolies
        my_new_q = sum(GROUP_QUALITY.get(s, 1.0)
                       for s in post_mm if s not in pre_mm)
        their_new_q = sum(GROUP_QUALITY.get(s, 1.0)
                          for s in post_tm if s not in pre_tm)
        if their_new_q > 0 and my_new_q > 0 and their_new_q > my_new_q * 1.40:
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

    def _find_sweetener(self, player, board, snap, exclude_positions):
        """Find cheapest singleton property to include as visible offer."""
        best = None
        best_price = float('inf')
        for prop in player.state.properties:
            if prop.is_mortgaged:
                continue
            if isinstance(prop, Street) and prop.number_of_houses > 0:
                continue
            pos = board.get_index(prop.name)
            if pos in exclude_positions:
                continue
            # Don't give away monopoly members
            if isinstance(prop, Street):
                se = prop.property_set.set_enum
                positions = SET_POSITIONS.get(se, [])
                my_count = sum(1 for p in positions
                               if snap.get(p, {}).get('owner') is player)
                if my_count >= len(positions):
                    continue  # don't break our monopoly
                if my_count >= len(positions) - 1:
                    continue  # don't give away near-monopoly piece
            if prop.price < best_price:
                best_price = prop.price
                best = pos
        return best

    def propose_deal(self, game_state, player):
        board = game_state.board
        snap = self._snapshot(board)

        # Score all candidates, skipping ones we already proposed
        scored = []
        for other in game_state.players:
            if other is player:
                continue
            cands = self._find_trade_candidates(snap, player, other)
            for c in cands:
                key = (other.name,
                       tuple(sorted(c['i_give'])),
                       tuple(sorted(c['i_want'])))
                if key in self._proposed_deals:
                    continue
                r = self._evaluate_trade(snap, board, player, other,
                                          c, game_state)
                if r:
                    scored.append((c, other, r, key))

        if not scored:
            return None

        # Pick best improvement
        scored.sort(key=lambda x: -x[2]['improvement'])
        c, other, r, key = scored[0]

        # Mark as proposed so we don't re-propose
        self._proposed_deals.add(key)

        give_pos = list(c['i_give'])
        want_pos = list(c['i_want'])

        # Cash-only deal (no properties offered): add a sweetener so
        # the opponent can actually see something in the offer
        if not give_pos:
            sw = self._find_sweetener(player, board, snap, set(want_pos))
            if sw is not None:
                give_pos.append(sw)

        give = [board.squares[p] for p in give_pos]
        want = [board.squares[p] for p in want_pos]
        fc = r['fair_cash']

        # Be generous on cash to maximize deal completion rate.
        # A completed monopoly is worth far more than saving 10-20% on cash.
        if fc > 0:
            # We're paying — offer 10% MORE than fair to close the deal
            offer = max(1, int(fc * 1.10))
            offer = min(offer, player.state.cash - ABSOLUTE_MIN_CASH)
            if offer <= 0:
                offer = max(1, fc)
            return DealProposal(
                propose_to_player=other,
                properties_offered=give,
                properties_wanted=want,
                maximum_cash_offered=offer)
        elif fc < 0:
            # They're paying — ask for 10% LESS than fair to close the deal
            ask = max(1, int(-fc * 0.90))
            return DealProposal(
                propose_to_player=other,
                properties_offered=give,
                properties_wanted=want,
                minimum_cash_wanted=ask)
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

    def deal_proposed(self, game_state, player, deal_proposal):
        board = game_state.board
        proposer = deal_proposal.proposed_by_player
        my_give = deal_proposal.properties_wanted   # they want from me
        my_recv = deal_proposal.properties_offered  # they give to me

        if not my_give and not my_recv:
            return DealResponse(DealResponse.Action.REJECT)

        # Quick reject: if we're giving properties and not receiving any,
        # only accept if we already have a monopoly to develop with the cash
        if my_give and not my_recv:
            has_monopoly = bool(player.state.owned_unmortgaged_sets)
            if not has_monopoly:
                return DealResponse(DealResponse.Action.REJECT)

        snap = self._snapshot(board)
        gp = [board.get_index(p.name) for p in my_give]
        rp = [board.get_index(p.name) for p in my_recv]
        n = max(0, len(game_state.players) - 2)

        # Compute raw property value differential (what sim misses)
        give_value = sum(self._property_value(p, board, snap, player)
                         for p in my_give)
        recv_value = sum(self._property_value(p, board, snap, player)
                         for p in my_recv)
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
            return DealResponse(DealResponse.Action.REJECT)

        # Quality filter
        my_nq = sum(GROUP_QUALITY.get(s, 1.0) for s in my_new_monops)
        their_nq = sum(GROUP_QUALITY.get(s, 1.0) for s in their_new_monops)
        if their_nq > 0 and my_nq > 0 and their_nq > my_nq * 1.60:
            return DealResponse(DealResponse.Action.REJECT)

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

        if my_imp <= 0:
            return DealResponse(DealResponse.Action.REJECT)
        if their_imp > my_imp * 2.5:
            return DealResponse(DealResponse.Action.REJECT)

        # Minimum cash: at least cover the raw property value gap
        min_cash_needed = max(0, value_gap)

        # Be generous on cash to make deals happen — a monopoly is worth
        # far more than the cash difference
        if fc > 0:
            # We'd pay them
            if value_gap > 0 and not my_new_monops:
                # Giving more property and paying cash without getting a monopoly
                return DealResponse(DealResponse.Action.REJECT)
            # Offer 10% more than fair to close the deal
            offer = max(1, int(fc * 1.10))
            offer = min(offer, player.state.cash - ABSOLUTE_MIN_CASH)
            if offer <= 0:
                offer = max(1, fc)
            return DealResponse(DealResponse.Action.ACCEPT,
                                maximum_cash_offered=offer)
        elif fc < 0 or min_cash_needed > 0:
            # They should pay us — ask 10% less than fair to close
            ask = max(min_cash_needed, int(abs(fc) * 0.90) if fc < 0 else 0)
            if ask <= 0:
                ask = max(1, give_value // 2)  # at least half mortgage value
            return DealResponse(DealResponse.Action.ACCEPT,
                                minimum_cash_wanted=ask)
        return DealResponse(DealResponse.Action.ACCEPT)

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
