import numpy as np

import pandas as pd
import scipy.stats as stat

from dataclasses import dataclass
from enum import Enum

from copy import copy, deepcopy
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

MAX_FEEDER_VESSEL_CAPACITY = 660
MAX_MOTHER_VESSEL_CAPACITY = 1200

MONTH_NAMES = [
    'December', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November'
]
MONTH_DAYS = [
    31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30
]

WINTER_MONTHS = {11, 0, 1, 2, 3, 4}

def days_until(month: int, day: int, period_boundary: tuple[str, int]):
    boundary_month_name, boundary_day = period_boundary
    boundary_month = MONTH_NAMES.index(boundary_month_name)

    days_in_month = MONTH_DAYS[month]

    # adjust for december being the last month of the year
    if month == 0:
        return (
            boundary_day - day
            if boundary_month == 0 and day <= boundary_day
            else -1
        )
    # at this point current month is not December, hence we only modify boundary_month
    if boundary_month == 0:
        boundary_month = 12

    if month < boundary_month:
        return (
            (days_in_month - day) +
            (
                0 if month + 1 == boundary_month else
                sum(MONTH_DAYS[i] for i in range(month + 1, boundary_month))
            ) +
            boundary_day
        )

    if month == boundary_month and day <= boundary_day:
        return boundary_day - day

    return -1


@dataclass
class SimulationConfig:
    simulation_period: int
    periods_in_day: int

    # Initialization
    start_month: int | str
    start_day: int

    interflood_start: tuple[str, int]
    interflood_end: tuple[str, int]

    alt_route_start: tuple[str, int]
    alt_route_end: tuple[str, int]

    n_containers: int
    main_monthly_prod_rate: list[int]
    alt_monthly_prod_rate: list[int]

    # Fleet sizes
    n_feeders_A: int
    n_feeders_B: int
    n_feeders_C: int
    n_mainlands_A: int
    n_mainlands_B: int

    main_share_A: float = 0.6

    # Initial values
    n_loaded_A: int = 0
    n_loaded_B: int = 0
    n_loaded_C: int = 0

    disable_underloaded_S2I: bool = True  # from S to A or B
    disable_underloaded_I2S: bool = True  # from A or B to S
    disable_underloaded_S2D: bool = True  # from S to D
    disable_underloaded_D2S: bool = True  # from S to D
    disable_underloaded_I2D: bool = True  # from A or P to D
    disable_underloaded_D2I: bool = True  # from D to A or P

    underload_share_S2I: float = 0.0
    underload_share_I2S: float = 0.0
    underload_share_S2D: float = 0.0
    underload_share_D2S: float = 0.0
    underload_share_I2D: float = 0.0
    underload_share_D2I: float = 0.0

    triangulation_time: int = 114
    triangulation_n_sims: int = 0
    triangulation_fail_tol: float = 0.0 

    # Surge capacity (vessels that only activate after interflood ends)
    n_surge_feeders_A: int = 0
    n_surge_feeders_B: int = 0

    # Base wait times (days before the FIRST vessel of that type departs)
    base_wait_feeder_A: int = 0
    base_wait_feeder_B: int = 0
    base_wait_mainland_A: int = 54
    base_wait_mainland_B: int = 90

    # Round trip times (used to calculate staggering intervals)
    rt_feeder_A: int = 34
    rt_feeder_B: int = 42
    rt_feeder_C: int = 148
    rt_mainland_A: int = 245
    rt_mainland_B: int = 237


class SimulationEnvironment:

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg  # save for later uses

        n_loaded = cfg.n_loaded_A + cfg.n_loaded_B + cfg.n_loaded_C
        if n_loaded > cfg.n_containers:
            raise ValueError(
                f"Total number of containers loaded at src is greater than the "
                f"number of available containers ({n_loaded} > {cfg.n_containers})"
            )

        # Environment variables
        self.n_unloaded_at_src = cfg.n_containers - n_loaded
        self.n_unloaded_at_src_port = 0
        self.n_loaded_at_src_port_A = cfg.n_loaded_A
        self.n_loaded_at_src_port_B = cfg.n_loaded_B
        self.n_loaded_at_src_port_C = cfg.n_loaded_C

        self.n_unloaded_at_A = 0
        self.n_loaded_at_A = 0

        self.n_unloaded_at_B = 0
        self.n_loaded_at_B = 0

        self.n_unloaded_at_P = 0
        self.n_loaded_at_P = 0

        self.n_unloaded_at_dest = 0
        self.n_unloaded_at_dest_port = 0

        self.n_loaded_at_dest_port_A = 0
        self.n_loaded_at_dest_port_B = 0
        self.n_loaded_at_dest_port_C = 0

        # Tracked variables; split between 3 routes where containers are shipped
        self.n_delivered_A = 0
        self.n_delivered_B = 0
        self.n_delivered_C = 0

        if isinstance(cfg.start_month, str):
            if cfg.start_month not in MONTH_NAMES:
                raise ValueError(
                    f"{cfg.start_month = } does not exist, check your calendar"
                )
            month_name = cfg.start_month
            self.month = MONTH_NAMES.index(month_name)
        else:
            if not (0 < cfg.start_month <= 12):
                raise ValueError(
                    f"{cfg.start_month = } must be an integer from 1 to 12 "
                    f"(1 -> January, 2 -> February, ... , 12 -> December)"
                )
            self.month = cfg.start_month % 12
            month_name = MONTH_NAMES[self.month]

        days_in_month = MONTH_DAYS[self.month]
        if not (0 < cfg.start_day <= days_in_month):
            raise ValueError(
                f"{cfg.start_day = } must be an integer from 1 to {days_in_month} "
                f"(number of days in {month_name})"
            )

        self.day = cfg.start_day
        self.period = 0

        self.days_until_interflood_start = self.days_until(cfg.interflood_start)
        self.days_until_interflood_end = self.days_until(cfg.interflood_end)

        self.days_until_alt_route_start = self.days_until(cfg.alt_route_start)
        self.days_until_alt_route_end = self.days_until(cfg.alt_route_end)

        self.min_unloaded_at_src = self.n_unloaded_at_src  # only decreases
        self.min_unloaded_timestamp = self.get_timestamp()

        if not (0.0 < cfg.main_share_A < 1.0):
            raise ValueError(f"{cfg.main_share_A = } is not within (0, 1) interval")

        if (
            not isinstance(cfg.main_monthly_prod_rate, list) or
            not isinstance(cfg.main_monthly_prod_rate[0], int) or
            len(cfg.main_monthly_prod_rate) != 12
        ):
            raise ValueError(
                "cfg.main_monthly_prod_rate must be a list of 12 integers representing how "
                "many containers are to be loaded each month starting from December"
            )

        if (
            not isinstance(cfg.alt_monthly_prod_rate, list) or
            not isinstance(cfg.alt_monthly_prod_rate[0], int) or
            len(cfg.alt_monthly_prod_rate) != 12
        ):
            raise ValueError(
                "cfg.alt_monthly_prod_rate must be a list of 12 integers representing how "
                "many containers are to be loaded each month starting from December"
            )

        # Fleet and transportation handler objects; must be initialized later
        self.feeder_vessels = []
        self.mainland_vessels = []

        self.src_transport_handler = None
        self.dest_transport_handler = None
        self.interm_transport_handler = None

        self.rng = None

        self.delayed_returns = []  # entries: [periods_remaining, route_id, n]
        self.n_triangulated = 0    # total containers rented out (stats only)
        self.triangulation_enabled = (  # False in a nested copies
            cfg.triangulation_n_sims > 0
        )

    def get_timestamp(self):
        month_name = MONTH_NAMES[self.month]
        return month_name + ("  " if self.day < 10 else " ") + str(self.day) + " (period " + str(self.period) + ")"

    def days_until(self, period_boundary: tuple[str, int]):
        return days_until(
            self.month, self.day, period_boundary
        )

    def _max_safe_rented(self, vessel):
        """
        Binary-search max n_rented <= n_loaded such that diverting n_rented
        containers (returned only after shipping_time + triangulation_time)
        does not stop production in more than fail_tol share of sub-sims
        """
        cfg = self.cfg
        shipping_time = vessel.transition_time
        buffer_time = 7 * self.cfg.periods_in_day
        return_time = shipping_time + cfg.triangulation_time
        horizon = return_time + buffer_time
        route_id = vessel.id[-1]

        def feasible(n_rented):
            fails = 0
            for _ in range(cfg.triangulation_n_sims):
                env_copy = deepcopy(self)
                env_copy._transit_to_next_period()
                env_copy.triangulation_enabled = False
                env_copy.cfg.simulation_period = horizon
                for ves in env_copy.mainland_vessels:
                    if ves.id == vessel.id:
                        ves.n_loaded -= n_rented
                        break

                env_copy.delayed_returns.append((return_time, route_id, n_rented))
                seed = int(self.rng.integers(0, 2**31 - 1))
                if env_copy.simulate(seed) != 0:
                    fails += 1
                    if fails > cfg.triangulation_fail_tol * cfg.triangulation_n_sims:
                        return False

            return fails <= cfg.triangulation_fail_tol * cfg.triangulation_n_sims

        lo, hi = 0, vessel.n_loaded  # feasible(0) is the status quo
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if feasible(mid):
                lo = mid
            else:
                hi = mid - 1

        return lo

    def _transit_to_next_period(self):
        self.period += 1
        if self.period == self.cfg.periods_in_day:
            self.period = 0

            self.day += 1

            self.days_until_interflood_start -= 1
            self.days_until_interflood_end -= 1

            self.days_until_alt_route_start -= 1
            self.days_until_alt_route_end -= 1

            if self.day > MONTH_DAYS[self.month]:
                self.month = (self.month + 1) % 12
                self.day = 1

                if self.month == 1:  # started new year
                    self.days_until_interflood_start = self.days_until(self.cfg.interflood_start)
                    self.days_until_interflood_end = self.days_until(self.cfg.interflood_end)

                    self.days_until_alt_route_start = self.days_until(self.cfg.alt_route_start)
                    self.days_until_alt_route_end = self.days_until(self.cfg.alt_route_end)

    def simulate_period(self, output_handle=None):
        if output_handle is not None:
            output_handle.write("*** " + self.get_timestamp() + " ***\n")

        if self.n_unloaded_at_src < self.min_unloaded_at_src:
            self.min_unloaded_at_src = self.n_unloaded_at_src
            self.min_unloaded_timestamp = self.get_timestamp()

        # =================================================================== #
        pre_states = {}
        if (
            self.triangulation_enabled and
            self.cfg.triangulation_n_sims > 0 and
            self.cfg.triangulation_time > 0
        ):
            pre_states = {ves.id: ves.state for ves in self.mainland_vessels}
        # =================================================================== #

        for vessel in self.feeder_vessels:
            vessel.simulate(self)

        for vessel in self.mainland_vessels:
            vessel.simulate(self)

        self.src_transport_handler.handle(self)
        self.dest_transport_handler.handle(self)
        self.interm_transport_handler.handle(self)

        if self.n_unloaded_at_src < 0:
            return 1  # run failed

        # =================================================================== #
        # return previously rented containers whose delay has elapsed
        still_pending = []
        for remaining_time, route_id, n in self.delayed_returns:
            if remaining_time == 1:
                if route_id == 'A':
                    self.n_unloaded_at_A += n
                else:
                    self.n_unloaded_at_P += n
            else:
                still_pending.append((remaining_time - 1, route_id, n))

        self.delayed_returns = still_pending

        if pre_states:
            # check if mainland vessel departed destination this period
            # and probe max safe rent
            for vessel in self.mainland_vessels:
                if (
                    vessel.state == VesselState.SHIPPING_FROM_DEST_TO_INTERM and
                    vessel.n_loaded > 0 and (
                        # either finished loading last period or loaded instantly
                        pre_states[vessel.id] == VesselState.LOADING_AT_DEST or
                        pre_states[vessel.id] == VesselState.WAITING_AT_DEST
                    )
                ):
                    n_rented = self._max_safe_rented(vessel)
                    if n_rented > 0:
                        vessel.n_loaded -= n_rented  # arrives with less
                        self.n_triangulated += n_rented
                        self.delayed_returns.append(
                            (
                                # rented part comes back after arrival + delay
                                vessel.transition_time + self.cfg.triangulation_time,
                                vessel.id[-1],
                                n_rented,
                            )
                        )
        # =================================================================== #

        if output_handle is not None:
            loaded_A = self.n_loaded_at_src_port_A
            loaded_B = self.n_loaded_at_src_port_B
            loaded_C = self.n_loaded_at_src_port_C
            n_loaded = loaded_A + loaded_B + loaded_C

            delivered_A = self.n_delivered_A
            delivered_B = self.n_delivered_B
            delivered_C = self.n_delivered_C
            n_delivered = delivered_A + delivered_B + delivered_C

            output_handle.write(
                f"S: n_unloaded = {self.n_unloaded_at_src} <- "
                f"{self.src_transport_handler.schedule_unloaded[:10]}, "
                f"n_loaded = {n_loaded} (\n"
                f"  loaded_A = {loaded_A:4} <- {self.src_transport_handler.schedule_A[:10]}\n"
                f"  loaded_B = {loaded_B:4} <- {self.src_transport_handler.schedule_B[:10]}\n"
                f"  loaded_C = {loaded_C:4} <- {self.src_transport_handler.schedule_C[:10]}\n)\n"
            )
            output_handle.write(
                f"A: n_unloaded = {self.n_unloaded_at_A}, n_loaded = {self.n_loaded_at_A}\n"
                f"B: n_unloaded = {self.n_unloaded_at_B}, n_loaded = {self.n_loaded_at_B}\n"
                f"P: n_unloaded = {self.n_unloaded_at_P}, n_loaded = {self.n_loaded_at_P} <- "
                f"{self.interm_transport_handler.schedule_loaded[:10]}\n"
            )
            output_handle.write(
                f"D: n_unloaded = {self.n_unloaded_at_dest} <- "
                f"{self.dest_transport_handler.schedule_unloaded[:10]}, "
                f"n_delivered = {n_delivered} (\n"
                f"  delivered_A = {delivered_A:4} <- {self.dest_transport_handler.schedule_A[:10]}\n"
                f"  delivered_B = {delivered_B:4} <- {self.dest_transport_handler.schedule_B[:10]}\n"
                f"  delivered_C = {delivered_C:4} <- {self.dest_transport_handler.schedule_C[:10]}\n)\n"
            )

            # n_loaded_at_src = self.n_loaded_at_src_port_A + self.n_loaded_at_src_port_B + self.n_loaded_at_src_port_C
            # n_loaded_at_src += sum(self.src_transport_handler.schedule_A)
            # n_loaded_at_src += sum(self.src_transport_handler.schedule_B)
            # n_loaded_at_src += sum(self.src_transport_handler.schedule_C)

            # n_unloaded_at_src = self.n_unloaded_at_src + self.n_unloaded_at_src_port + sum(self.src_transport_handler.schedule_unloaded)

            # n_loaded_at_A = self.n_loaded_at_A
            # n_unloaded_at_A = self.n_unloaded_at_A

            # n_loaded_at_B = self.n_loaded_at_B
            # n_unloaded_at_B = self.n_unloaded_at_B + sum(self.interm_transport_handler.schedule_unloaded)

            # n_loaded_at_P = self.n_loaded_at_P + sum(self.interm_transport_handler.schedule_loaded)
            # n_unloaded_at_P = self.n_unloaded_at_P

            # n_loaded_at_dest = self.n_loaded_at_dest_port_A + self.n_loaded_at_dest_port_B + self.n_loaded_at_dest_port_C
            # n_loaded_at_dest += sum(self.dest_transport_handler.schedule_A)
            # n_loaded_at_dest += sum(self.dest_transport_handler.schedule_B)
            # n_loaded_at_dest += sum(self.dest_transport_handler.schedule_C)

            # n_unloaded_at_dest = self.n_unloaded_at_dest + self.n_unloaded_at_dest_port + sum(self.dest_transport_handler.schedule_unloaded)
            # n_loaded_on_ships = 0

            for vessel in self.feeder_vessels:
                output_handle.write(vessel.id + " > " + vessel.log() + "\n")
                # n_loaded_on_ships += vessel.n_loaded

            for vessel in self.mainland_vessels:
                output_handle.write(vessel.id + " > " + vessel.log() + "\n")
                # n_loaded_on_ships += vessel.n_loaded

            # S = n_unloaded_at_src + n_loaded_at_src + n_loaded_at_A + n_unloaded_at_A + n_unloaded_at_P + n_loaded_at_P + n_unloaded_at_B + n_loaded_at_B + n_loaded_at_dest + n_unloaded_at_dest + n_loaded_on_ships
            # if S != self.cfg.n_containers:
            #     raise RuntimeError(
            #         f"Number of containers mismatch {S} != {self.cfg.n_containers} (initial number of containers) with\n"
            #         f"{n_unloaded_at_src = } {n_loaded_at_src = } {n_loaded_at_A = } {n_unloaded_at_A = } {n_unloaded_at_P = } {n_loaded_at_P = } {n_unloaded_at_B = } {n_loaded_at_B = } {n_loaded_at_dest = } {n_unloaded_at_dest = } {n_loaded_on_ships = }"
            #     )

        self._transit_to_next_period()

        return 0

    def simulate(
        self, seed,
        output_handle=None,
        verbose_start=None,
        verbose_end=None,
        print_final_stats=False,
    ):
        self.rng = np.random.default_rng(seed)

        if verbose_start is not None:
            start_month_name, start_day = verbose_start
            if start_month_name not in MONTH_NAMES:
                raise ValueError(
                    f"{start_month_name = } does not exist, check your calendar"
                )
            start_month = MONTH_NAMES.index(start_month_name)

            days_in_month = MONTH_DAYS[start_month]
            if not (0 < start_day <= days_in_month):
                raise ValueError(
                    f"{start_day = } must be integer from 1 to {days_in_month} "
                    f"(number of days in {start_month_name})"
                )
        else:
            start_month = start_day = -1

        if verbose_end is not None:
            end_month_name, end_day = verbose_end
            if end_month_name not in MONTH_NAMES:
                raise ValueError(
                    f"{end_month_name = } does not exist, check your calendar"
                )
            end_month = MONTH_NAMES.index(end_month_name)

            days_in_month = MONTH_DAYS[end_month]
            if not (0 < end_day <= days_in_month):
                raise ValueError(
                    f"{end_day = } must be integer from 1 to {days_in_month} "
                    f"(number of days in {end_month_name})"
                )
        else:
            end_month = end_day = -1

        ret_code = 0
        verbose = False
        for i in range(self.cfg.simulation_period):
            if self.month == start_month and self.day == start_day:
                verbose = True  # toggle on
            if print_final_stats and i + 1 == self.cfg.simulation_period:
                verbose = True  # print final day logs

            ret_code = self.simulate_period(
                output_handle if verbose else None
            )
            if ret_code != 0:
                break

            if self.month == end_month and self.day == end_day:
                verbose = False  # toggle off

        if print_final_stats:
            if ret_code != 0:
                output_handle.write(f"!!! RUN FAILED (code {ret_code}) !!!\n")

            n_delivered = (
                self.n_delivered_A + self.n_delivered_B + self.n_delivered_C
            )
            n_delivered += sum(self.dest_transport_handler.schedule_A)
            n_delivered += sum(self.dest_transport_handler.schedule_B)
            n_delivered += sum(self.dest_transport_handler.schedule_C)

            output_handle.write(
                f"*** Final stats ***\n"
                f"Delivered to destination: {n_delivered}\n"
                f"Minimum number of available containers: "
                f"{self.min_unloaded_at_src} ({self.min_unloaded_timestamp})\n"
            )
            if self.triangulation_enabled:
                output_handle.write(f"{self.n_triangulated}")

        return ret_code


class VesselState(Enum):
    WAITING_AT_SRC = 0
    WAITING_AT_INTERM = 1
    WAITING_AT_DEST = 2

    LOADING_AT_SRC = 3
    LOADING_AT_INTERM = 4
    LOADING_AT_DEST = 5

    UNLOADING_AT_SRC = 6
    UNLOADING_AT_INTERM = 7
    UNLOADING_AT_DEST = 8

    SHIPPING_FROM_SRC_TO_INTERM = 9
    SHIPPING_FROM_SRC_TO_DEST = 10

    SHIPPING_FROM_INTERM_TO_SRC = 11
    SHIPPING_FROM_INTERM_TO_DEST = 12

    SHIPPING_FROM_DEST_TO_INTERM = 13
    SHIPPING_FROM_DEST_TO_SRC = 14


class FeederVessel_Main:

    def __init__(
        self,
        vessel_id: str,
        max_capacity: int,
        wait_time: int,  # starting wait time
        get_wait_time,
        get_load_rate,
        get_shipping_time,
    ):
        self.state = VesselState.WAITING_AT_SRC

        if not (vessel_id[-1] in ('A', 'B') and vessel_id[-2] == ':'):
            raise ValueError("vessel_id must end with either ':A' or ':B' depending on route")

        self.id = vessel_id
        self.max_capacity = max_capacity  # in TEU's

        self.transition_time = wait_time
        self.anchor_load_rate = 0

        self.n_loaded = 0  # how many containers are currently loaded on the ship

        self.get_wait_time = get_wait_time
        self.get_load_rate = get_load_rate
        self.get_shipping_time = get_shipping_time

    def simulate(self, env: SimulationEnvironment):
        if self.transition_time > 0:
            self.transition_time -= 1
            return  # go to the next time period

        periods_in_day = env.cfg.periods_in_day

        if self.state == VesselState.WAITING_AT_SRC:
            if env.days_until_interflood_start <= 0 and env.days_until_interflood_end > 0:
                # wait at source until interflood ends
                self.transition_time = env.days_until_interflood_end * periods_in_day
                self.state = VesselState.WAITING_AT_SRC
                return  # do not fall through

            self.anchor_load_rate = 0
            self.state = VesselState.LOADING_AT_SRC

        if self.state == VesselState.WAITING_AT_INTERM:
            self.anchor_load_rate = 0
            self.state = VesselState.LOADING_AT_INTERM

        if self.state == VesselState.LOADING_AT_SRC:
            if env.days_until_interflood_start == 1:
                # depart as is, should return by the time interflood ends
                self.transition_time = self.get_shipping_time(env, self.id, self.state)
                self.state = VesselState.SHIPPING_FROM_SRC_TO_INTERM
                return  # do not fall through

            n_loaded_at_src = (
                env.n_loaded_at_src_port_A if self.id[-1] == 'A' else
                env.n_loaded_at_src_port_B
            )
            n_required = (
                round(env.cfg.underload_share_S2I * self.max_capacity)
                if env.cfg.disable_underloaded_S2I
                else -1
            )

            if self.anchor_load_rate == 0:
                # sample base load rate
                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                factor = 1.0
            else:
                # sample small deviation
                factor = self.get_load_rate(env, self.id, self.state)

            if self.anchor_load_rate > 0:
                load_rate = round(self.anchor_load_rate * factor)
                n_load = min(
                    n_loaded_at_src, load_rate, self.max_capacity - self.n_loaded
                )

                if self.id[-1] == 'A':
                    env.n_loaded_at_src_port_A -= n_load
                else:
                    env.n_loaded_at_src_port_B -= n_load

                self.n_loaded += n_load
            else:
                n_load = 0

            if self.n_loaded == self.max_capacity or (
                self.n_loaded > n_required and (
                    n_load == 0 or n_load < load_rate
                )
            ):
                # depart after loading to sufficient capacity
                self.transition_time = self.get_shipping_time(env, self.id, self.state)
                self.state = VesselState.SHIPPING_FROM_SRC_TO_INTERM

        elif self.state == VesselState.LOADING_AT_INTERM:
            n_unloaded_at_interm = (
                env.n_unloaded_at_A if self.id[-1] == 'A' else
                env.n_unloaded_at_B
            )
            n_required = (
                round(env.cfg.underload_share_I2S * self.max_capacity)
                if env.cfg.disable_underloaded_I2S
                else -1
            )

            if self.anchor_load_rate == 0:
                # sample base load rate
                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                factor = 1.0
            else:
                # sample small deviation
                factor = self.get_load_rate(env, self.id, self.state)

            if self.anchor_load_rate > 0:
                load_rate = round(self.anchor_load_rate * factor)
                n_load = min(
                    n_unloaded_at_interm, load_rate, self.max_capacity - self.n_loaded
                )

                if self.id[-1] == 'A':
                    env.n_unloaded_at_A -= n_load
                else:
                    env.n_unloaded_at_B -= n_load

                self.n_loaded += n_load
            else:
                n_load = 0

            if self.n_loaded == self.max_capacity or (
                self.n_loaded > n_required and (
                    n_load == 0 or n_load < load_rate
                )
            ):
                shipping_time = self.get_shipping_time(env, self.id, self.state)
                shipping_time_days = (shipping_time + periods_in_day - 1) // periods_in_day

                self.state = VesselState.UNLOADING_AT_SRC
                self.anchor_load_rate = 0

                # NOTE: will reuse this load rate if we ship to source
                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                if self.anchor_load_rate <= 0:
                    raise ValueError(
                        f"get_load_rate must return positive anchor load rate for "
                        f"unloading, instead got {self.anchor_load_rate = }"
                    )

                unloading_time = (
                    self.n_loaded + self.anchor_load_rate - 1
                ) // self.anchor_load_rate
                shipping_unloading_time_days = (
                    shipping_time + unloading_time + periods_in_day - 1
                ) // periods_in_day

                if (
                    env.days_until_interflood_start > 0 and
                    shipping_unloading_time_days >= env.days_until_interflood_start
                ) or (
                    env.days_until_interflood_start <= 0 and
                    env.days_until_interflood_end > 0 and
                    shipping_time_days < env.days_until_interflood_end
                ):
                    # wait at intermidiate point until interflood ends
                    self.transition_time = (
                        env.days_until_interflood_end * periods_in_day - shipping_time
                    )
                    self.state = VesselState.WAITING_AT_INTERM
                else:
                    # depart after loading to sufficient capacity
                    self.transition_time = shipping_time
                    self.state = VesselState.SHIPPING_FROM_INTERM_TO_SRC

        elif self.state == VesselState.UNLOADING_AT_SRC:
            # sample small deviation; base load rate is already sampled
            factor = self.get_load_rate(env, self.id, self.state)

            load_rate = round(self.anchor_load_rate * factor)
            n_load = min(self.n_loaded, load_rate)

            env.src_transport_handler.n_unloaded_at_src_port += n_load
            self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_SRC

        elif self.state == VesselState.UNLOADING_AT_INTERM:
            # sample small deviation; base load rate is already sampled
            factor = self.get_load_rate(env, self.id, self.state)

            load_rate = round(self.anchor_load_rate * factor)
            n_load = min(self.n_loaded, load_rate)

            if self.id[-1] == 'A':
                env.interm_transport_handler.n_loaded_at_A += n_load
            else:
                env.interm_transport_handler.n_loaded_at_B += n_load

            self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_INTERM

        elif self.state == VesselState.SHIPPING_FROM_SRC_TO_INTERM:
            # unload after arriving to intermidiate point, if there is anything
            if self.n_loaded > 0:
                self.state = VesselState.UNLOADING_AT_INTERM
                self.anchor_load_rate = 0

                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                if self.anchor_load_rate <= 0:
                    raise ValueError(
                        f"get_load_rate must return positive anchor load rate for "
                        f"unloading, instead got {self.anchor_load_rate = }"
                    )

                n_load = min(self.n_loaded, self.anchor_load_rate)

                if self.id[-1] == 'A':
                    env.interm_transport_handler.n_loaded_at_A += n_load
                else:
                    env.interm_transport_handler.n_loaded_at_B += n_load

                self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_INTERM

        elif self.state == VesselState.SHIPPING_FROM_INTERM_TO_SRC:
            # unload after arriving to source, if there is anything
            if self.n_loaded > 0:
                self.state = VesselState.UNLOADING_AT_SRC

                # NOTE: already sampled anchor load rate in LOADING_AT_INTERM branch
                n_load = min(self.n_loaded, self.anchor_load_rate)

                env.src_transport_handler.n_unloaded_at_src_port += n_load
                self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_SRC

        else:
            raise ValueError(f"Unexpected state = {self.state}")

    def log(self):
        interm = self.id[-1]

        if (
            self.state == VesselState.WAITING_AT_SRC or
            self.state == VesselState.WAITING_AT_INTERM
        ):
            point = (
                'S' if self.state == VesselState.WAITING_AT_SRC else interm
            )
            return f"waiting at {point}: t = {self.transition_time}"
        elif (
            self.state == VesselState.LOADING_AT_SRC or
            self.state == VesselState.LOADING_AT_INTERM
        ):
            point = (
                'S' if self.state == VesselState.LOADING_AT_SRC else interm
            )
            return (
                f"loading at {point}: v_anchor = {self.anchor_load_rate}, "
                f"n = {self.n_loaded}/{self.max_capacity}"
            )
        elif (
            self.state == VesselState.UNLOADING_AT_SRC or
            self.state == VesselState.UNLOADING_AT_INTERM
        ):
            point = (
                'S' if self.state == VesselState.UNLOADING_AT_SRC else interm
            )
            return (
                f"unloading at {point}: v_anchor = {self.anchor_load_rate}, "
                f"n = {self.n_loaded}/{self.max_capacity}"
            )
        else:
            if self.state == VesselState.SHIPPING_FROM_SRC_TO_INTERM:
                departure_point = 'S'
                arrival_point = interm
            else:
                departure_point = interm
                arrival_point = 'S'

            return (
                f"shipping from {departure_point} to {arrival_point}: t = "
                f"{self.transition_time}, n = {self.n_loaded}/{self.max_capacity}"
            )


class FeederVessel_Alt:

    def __init__(
        self,
        vessel_id: str,
        max_capacity: int,
        wait_time: int,  # starting wait time
        get_wait_time,
        get_load_rate,
        get_shipping_time,
        get_shipping_time_backup,
    ):
        self.state = VesselState.WAITING_AT_SRC

        if not (vessel_id[-1] == 'C' and vessel_id[-2] == ':'):
            raise ValueError("vessel_id must end with ':C'")

        self.id = vessel_id
        self.max_capacity = max_capacity  # in TEU's

        self.transition_time = wait_time
        self.anchor_load_rate = 0

        self.n_loaded = 0  # how many containers are currently loaded on the ship

        self.get_wait_time = get_wait_time
        self.get_load_rate = get_load_rate
        self.get_shipping_time = get_shipping_time
        self.get_shipping_time_backup = get_shipping_time_backup

    def simulate(self, env: SimulationEnvironment):
        if self.transition_time > 0:
            self.transition_time -= 1
            return  # go to the next time period

        periods_in_day = env.cfg.periods_in_day

        if self.state == VesselState.WAITING_AT_SRC:
            if env.days_until_alt_route_start > 0:
                # wait at source until alternative route is available
                self.transition_time = env.days_until_alt_route_start * periods_in_day
                self.state = VesselState.WAITING_AT_SRC
                return  # do not fall through

            if env.days_until_alt_route_end <= 0:
                # wait at source until alternative route is available
                self.transition_time = (
                    env.days_until(('December', 31)) + 1 +
                    days_until(1, 1, env.cfg.alt_route_start)
                ) * periods_in_day
                self.state = VesselState.WAITING_AT_SRC
                return  # do not fall through

            self.anchor_load_rate = 0
            self.state = VesselState.LOADING_AT_SRC

        if self.state == VesselState.WAITING_AT_DEST:
            self.anchor_load_rate = 0
            self.state = VesselState.LOADING_AT_DEST

        if self.state == VesselState.LOADING_AT_SRC:
            n_loaded_at_src = env.n_loaded_at_src_port_C
            n_required = (
                round(env.cfg.underload_share_S2D * self.max_capacity)
                if env.cfg.disable_underloaded_S2D
                else -1
            )

            if self.anchor_load_rate == 0:
                # sample base load rate
                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                factor = 1.0
            else:
                # sample small deviation
                factor = self.get_load_rate(env, self.id, self.state)

            if self.anchor_load_rate > 0:
                load_rate = round(self.anchor_load_rate * factor)
                n_load = min(
                    n_loaded_at_src, load_rate, self.max_capacity - self.n_loaded
                )

                env.n_loaded_at_src_port_C -= n_load
                self.n_loaded += n_load
            else:
                if env.days_until_interflood_start == 0:
                    # wait at source until interflood ends
                    self.transition_time = (
                        env.days_until_interflood_end * periods_in_day
                    )
                    self.state = VesselState.WAITING_AT_SRC
                    return  # do not fall through

                n_load = 0

            if self.n_loaded == self.max_capacity or (
                self.n_loaded > n_required and (
                    n_load == 0 or n_load < load_rate
                )
            ):
                shipping_time = self.get_shipping_time(env, self.id, self.state)
                shipping_time_days = (
                    shipping_time + periods_in_day - 1
                ) // periods_in_day

                if (
                    env.days_until_alt_route_start <= 0 and
                    env.days_until_alt_route_end > 0 and
                    shipping_time_days < env.days_until_alt_route_end
                ):
                    # depart after loading to sufficient capacity
                    self.transition_time = shipping_time
                    self.state = VesselState.SHIPPING_FROM_SRC_TO_DEST
                else:
                    # wait at source until alternative route is available
                    self.transition_time = (
                        env.days_until(('December', 31)) + 1 +
                        days_until(1, 1, env.cfg.alt_route_start)
                    ) * periods_in_day
                    self.state = VesselState.WAITING_AT_SRC

        elif self.state == VesselState.LOADING_AT_DEST:
            n_unloaded_at_dest = env.n_unloaded_at_dest_port
            n_required = (
                round(env.cfg.underload_share_D2S * self.max_capacity)
                if env.cfg.disable_underloaded_D2S
                else -1
            )

            if self.anchor_load_rate == 0:
                # sample base load rate
                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                factor = 1.0
            else:
                # sample small deviation
                factor = self.get_load_rate(env, self.id, self.state)

            if self.anchor_load_rate > 0:
                load_rate = round(self.anchor_load_rate * factor)
                n_load = min(
                    n_unloaded_at_dest, load_rate, self.max_capacity - self.n_loaded
                )

                env.n_unloaded_at_dest_port -= n_load
                self.n_loaded += n_load
            else:
                n_load = 0

            if self.n_loaded == self.max_capacity or (
                self.n_loaded > n_required and (
                    n_load == 0 or n_load < load_rate
                )
            ):
                shipping_time = self.get_shipping_time(env, self.id, self.state)
                shipping_time_days = (shipping_time + periods_in_day - 1) // periods_in_day

                if (
                    env.days_until_alt_route_start <= 0 and
                    env.days_until_alt_route_end > 0 and
                    shipping_time_days < env.days_until_alt_route_end
                ):
                    # return via alternative route
                    self.transition_time = shipping_time
                    self.state = VesselState.SHIPPING_FROM_DEST_TO_SRC
                else:
                    # return via main route
                    self.transition_time = self.get_shipping_time_backup(env, self.id, self.state)
                    self.state = VesselState.SHIPPING_FROM_DEST_TO_INTERM

        elif self.state == VesselState.UNLOADING_AT_SRC:
            # sample small deviation; base load rate is already sampled
            factor = self.get_load_rate(env, self.id, self.state)

            load_rate = round(self.anchor_load_rate * factor)
            n_load = min(self.n_loaded, load_rate)

            env.src_transport_handler.n_unloaded_at_src_port += n_load
            self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_SRC

        elif self.state == VesselState.UNLOADING_AT_DEST:
            # sample small deviation; base load rate is already sampled
            factor = self.get_load_rate(env, self.id, self.state)

            load_rate = round(self.anchor_load_rate * factor)
            n_load = min(self.n_loaded, load_rate)

            env.dest_transport_handler.loaded_C += n_load
            self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_DEST

        elif self.state == VesselState.SHIPPING_FROM_SRC_TO_DEST:
            # unload after arriving to destination, if there is anything
            if self.n_loaded > 0:
                self.state = VesselState.UNLOADING_AT_DEST
                self.anchor_load_rate = 0

                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                if self.anchor_load_rate <= 0:
                    raise ValueError(
                        f"get_load_rate must return positive anchor load rate for "
                        f"unloading, instead got {self.anchor_load_rate = }"
                    )

                n_load = min(self.n_loaded, self.anchor_load_rate)

                env.dest_transport_handler.loaded_C += n_load
                self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_DEST

        elif self.state == VesselState.SHIPPING_FROM_DEST_TO_SRC:
            # unload after arriving to source, if there is anything
            if self.n_loaded > 0:
                self.state = VesselState.UNLOADING_AT_SRC
                self.anchor_load_rate = 0

                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                if self.anchor_load_rate <= 0:
                    raise ValueError(
                        f"get_load_rate must return positive anchor load rate for "
                        f"unloading, instead got {self.anchor_load_rate = }"
                    )

                n_load = min(self.n_loaded, self.anchor_load_rate)

                env.src_transport_handler.n_unloaded_at_src_port += n_load
                self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_SRC

        elif self.state == VesselState.SHIPPING_FROM_DEST_TO_INTERM:
            # unload after arriving to intermidiate point, if there is anything
            if self.n_loaded > 0:
                self.state = VesselState.UNLOADING_AT_INTERM
                self.anchor_load_rate = 0

                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                if self.anchor_load_rate <= 0:
                    raise ValueError(
                        f"get_load_rate must return positive anchor load rate for "
                        f"unloading, instead got {self.anchor_load_rate = }"
                    )

                n_load = min(self.n_loaded, self.anchor_load_rate)

                env.interm_transport_handler.n_unloaded_at_A += n_load
                self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_INTERM

        elif self.state == VesselState.UNLOADING_AT_INTERM:
            # sample small deviation; base load rate is already sampled
            factor = self.get_load_rate(env, self.id, self.state)

            load_rate = round(self.anchor_load_rate * factor)
            n_load = min(self.n_loaded, load_rate)

            env.interm_transport_handler.n_unloaded_at_A += n_load
            self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_INTERM

        elif self.state == VesselState.WAITING_AT_INTERM:
            # depart without any loading as it is not part of the ship routine
            self.transition_time = self.get_shipping_time_backup(env, self.id, self.state)
            self.state = VesselState.SHIPPING_FROM_INTERM_TO_SRC

        elif self.state == VesselState.SHIPPING_FROM_INTERM_TO_SRC:
            # wait at source until alternative route is available
            if env.days_until_alt_route_start >= 0:
                self.transition_time = env.days_until_alt_route_start * periods_in_day
            else:
                self.transition_time = (
                    env.days_until(('December', 31)) + 1 +
                    days_until(1, 1, env.cfg.alt_route_start)
                ) * periods_in_day
            self.state = VesselState.WAITING_AT_SRC

        else:
            raise ValueError(f"Unexpected state = {self.state}")

    def log(self):
        if (
            self.state == VesselState.WAITING_AT_SRC or
            self.state == VesselState.WAITING_AT_DEST or
            self.state == VesselState.WAITING_AT_INTERM
        ):
            if self.state == VesselState.WAITING_AT_SRC:
                point = 'S'
            elif self.state == VesselState.WAITING_AT_DEST:
                point = 'D'
            else:
                point = 'B'

            return f"waiting at {point}: t = {self.transition_time}"
        elif (
            self.state == VesselState.LOADING_AT_SRC or
            self.state == VesselState.LOADING_AT_DEST
        ):
            point = (
                'S' if self.state == VesselState.LOADING_AT_SRC else 'D'
            )
            return (
                f"loading at {point}: v_anchor = {self.anchor_load_rate}, "
                f"n = {self.n_loaded}/{self.max_capacity}"
            )
        elif (
            self.state == VesselState.UNLOADING_AT_SRC or
            self.state == VesselState.UNLOADING_AT_DEST or
            self.state == VesselState.UNLOADING_AT_INTERM
        ):
            if self.state == VesselState.UNLOADING_AT_SRC:
                point = 'S'
            elif self.state == VesselState.UNLOADING_AT_DEST:
                point = 'D'
            else:
                point = 'B'

            return (
                f"unloading at {point}: v_anchor = {self.anchor_load_rate}, "
                f"n = {self.n_loaded}/{self.max_capacity}"
            )
        elif (
            self.state == VesselState.SHIPPING_FROM_SRC_TO_DEST or
            self.state == VesselState.SHIPPING_FROM_DEST_TO_SRC or
            self.state == VesselState.SHIPPING_FROM_DEST_TO_INTERM
        ):
            if self.state == VesselState.SHIPPING_FROM_SRC_TO_DEST:
                departure_point = 'S'
                arrival_point = 'D'
            else:
                departure_point = 'D'
                arrival_point = (
                    'S' if self.state == VesselState.SHIPPING_FROM_DEST_TO_SRC else 'B'
                )

            return (
                f"shipping from {departure_point} to {arrival_point}: t = "
                f"{self.transition_time}, n = {self.n_loaded}/{self.max_capacity}"
            )
        else:  # self.state == VesselState.SHIPPING_FROM_INTERM_TO_SRC
            return (
                f"shipping from A to S: t = {self.transition_time}, n = 0/{self.max_capacity}"
            )


class MainlandVessel:

    def __init__(
        self,
        vessel_id: str,
        max_capacity: int,
        wait_time: int,  # starting wait time
        get_wait_time,
        get_load_rate,
        get_shipping_time,
    ):
        self.state = VesselState.WAITING_AT_INTERM

        if not (vessel_id[-1] in ('A', 'B') and vessel_id[-2] == ':'):
            raise ValueError("vessel_id must end with either ':A' or ':B' depending on route")

        self.id = vessel_id
        self.max_capacity = max_capacity  # in TEU's

        self.transition_time = wait_time
        self.anchor_load_rate = 0

        self.n_loaded = 0  # how many containers are currently loaded on the ship

        self.get_wait_time = get_wait_time
        self.get_load_rate = get_load_rate
        self.get_shipping_time = get_shipping_time

    def simulate(self, env: SimulationEnvironment):
        if self.transition_time > 0:
            self.transition_time -= 1
            return  # go to the next time period

        if self.state == VesselState.WAITING_AT_INTERM:
            self.anchor_load_rate = 0
            self.state = VesselState.LOADING_AT_INTERM

        if self.state == VesselState.WAITING_AT_DEST:
            self.anchor_load_rate = 0
            self.state = VesselState.LOADING_AT_DEST

        if self.state == VesselState.LOADING_AT_INTERM:
            n_loaded_at_interm = (
                env.n_loaded_at_A if self.id[-1] == 'A' else
                env.n_loaded_at_P
            )
            n_required = (
                round(env.cfg.underload_share_I2D * self.max_capacity)
                if env.cfg.disable_underloaded_I2D
                else -1
            )

            if self.anchor_load_rate == 0:
                # sample base load rate
                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                factor = 1.0
            else:
                # sample small deviation
                factor = self.get_load_rate(env, self.id, self.state)

            if self.anchor_load_rate > 0:
                load_rate = round(self.anchor_load_rate * factor)
                n_load = min(
                    n_loaded_at_interm, load_rate, self.max_capacity - self.n_loaded
                )

                if self.id[-1] == 'A':
                    env.n_loaded_at_A -= n_load
                else:
                    env.n_loaded_at_P -= n_load

                self.n_loaded += n_load
            else:
                n_load = 0

            if self.n_loaded == self.max_capacity or (
                self.n_loaded > n_required and (
                    n_load == 0 or n_load < load_rate
                )
            ):
                # depart after loading to sufficient capacity
                self.transition_time = self.get_shipping_time(env, self.id, self.state)
                self.state = VesselState.SHIPPING_FROM_INTERM_TO_DEST

        elif self.state == VesselState.LOADING_AT_DEST:
            n_unloaded_at_dest = env.n_unloaded_at_dest_port
            n_required = (
                round(env.cfg.underload_share_D2I * self.max_capacity)
                if env.cfg.disable_underloaded_D2I
                else -1
            )

            if self.anchor_load_rate == 0:
                # sample base load rate
                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                factor = 1.0
            else:
                # sample small deviation
                factor = self.get_load_rate(env, self.id, self.state)

            if self.anchor_load_rate > 0:
                load_rate = round(self.anchor_load_rate * factor)
                n_load = min(
                    n_unloaded_at_dest, load_rate, self.max_capacity - self.n_loaded
                )

                env.n_unloaded_at_dest_port -= n_load
                self.n_loaded += n_load
            else:
                n_load = 0

            if self.n_loaded == self.max_capacity or (
                self.n_loaded > n_required and (
                    n_load == 0 or n_load < load_rate
                )
            ):
                # depart after loading to sufficient capacity
                self.transition_time = self.get_shipping_time(env, self.id, self.state)
                self.state = VesselState.SHIPPING_FROM_DEST_TO_INTERM

        elif self.state == VesselState.UNLOADING_AT_INTERM:
            # sample small deviation; base load rate is already sampled
            factor = self.get_load_rate(env, self.id, self.state)

            load_rate = round(self.anchor_load_rate * factor)
            n_load = min(self.n_loaded, load_rate)

            if self.id[-1] == 'A':
                env.interm_transport_handler.n_unloaded_at_A += n_load
            else:
                env.interm_transport_handler.n_unloaded_at_P += n_load

            self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_INTERM

        elif self.state == VesselState.UNLOADING_AT_DEST:
            # sample small deviation; base load rate is already sampled
            factor = self.get_load_rate(env, self.id, self.state)

            load_rate = round(self.anchor_load_rate * factor)
            n_load = min(self.n_loaded, load_rate)

            if self.id[-1] == 'A':
                env.dest_transport_handler.loaded_A += n_load
            else:
                env.dest_transport_handler.loaded_B += n_load

            self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_DEST

        elif self.state == VesselState.SHIPPING_FROM_INTERM_TO_DEST:
            # unload after arriving to destination, if there is anything
            if self.n_loaded > 0:
                self.state = VesselState.UNLOADING_AT_DEST
                self.anchor_load_rate = 0

                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                if self.anchor_load_rate <= 0:
                    raise ValueError(
                        f"get_load_rate must return positive anchor load rate for "
                        f"unloading, instead got {self.anchor_load_rate = }"
                    )

                n_load = min(self.n_loaded, self.anchor_load_rate)

                if self.id[-1] == 'A':
                    env.dest_transport_handler.loaded_A += n_load
                else:
                    env.dest_transport_handler.loaded_B += n_load

                self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_DEST

        elif self.state == VesselState.SHIPPING_FROM_DEST_TO_INTERM:
            # unload after arriving to intermidiate point, if there is anything
            if self.n_loaded > 0:
                self.state = VesselState.UNLOADING_AT_INTERM
                self.anchor_load_rate = 0

                self.anchor_load_rate = self.get_load_rate(env, self.id, self.state)
                if self.anchor_load_rate <= 0:
                    raise ValueError(
                        f"get_load_rate must return positive anchor load rate for "
                        f"unloading, instead got {self.anchor_load_rate = }"
                    )

                n_load = min(self.n_loaded, self.anchor_load_rate)

                if self.id[-1] == 'A':
                    env.interm_transport_handler.n_unloaded_at_A += n_load
                else:
                    env.interm_transport_handler.n_unloaded_at_P += n_load

                self.n_loaded -= n_load

            if self.n_loaded == 0:
                self.transition_time = self.get_wait_time(env, self.id, self.state)
                self.state = VesselState.WAITING_AT_INTERM

        else:
            raise ValueError(f"Unexpected state = {self.state}")

    def log(self):
        interm = 'A' if self.id[-1] == 'A' else 'P'

        if (
            self.state == VesselState.WAITING_AT_INTERM or
            self.state == VesselState.WAITING_AT_DEST
        ):
            point = (
                interm if self.state == VesselState.WAITING_AT_INTERM else 'D'
            )
            return f"waiting at {point}: t = {self.transition_time}"
        elif (
            self.state == VesselState.LOADING_AT_INTERM or
            self.state == VesselState.LOADING_AT_DEST
        ):
            point = (
                interm if self.state == VesselState.LOADING_AT_INTERM else 'D'
            )
            return (
                f"loading at {point}: v_anchor = {self.anchor_load_rate}, "
                f"n = {self.n_loaded}/{self.max_capacity}"
            )
        elif (
            self.state == VesselState.UNLOADING_AT_INTERM or
            self.state == VesselState.UNLOADING_AT_DEST
        ):
            point = (
                interm if self.state == VesselState.UNLOADING_AT_INTERM else 'D'
            )
            return (
                f"unloading at {point}: v_anchor = {self.anchor_load_rate}, "
                f"n = {self.n_loaded}/{self.max_capacity}"
            )
        else:
            if self.state == VesselState.SHIPPING_FROM_INTERM_TO_DEST:
                departure_point = interm
                arrival_point = 'D'
            else:
                departure_point = 'D'
                arrival_point = interm

            return (
                f"shipping from {departure_point} to {arrival_point}: t = "
                f"{self.transition_time}, n = {self.n_loaded}/{self.max_capacity}"
            )


class SrcTransportHandler:

    def __init__(
        self,
        max_transportation_time,
        get_transportation_time,
    ):
        self.get_transportation_time = get_transportation_time

        self.n_unloaded_at_src_port = 0

        self.schedule_A = [0] * max_transportation_time
        self.schedule_B = [0] * max_transportation_time
        self.schedule_C = [0] * max_transportation_time
        self.schedule_unloaded = [0] * max_transportation_time

    def handle(self, env: SimulationEnvironment):
        env.n_unloaded_at_src_port += self.n_unloaded_at_src_port
        self.n_unloaded_at_src_port = 0

        if env.period == 0:
            month, day = env.month, env.day
            days_in_month = MONTH_DAYS[month]

            main_monthly_prod_rate = env.cfg.main_monthly_prod_rate[month]
            alt_monthly_prod_rate = env.cfg.alt_monthly_prod_rate[month]

            main_daily_load = main_monthly_prod_rate // days_in_month
            remainder = main_monthly_prod_rate - days_in_month * main_daily_load
            if day + remainder > days_in_month:
                main_daily_load += 1  # distribute remainder across the days

            # NOTE: when alt_monthly_prod_rate is 0, remainder is 0 as well
            alt_daily_load = alt_monthly_prod_rate // days_in_month
            remainder = alt_monthly_prod_rate - days_in_month * alt_daily_load
            if day + remainder > days_in_month:
                alt_daily_load += 1  # distribute remainder across the days

            n_load = main_daily_load + alt_daily_load

            env.n_unloaded_at_src -= n_load
            # NOTE: we want to have enough containers for production at the beginning of
            # each day before accounting for possible returns
            if env.n_unloaded_at_src < 0:
                return  # run failed

            n_loaded_A = round(env.cfg.main_share_A * main_daily_load)
            n_loaded_B = main_daily_load - n_loaded_A
            n_loaded_C = alt_daily_load

            direct = True
            transportation_time = self.get_transportation_time(env, direct)
            # NOTE: containers will be fully loaded at the end of the day
            transportation_time += env.cfg.periods_in_day - 1

            self.schedule_A[transportation_time] += n_loaded_A
            self.schedule_B[transportation_time] += n_loaded_B
            self.schedule_C[transportation_time] += n_loaded_C

            direct = False
            transportation_time = self.get_transportation_time(env, direct)

            self.schedule_unloaded[transportation_time] += env.n_unloaded_at_src_port
            env.n_unloaded_at_src_port = 0

        env.n_loaded_at_src_port_A += self.schedule_A[0]
        env.n_loaded_at_src_port_B += self.schedule_B[0]
        env.n_loaded_at_src_port_C += self.schedule_C[0]
        env.n_unloaded_at_src += self.schedule_unloaded[0]

        for i in range(len(self.schedule_A) - 1):
            self.schedule_A[i] = self.schedule_A[i + 1]
            self.schedule_B[i] = self.schedule_B[i + 1]
            self.schedule_C[i] = self.schedule_C[i + 1]
            self.schedule_unloaded[i] = self.schedule_unloaded[i + 1]

        self.schedule_A[-1] = 0
        self.schedule_B[-1] = 0
        self.schedule_C[-1] = 0
        self.schedule_unloaded[-1] = 0


class DestTransportHandler:

    def __init__(
        self,
        max_transportation_time,
        get_transportation_time,
    ):
        self.get_transportation_time = get_transportation_time

        self.loaded_A = 0
        self.loaded_B = 0
        self.loaded_C = 0

        self.schedule_A = [0] * max_transportation_time
        self.schedule_B = [0] * max_transportation_time
        self.schedule_C = [0] * max_transportation_time
        self.schedule_unloaded = [0] * max_transportation_time

    def handle(self, env: SimulationEnvironment):
        env.n_loaded_at_dest_port_A += self.loaded_A
        env.n_loaded_at_dest_port_B += self.loaded_B
        env.n_loaded_at_dest_port_C += self.loaded_C

        self.loaded_A = 0
        self.loaded_B = 0
        self.loaded_C = 0

        if env.period == 0:
            direct = True
            transportation_time = self.get_transportation_time(env, direct)

            self.schedule_A[transportation_time] += env.n_loaded_at_dest_port_A
            self.schedule_B[transportation_time] += env.n_loaded_at_dest_port_B
            self.schedule_C[transportation_time] += env.n_loaded_at_dest_port_C

            env.n_loaded_at_dest_port_A = 0
            env.n_loaded_at_dest_port_B = 0
            env.n_loaded_at_dest_port_C = 0

            direct = False
            transportation_time = self.get_transportation_time(env, direct)

            self.schedule_unloaded[transportation_time] += env.n_unloaded_at_dest
            env.n_unloaded_at_dest = 0

        env.n_delivered_A += self.schedule_A[0]
        env.n_delivered_B += self.schedule_B[0]
        env.n_delivered_C += self.schedule_C[0]
        env.n_unloaded_at_dest_port += self.schedule_unloaded[0]

        env.n_unloaded_at_dest += (
            self.schedule_A[0] + self.schedule_B[0] + self.schedule_C[0]
        )

        for i in range(len(self.schedule_A) - 1):
            self.schedule_A[i] = self.schedule_A[i + 1]
            self.schedule_B[i] = self.schedule_B[i + 1]
            self.schedule_C[i] = self.schedule_C[i + 1]
            self.schedule_unloaded[i] = self.schedule_unloaded[i + 1]

        self.schedule_A[-1] = 0
        self.schedule_B[-1] = 0
        self.schedule_C[-1] = 0
        self.schedule_unloaded[-1] = 0


class IntermTransportHandler:

    def __init__(
        self,
        max_transportation_time,
        get_transportation_time,
    ):
        self.get_transportation_time = get_transportation_time

        self.n_loaded_at_A = 0
        self.n_unloaded_at_A = 0

        self.n_loaded_at_B = 0
        self.n_unloaded_at_P = 0

        self.schedule_loaded = [0] * max_transportation_time
        self.schedule_unloaded = [0] * max_transportation_time

    def handle(self, env: SimulationEnvironment):
        env.n_loaded_at_A += self.n_loaded_at_A
        env.n_unloaded_at_A += self.n_unloaded_at_A

        self.n_loaded_at_A = 0
        self.n_unloaded_at_A = 0

        env.n_loaded_at_B += self.n_loaded_at_B
        env.n_unloaded_at_P += self.n_unloaded_at_P

        self.n_loaded_at_B = 0
        self.n_unloaded_at_P = 0

        if env.period == 0:
            direct = True
            transportation_time = self.get_transportation_time(env, direct)

            self.schedule_loaded[transportation_time] += env.n_loaded_at_B
            env.n_loaded_at_B = 0

            direct = False
            transportation_time = self.get_transportation_time(env, direct)

            self.schedule_unloaded[transportation_time] += env.n_unloaded_at_P
            env.n_unloaded_at_P = 0

        env.n_loaded_at_P += self.schedule_loaded[0]
        env.n_unloaded_at_B += self.schedule_unloaded[0]

        for i in range(len(self.schedule_loaded) - 1):
            self.schedule_loaded[i] = self.schedule_loaded[i + 1]
            self.schedule_unloaded[i] = self.schedule_unloaded[i + 1]

        self.schedule_loaded[-1] = 0
        self.schedule_unloaded[-1] = 0


def get_feeder_wait_time(
    env: SimulationEnvironment, vessel_id: str, state: VesselState
):
    # no reference; just some values
    return env.rng.choice([1, 2], p=[0.75, 0.25])

def get_feeder_shipping_time(
    env: SimulationEnvironment, vessel_id: str, state: VesselState
):
    route_id = vessel_id[-1]
    if route_id in ('A', 'B'):
        if env.month in WINTER_MONTHS:
            shape = 0.1015187340040396
            scale = 17.467035830073918
        else:
            shape = 0.0736504756651557
            scale = 15.105380789572115

        periods = stat.lognorm.rvs(
            shape, loc=0.0, scale=scale, random_state=env.rng
        )

        # NOTE: 13 x 8h periods is the estimate for a physical lower bound
        shipping_time = max(13, int(round(periods)))

        if route_id == 'B':
            # no reference; just some values
            shipping_time += env.rng.choice([3, 4], p=[0.2, 0.8])

        return shipping_time

    else:  # route_id == 'C'
        if not hasattr(get_feeder_shipping_time, "probs"):
            # no reference; just some values
            min_time = 68
            max_time = 84
            mean_time = 74

            values = np.arange(min_time, max_time + 1)
            probs = stat.norm.pdf(values, loc=mean_time, scale=1.0)

            get_feeder_shipping_time.values = values
            get_feeder_shipping_time.probs = (
                probs / probs.sum()
            ).tolist()

        values = get_feeder_shipping_time.values
        probs = get_feeder_shipping_time.probs

        return env.rng.choice(values, p=probs)

def get_feeder_shipping_time_backup(
    env: SimulationEnvironment, vessel_id: str, state: VesselState
):
    if state == VesselState.LOADING_AT_DEST:
        if not hasattr(get_feeder_shipping_time_backup, "probs"):
            # no reference; just some values
            min_time = 125
            max_time = 148
            mean_time = 135

            values = np.arange(min_time, max_time + 1)
            probs = stat.norm.pdf(values, loc=mean_time, scale=1.0)

            get_feeder_shipping_time_backup.values = values
            get_feeder_shipping_time_backup.probs = (
                probs / probs.sum()
            ).tolist()

        values = get_feeder_shipping_time_backup.values
        probs = get_feeder_shipping_time_backup.probs

        return env.rng.choice(values, p=probs)

    elif state == VesselState.WAITING_AT_INTERM:
        if env.month in WINTER_MONTHS:
            shape = 0.1015187340040396
            scale = 17.467035830073918
        else:
            shape = 0.0736504756651557
            scale = 15.105380789572115

        periods = stat.lognorm.rvs(
            shape, loc=0.0, scale=scale, random_state=env.rng
        )

        # NOTE: return using route A
        return max(13, int(round(periods)))

    else:
        raise ValueError(f"Unexpected {state = }")

def get_feeder_load_rate(
    env: SimulationEnvironment, vessel_id: str, state: VesselState
):
    # NOTE: sample small deviation when anchor load rate is already sampled
    for vessel in env.feeder_vessels:
        if vessel.id == vessel_id and vessel.anchor_load_rate > 0:
            return min(max(env.rng.uniform(0.9, 1.1), 0.96), 1.04)

    route_id = vessel_id[-1]
    if (
        state == VesselState.LOADING_AT_SRC or
        state == VesselState.UNLOADING_AT_SRC
    ):
        if state == VesselState.LOADING_AT_SRC:
            if route_id == 'A':
                n_avail = env.n_loaded_at_src_port_A
            elif route_id == 'B':
                n_avail = env.n_loaded_at_src_port_B
            elif route_id == 'C':
                n_avail = env.n_loaded_at_src_port_C

            n_to_load = 0

            if route_id in ('A', 'B') and env.cfg.disable_underloaded_S2I:
                k = env.cfg.underload_share_S2I
            elif route_id == 'C' and env.cfg.disable_underloaded_S2D:
                k = env.cfg.underload_share_S2D
            else:
                k = 0.0

            for vessel in env.feeder_vessels:
                if (
                    vessel.id[-1] == route_id and
                    vessel.state == VesselState.LOADING_AT_SRC and
                    vessel.anchor_load_rate > 0
                ):
                    capacity = vessel.max_capacity
                    n_loaded = vessel.n_loaded
                    if k > 0.0:
                        n_required = round(k * capacity)
                        n_to_load += (
                            min(capacity - n_loaded, vessel.anchor_load_rate)
                            if n_loaded > n_required
                            else n_required - n_loaded
                        )
                    else:
                        n_to_load += min(capacity - n_loaded, vessel.anchor_load_rate)

            if n_avail < n_to_load:
                return 0

        if not hasattr(get_feeder_load_rate, "src_winter_probs"):
            hours_in_period = 24 // env.cfg.periods_in_day

            # no reference, just some values
            width = 3  # [6, 7, 8]
            one_hour_probs = [0.25, 0.5, 0.25]

            probs = [1.0]
            for _ in range(hours_in_period):
                new_probs = [0] * (len(probs) + width - 1)
                for i, count_i in enumerate(probs):
                    if count_i == 0:
                        continue
                    for j, count_j in enumerate(one_hour_probs):
                        new_probs[i + j] += count_i * count_j
                probs = new_probs

            # NOTE: should check if sum(probs) == 1.0, but it is ok
            get_feeder_load_rate.src_winter_vals = np.arange(
                6 * hours_in_period, 8 * hours_in_period + 1
            )
            get_feeder_load_rate.src_winter_probs = copy(probs)

            # =============================================================== #

            width = 5  # [8, 9, 10, 11, 12]
            one_hour_probs = [0.15, 0.2, 0.3, 0.2, 0.15]

            probs = [1.0]
            for _ in range(hours_in_period):
                new_probs = [0] * (len(probs) + width - 1)
                for i, count_i in enumerate(probs):
                    if count_i == 0:
                        continue
                    for j, count_j in enumerate(one_hour_probs):
                        new_probs[i + j] += count_i * count_j
                probs = new_probs

            get_feeder_load_rate.src_summer_vals = np.arange(
                8 * hours_in_period, 12 * hours_in_period + 1
            )
            get_feeder_load_rate.src_summer_probs = copy(probs)

        if env.month in WINTER_MONTHS:
            values = get_feeder_load_rate.src_winter_vals
            probs = get_feeder_load_rate.src_winter_probs
        else:
            values = get_feeder_load_rate.src_summer_vals
            probs = get_feeder_load_rate.src_summer_probs

        return env.rng.choice(values, p=probs)

    elif (
        state == VesselState.LOADING_AT_INTERM or
        state == VesselState.UNLOADING_AT_INTERM
    ):
        if state == VesselState.LOADING_AT_INTERM:
            n_avail = (
                env.n_unloaded_at_A if route_id == 'A' else
                env.n_unloaded_at_B
            )
            n_to_load = 0

            if env.cfg.disable_underloaded_I2S:
                k = env.cfg.underload_share_I2S
            else:
                k = 0.0

            for vessel in env.feeder_vessels:
                if (
                    vessel.id[-1] == route_id and
                    vessel.state == VesselState.LOADING_AT_INTERM and
                    vessel.anchor_load_rate > 0
                ):
                    capacity = vessel.max_capacity
                    n_loaded = vessel.n_loaded
                    if k > 0.0:
                        n_required = round(k * capacity)
                        n_to_load += (
                            min(capacity - n_loaded, vessel.anchor_load_rate)
                            if n_loaded > n_required
                            else n_required - n_loaded
                        )
                    else:
                        n_to_load += min(capacity - n_loaded, vessel.anchor_load_rate)

            if n_avail < n_to_load:
                return 0

        if not hasattr(get_feeder_load_rate, "interm_probs"):
            # no reference; just some values
            min_rate = 103
            max_rate = 110
            mean_rate = 106.67

            values = np.arange(min_rate, max_rate + 1)
            probs = stat.norm.pdf(values, loc=mean_rate, scale=1.0)

            get_feeder_load_rate.interm_vals = values
            get_feeder_load_rate.interm_probs = (probs / probs.sum()).tolist()

        return env.rng.choice(
            get_feeder_load_rate.interm_vals, p=get_feeder_load_rate.interm_probs
        )

    elif (
        state == VesselState.LOADING_AT_DEST or
        state == VesselState.UNLOADING_AT_DEST
    ):
        if state == VesselState.LOADING_AT_DEST:
            n_avail = env.n_unloaded_at_dest_port
            n_to_load = 0

            if env.cfg.disable_underloaded_D2S:
                k = env.cfg.underload_share_D2S
            else:
                k = 0.0

            for vessel in env.feeder_vessels:
                if (
                    vessel.state == VesselState.LOADING_AT_DEST and
                    vessel.anchor_load_rate > 0
                ):
                    capacity = vessel.max_capacity
                    n_loaded = vessel.n_loaded
                    if k > 0.0:
                        n_required = round(k * capacity)
                        n_to_load += (
                            min(capacity - n_loaded, vessel.anchor_load_rate)
                            if n_loaded > n_required
                            else n_required - n_loaded
                        )
                    else:
                        n_to_load += min(capacity - n_loaded, vessel.anchor_load_rate)

            if env.cfg.disable_underloaded_D2I:
                k = env.cfg.underload_share_D2I
            else:
                k = 0.0

            for vessel in env.mainland_vessels:
                if (
                    vessel.state == VesselState.LOADING_AT_DEST and
                    vessel.anchor_load_rate > 0
                ):
                    capacity = vessel.max_capacity
                    n_loaded = vessel.n_loaded
                    if k > 0.0:
                        n_required = round(k * capacity)
                        n_to_load += (
                            min(capacity - n_loaded, vessel.anchor_load_rate)
                            if n_loaded > n_required
                            else n_required - n_loaded
                        )
                    else:
                        n_to_load += min(capacity - n_loaded, vessel.anchor_load_rate)

            if n_avail < n_to_load:
                return 0

        # NOTE: using the same values as in loading/unloaing at intermidiate point
        if not hasattr(get_feeder_load_rate, "interm_probs"):
            # no reference; just some values
            min_rate = 103
            max_rate = 109
            mean_rate = 106.67

            values = np.arange(min_rate, max_rate + 1)
            probs = stat.norm.pdf(values, loc=mean_rate, scale=1.0)

            get_feeder_load_rate.interm_vals = values
            get_feeder_load_rate.interm_probs = (probs / probs.sum()).tolist()

        return env.rng.choice(
            get_feeder_load_rate.interm_vals, p=get_feeder_load_rate.interm_probs
        )

    else:
        raise ValueError(f"Unexpected {state = }")


def get_mainland_wait_time(env: SimulationEnvironment, vessel_id: str, state: VesselState):
    # no reference; just some values
    return env.rng.choice([1, 2], p=[0.7, 0.3])

def get_mainland_shipping_time(env: SimulationEnvironment, vessel_id: str, state: VesselState):
    if not hasattr(get_mainland_load_rate, "probs_A"):
        # no reference; just some values
        min_time = 116
        max_time = 136
        mean_time = 122.67

        values = np.arange(min_time, max_time + 1)
        probs = stat.norm.pdf(values, loc=mean_time, scale=1.0)

        get_mainland_load_rate.vals_A = values
        get_mainland_load_rate.probs_A = (probs / probs.sum()).tolist()

        # =================================================================== #
        min_time = 112
        max_time = 132
        mean_time = 118.67

        values = np.arange(min_time, max_time + 1)
        probs = stat.norm.pdf(values, loc=mean_time, scale=1.0)

        get_mainland_load_rate.vals_B = values
        get_mainland_load_rate.probs_B = (probs / probs.sum()).tolist()

    route_id = vessel_id[-1]

    if route_id == 'A':
        values = get_mainland_load_rate.vals_A
        probs = get_mainland_load_rate.probs_A
    else:
        values = get_mainland_load_rate.vals_B
        probs = get_mainland_load_rate.probs_B

    return env.rng.choice(values, p=probs)

def get_mainland_load_rate(
    env: SimulationEnvironment, vessel_id: str, state: VesselState
):
    # NOTE: sample small deviation when anchor load rate is already sampled
    for vessel in env.mainland_vessels:
        if vessel.id == vessel_id and vessel.anchor_load_rate > 0:
            return min(max(env.rng.uniform(0.95, 1.05), 0.96875), 1.03125)

    if state not in (
        VesselState.LOADING_AT_INTERM,
        VesselState.UNLOADING_AT_INTERM,
        VesselState.LOADING_AT_DEST,
        VesselState.UNLOADING_AT_DEST
    ):
        raise ValueError(f"Unexpected {state = }")

    route_id = vessel_id[-1]

    if state == VesselState.LOADING_AT_INTERM:
        n_avail = (
            env.n_loaded_at_A if route_id == 'A' else
            env.n_loaded_at_P
        )
        n_to_load = 0

        if env.cfg.disable_underloaded_I2D:
            k = env.cfg.underload_share_I2D
        else:
            k = 0.0

        for vessel in env.mainland_vessels:
            if (
                vessel.id[-1] == route_id and
                vessel.state == VesselState.LOADING_AT_INTERM and
                vessel.anchor_load_rate > 0
            ):
                capacity = vessel.max_capacity
                n_loaded = vessel.n_loaded
                if k > 0.0:
                    n_required = round(k * capacity)
                    n_to_load += (
                        min(capacity - n_loaded, vessel.anchor_load_rate)
                        if n_loaded > n_required
                        else n_required - n_loaded
                    )
                else:
                    n_to_load += min(capacity - n_loaded, vessel.anchor_load_rate)

        if n_avail < n_to_load:
            return 0

    if state == VesselState.LOADING_AT_DEST:
        n_avail = env.n_unloaded_at_dest_port
        n_to_load = 0

        if env.cfg.disable_underloaded_D2S:
            k = env.cfg.underload_share_D2S
        else:
            k = 0.0

        for vessel in env.feeder_vessels:
            if (
                vessel.state == VesselState.LOADING_AT_DEST and
                vessel.anchor_load_rate > 0
            ):
                capacity = vessel.max_capacity
                n_loaded = vessel.n_loaded
                if k > 0.0:
                    n_required = round(k * capacity)
                    n_to_load += (
                        min(capacity - n_loaded, vessel.anchor_load_rate)
                        if n_loaded > n_required
                        else n_required - n_loaded
                    )
                else:
                    n_to_load += min(capacity - n_loaded, vessel.anchor_load_rate)

        if env.cfg.disable_underloaded_D2I:
            k = env.cfg.underload_share_D2I
        else:
            k = 0.0

        for vessel in env.mainland_vessels:
            if (
                vessel.id[-1] == route_id and
                vessel.state == VesselState.LOADING_AT_DEST and
                vessel.anchor_load_rate > 0
            ):
                capacity = vessel.max_capacity
                n_loaded = vessel.n_loaded
                if k > 0.0:
                    n_required = round(k * capacity)
                    n_to_load += (
                        min(capacity - n_loaded, vessel.anchor_load_rate)
                        if n_loaded > n_required
                        else n_required - n_loaded
                    )
                else:
                    n_to_load += min(capacity - n_loaded, vessel.anchor_load_rate)

        if n_avail < n_to_load:
            return 0

    if not hasattr(get_mainland_load_rate, "interm_probs"):
        # no reference; just some values
        min_rate = 155
        max_rate = 165
        mean_rate = 160

        values = np.arange(min_rate, max_rate + 1)
        probs = stat.norm.pdf(values, loc=mean_rate, scale=1.0)

        get_mainland_load_rate.interm_vals = values
        get_mainland_load_rate.interm_probs = (probs / probs.sum()).tolist()

    return env.rng.choice(
        get_mainland_load_rate.interm_vals, p=get_mainland_load_rate.interm_probs
    )


def get_src_transportation_time(env: SimulationEnvironment, direct: bool):
    return env.cfg.periods_in_day

def get_dest_transportation_time(env: SimulationEnvironment, direct: bool):
    if not hasattr(get_dest_transportation_time, "probs"):
        # no reference; just some values
        min_rate = 10 * env.cfg.periods_in_day
        max_rate = 14 * env.cfg.periods_in_day
        mean_rate = 12 * env.cfg.periods_in_day

        values = np.arange(min_rate, max_rate + 1)
        probs = stat.norm.pdf(values, loc=mean_rate, scale=1.0)

        get_dest_transportation_time.vals = values
        get_dest_transportation_time.probs = (probs / probs.sum()).tolist()

    if direct:
        return env.rng.choice(
            get_dest_transportation_time.vals, p=get_dest_transportation_time.probs
        )
    else:
        return 5 * env.cfg.periods_in_day

def get_interm_transportation_time(env: SimulationEnvironment, direct: bool):
    return 6 * env.cfg.periods_in_day


def setup_fleet(env: SimulationEnvironment):
    """Dynamically initializes vessels and staggers their departures."""

    # Route A Feeders
    if env.cfg.n_feeders_A > 0:
        stagger = env.cfg.rt_feeder_A // env.cfg.n_feeders_A
        for i in range(env.cfg.n_feeders_A):
            wait_time = env.cfg.base_wait_feeder_A + (i * stagger)
            env.feeder_vessels.append(
                FeederVessel_Main(
                    f'feeder_{i+1}:A',
                    MAX_FEEDER_VESSEL_CAPACITY,
                    wait_time,
                    get_feeder_wait_time,
                    get_feeder_load_rate,
                    get_feeder_shipping_time,
                )
            )

    # Route B Feeders
    if env.cfg.n_feeders_B > 0:
        stagger = env.cfg.rt_feeder_B // env.cfg.n_feeders_B
        for i in range(env.cfg.n_feeders_B):
            wait_time = env.cfg.base_wait_feeder_B + (i * stagger)
            env.feeder_vessels.append(
                FeederVessel_Main(
                    f'feeder_{i+1}:B',
                    MAX_FEEDER_VESSEL_CAPACITY,
                    wait_time,
                    get_feeder_wait_time,
                    get_feeder_load_rate,
                    get_feeder_shipping_time,
                )
            )

    # Route A Mainlands
    if env.cfg.n_mainlands_A > 0:
        stagger = env.cfg.rt_mainland_A // env.cfg.n_mainlands_A
        for i in range(env.cfg.n_mainlands_A):
            wait_time = env.cfg.base_wait_mainland_A + (i * stagger)
            env.mainland_vessels.append(
                MainlandVessel(
                    f'mainland_{i+1}:A',
                    MAX_MOTHER_VESSEL_CAPACITY,
                    wait_time,
                    get_mainland_wait_time,
                    get_mainland_load_rate,
                    get_mainland_shipping_time,
                )
            )

    # Route B Mainlands
    if env.cfg.n_mainlands_B > 0:
        stagger = env.cfg.rt_mainland_B // env.cfg.n_mainlands_B
        for i in range(env.cfg.n_mainlands_B):
            wait_time = env.cfg.base_wait_mainland_B + (i * stagger)
            env.mainland_vessels.append(
                MainlandVessel(
                    f'mainland_{i+1}:B',
                    MAX_MOTHER_VESSEL_CAPACITY,
                    wait_time,
                    get_mainland_wait_time,
                    get_mainland_load_rate,
                    get_mainland_shipping_time,
                )
            )

    # Interflood Recovery (Surge Feeders)
    periods_in_day = env.cfg.periods_in_day
    interflood_end_wait_time = env.days_until_interflood_end * periods_in_day

    if env.cfg.n_surge_feeders_A > 0:
        # stagger = env.cfg.rt_feeder_A // env.cfg.n_surge_feeders_A
        wait_time = interflood_end_wait_time
        for i in range(env.cfg.n_surge_feeders_A):
            j = env.cfg.n_feeders_A + i + 1
            env.feeder_vessels.append(
                FeederVessel_Main(
                    f'feeder_{j}:A',
                    MAX_FEEDER_VESSEL_CAPACITY,
                    wait_time,
                    get_feeder_wait_time,
                    get_feeder_load_rate,
                    get_feeder_shipping_time,
                )
            )

    if env.cfg.n_surge_feeders_B > 0:
        # stagger = env.cfg.rt_feeder_B // env.cfg.n_surge_feeders_B
        wait_time = interflood_end_wait_time
        for i in range(env.cfg.n_surge_feeders_B):
            j = env.cfg.n_feeders_B + i + 1
            env.feeder_vessels.append(
                FeederVessel_Main(
                    f'feeder_{j}:B',
                    MAX_FEEDER_VESSEL_CAPACITY,
                    wait_time,
                    get_feeder_wait_time,
                    get_feeder_load_rate,
                    get_feeder_shipping_time,
                )
            )

    # Route C Feeders
    alt_route_wait_time = env.days_until_alt_route_start * periods_in_day

    if env.cfg.n_feeders_C > 0:
        stagger = env.cfg.rt_feeder_C // env.cfg.n_feeders_C
        for i in range(env.cfg.n_feeders_C):
            wait_time = alt_route_wait_time + (i * stagger)
            env.feeder_vessels.append(
                FeederVessel_Alt(
                    f'feeder_{i+1}:C',
                    MAX_FEEDER_VESSEL_CAPACITY,
                    wait_time,
                    get_feeder_wait_time,
                    get_feeder_load_rate,
                    get_feeder_shipping_time,
                    get_feeder_shipping_time_backup,
                )
            )

    # Transportation handlers
    env.src_transport_handler = SrcTransportHandler(
        max_transportation_time=16, get_transportation_time=get_src_transportation_time
    )
    env.dest_transport_handler = DestTransportHandler(
        max_transportation_time=48, get_transportation_time=get_dest_transportation_time
    )
    env.interm_transport_handler = IntermTransportHandler(
        max_transportation_time=24, get_transportation_time=get_interm_transportation_time
    )


def run_single_simulation(args):
    cfg, seed = args

    env = SimulationEnvironment(cfg)
    setup_fleet(env)

    ret_code = env.simulate(seed=seed)

    n_delivered_A = env.n_delivered_A + sum(env.dest_transport_handler.schedule_A)
    n_delivered_B = env.n_delivered_B + sum(env.dest_transport_handler.schedule_B)
    n_delivered_C = env.n_delivered_C + sum(env.dest_transport_handler.schedule_C)

    return {
        'seed': seed,
        'failed': ret_code != 0,
        'failed_during_interflood': (
            ret_code != 0 and
            env.days_until_interflood_start <= 0 and
            env.days_until_interflood_end > 0
        ),
        'min_containers': env.min_unloaded_at_src,
        'total_delivered': (
            n_delivered_A + n_delivered_B + n_delivered_C
        ),
        'delivered_A': n_delivered_A,
        'delivered_B': n_delivered_B,
        'delivered_C': n_delivered_C,
        'triangulated': env.n_triangulated,
    }


def evaluate_config(cfg: SimulationConfig, n_sims: int):
    """Runs Monte Carlo for a specific fleet cfguration and returns summary stats."""

    # generate unique seeds
    seed_seq = np.random.SeedSequence(42)
    seeds = seed_seq.spawn(n_sims)

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_single_simulation, (cfg, seed))
            for seed in seeds
        ]
        for future in tqdm(
            as_completed(futures),
            total=n_sims,
            unit="sim",
        ):
            results.append(future.result())

    df = pd.DataFrame(results)

    return {
        'n_containers': cfg.n_containers,

        'n_main_A': cfg.n_mainlands_A,
        'n_main_B': cfg.n_mainlands_B,

        'n_feeder_A': f'{cfg.n_feeders_A}+{cfg.n_surge_feeders_A}',
        'n_feeder_B': f'{cfg.n_feeders_B}+{cfg.n_surge_feeders_B}',

        'failure_rate': df['failed'].mean(),
        'interflood_failure_rate': df['failed_during_interflood'].mean(),

        'avg_delivered': f"{df['total_delivered'].mean():.2f} +/- {df['total_delivered'].std():.2f}",
        # 'avg_delivered*': df['total_delivered'][~df['failed']].mean(),

        # 5-th percentile represents the "worst case" inventory drop across 95% of scenarios
        # 'worst_case_95%*': np.percentile(df['min_containers'][~df['failed']], 5),
        'n_estimated_95%': cfg.n_containers - np.percentile(df['min_containers'], 5),
        'n_estimated_98%': cfg.n_containers - np.percentile(df['min_containers'], 2),

        # 'S2I': cfg.underload_share_S2I, 'I2D': cfg.underload_share_I2D,
        'triangulated': f"{df['triangulated'].mean():.2f} +/- {df['triangulated'].std():.2f}"
    }


if __name__ == '__main__':
    cfg = SimulationConfig(
        simulation_period=2200,
        periods_in_day=3,
        start_month='February', start_day=2,
        interflood_start=('May', 10),
        interflood_end=('June', 30),
        alt_route_start=('July', 1),
        alt_route_end=('November', 10),
        n_containers=27_500,
        n_loaded_A=600, n_loaded_B=600,
        main_monthly_prod_rate=([3000] * 12),
        alt_monthly_prod_rate=[0, 0, 0, 0, 0, 666, 666, 667, 667, 667, 667, 0],

        n_feeders_A=3, n_feeders_B=2, n_feeders_C=4,
        n_mainlands_A=6, n_mainlands_B=4,
        n_surge_feeders_A=0, n_surge_feeders_B=0,

        # if we disable it, then everything collapses after interflood ends...
        disable_underloaded_I2S=False,

        underload_share_I2D=0.9,
        underload_share_S2I=0.5,
        underload_share_S2D=0.9,

        triangulation_n_sims=0,
    )

    print(evaluate_config(cfg, n_sims=1000))
    # ---- OR ---- #
    # env = SimulationEnvironment(cfg)
    # setup_fleet(env)
    # with open("Logs.txt", 'w') as f_out:
    #     env.simulate(
    #         seed=0,
    #         output_handle=f_out,
    #         verbose_start=('February', 2),
    #         verbose_end=None,
    #         print_final_stats=True,
    #     )
