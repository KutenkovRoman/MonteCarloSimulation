import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

MAX_FEEDER_VESSEL_CAPACITY = 660
MAX_MOTHER_VESSEL_CAPACITY = 1200

def route_A_local_uniform(month: int, day: int):
    # according to excel table it takes 5.44 days from S to A
    return np.random.randint(6, 9)

def route_A_global_uniform(month: int, day: int):
    # according to my own estimations global region of route A is ~13k miles
    return np.random.randint(38, 46)

def route_B_local_uniform(month: int, day: int):
    # according to excel table it takes 6.73 days from S to B
    return np.random.randint(7, 10)

def route_B_global_uniform(month: int, day: int):
    # according to my own estimations global region of route B is ~12.5k miles
    return np.random.randint(37, 45)

def route_B_transportation_const(month: int, day: int):
    # from excel table
    return 6

def route_C_uniform(month: int, day: int):
    # my own estimations closely matches with length for route C, ~6k miles
    return np.random.randint(23, 31)

MONTH_NAMES = [
    'December', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November'
]
MONTH_DAYS = [
    31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30
]

def in_period(month: int, day: int, period_start: tuple[str, int], period_end: tuple[str, int]):
    start_month_name, start_day = period_start
    assert start_month_name in MONTH_NAMES, (
        f"{start_month_name = } does not exist, check your calendar"
    )
    start_month = MONTH_NAMES.index(start_month_name)

    days_in_month = MONTH_DAYS[start_month]
    assert 0 < start_day <= days_in_month, (
        f"{start_day = } must be integer from 1 to {days_in_month} "
        f"(# of days in {start_month_name})"
    )

    end_month_name, end_day = period_end
    assert end_month_name in MONTH_NAMES, (
        f"{end_month_name = } does not exist, check your calendar"
    )
    end_month = MONTH_NAMES.index(end_month_name)

    days_in_month = MONTH_DAYS[end_month]
    assert 0 < end_day <= days_in_month, (
        f"{end_day = } must be integer from 1 to {days_in_month} "
        f"(# of days in {end_month_name})"
    )

    assert (start_month == end_month and start_day < end_day) or (start_month < end_month)

    return (
        ((month == start_month and day >= start_day) or (month > start_month)) and
        ((month == end_month and day <= end_day) or (month < end_month))
    )


def days_until(month: int, day: int, period_boundary: tuple[str, int]):
    boundary_month_name, boundary_day = period_boundary
    boundary_month = MONTH_NAMES.index(boundary_month_name)
    days_in_month = MONTH_DAYS[month]

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

    # Initialization
    start_month: int | str
    start_day: int

    interflood_start: tuple[str, int]
    interflood_end: tuple[str, int]

    alt_route_start: tuple[str, int]
    alt_route_end: tuple[str, int]

    n_containers: int
    monthly_prod_rate: list[int]

    # Fleet sizes
    n_feeders_A: int
    n_feeders_B: int
    n_feeders_C: int
    n_mainlands_A: int
    n_mainlands_B: int

    n_loaded_for_A: int = 0
    n_loaded_for_B: int = 0
    n_loaded_for_C: int = 0

    main_share_A: float = 0.6  # 60% of main traffic goes through route A
    alt_share: float = 0.1     # 10% of all trafic goes through route C

    route_A_local_distr: Callable[[int, int], int] = route_A_local_uniform
    route_A_global_distr: Callable[[int, int], int] = route_A_global_uniform
    route_B_local_distr: Callable[[int, int], int] = route_B_local_uniform
    route_B_global_distr: Callable[[int, int], int] = route_B_global_uniform
    route_B_transportation_distr: Callable[[int, int], int] = route_B_transportation_const
    route_C_distr: Callable[[int, int], int] = route_C_uniform

    max_feeder_vessel_capacity: int = MAX_FEEDER_VESSEL_CAPACITY
    max_mainland_vessel_capacity: int = MAX_MOTHER_VESSEL_CAPACITY

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

    # Surge capacity (vessels that only activate after interflood ends)
    n_post_interflood_A: int = 0
    n_post_interflood_B: int = 0

    # Base wait times (days before the FIRST vessel of that type departs)
    base_wait_feeder_A: int = 10
    base_wait_feeder_B: int = 15
    base_wait_mainland_A: int = 16  # base_wait_feeder_A + feeder_A
    base_wait_mainland_B: int = 28  # base_wait_feeder_B + feeder_B + railway

    # Round trip times (used to calculate staggering intervals)
    rt_feeder_A: int = 12   # feeder_A * 2 (feeder_A = 6 days)
    rt_feeder_B: int = 14   # feeder_B * 2 (feeder_B = 7 days)
    rt_feeder_C: int = 48   # feeder_C * 2 (feeder_C = ~24 days)
    rt_mainland_A: int = 76 # mainland_A * 2 (mainland_A = 38 days)
    rt_mainland_B: int = 74 # mainland_A * 2 (mainland_B = 37 days)


class SimulationEnvironment:

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg  # save for later uses

        n_loaded = cfg.n_loaded_for_A + cfg.n_loaded_for_B + cfg.n_loaded_for_C
        if n_loaded > cfg.n_containers:
            raise ValueError(
                f"Total number of containers loaded at source is greater than the "
                f"number of available containers ({n_loaded} > {cfg.n_containers})"
            )

        # Environment variables
        self.n_unloaded_at_source = cfg.n_containers - n_loaded
        self.n_loaded_at_source_for_A = cfg.n_loaded_for_A
        self.n_loaded_at_source_for_B = cfg.n_loaded_for_B
        self.n_loaded_at_source_for_C = cfg.n_loaded_for_C

        self.n_loaded_at_A = 0
        self.n_unloaded_at_A = 0

        self.n_loaded_at_P = 0
        self.n_unloaded_at_B = 0

        self.n_unloaded_at_destination = 0
        self.n_delivered_from_A = 0
        self.n_delivered_from_B = 0
        self.n_delivered_from_C = 0

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

        # TODO: add check for alternative route, does not break, but init won't work as intended
        if self.in_period(cfg.interflood_start, cfg.interflood_end):
            raise NotImplementedError(
                "Initialization during interflood is not yet implemented"
            )

        self.min_unloaded_at_source = self.n_unloaded_at_source  # only decreases
        self.min_unloaded_timestamp = self.get_timestamp()

        if not (0.0 < cfg.main_share_A < 1.0 and 0.0 < cfg.alt_share < 1.0):
            raise ValueError(f"One of the shares is not within (0, 1) interval")

        if (
            not isinstance(cfg.monthly_prod_rate, list) or
            not isinstance(cfg.monthly_prod_rate[0], int) or
            len(cfg.monthly_prod_rate) != 12
        ):
            raise ValueError(
                "cfg.monthly_prod_rate must be a list of 12 integers representing how "
                "many containers will be loaded each month starting from December"
            )

        # State variables
        self.events = {}
        self.events_to_add = []
        self.callbacks = []

    def setup_fleet(self):
        """Dynamically initializes vessels and staggers their departures."""

        # Helper to safely bind loop variables in lambdas (important!)
        def make_wait_src(route_id, vessel_id, wait_time):
            return (
                lambda env: WaitAtSrc_Event(
                    env, route_id, vessel_id, wait_time
                )
            )

        def make_wait_interm(route_id, vessel_id, wait_time):
            return (
                lambda env: WaitAtInterm_Event(
                    env, route_id, vessel_id, wait_time, is_local_section=False
                )
            )

        # Route A Feeders
        if self.cfg.n_feeders_A > 0:
            stagger = self.cfg.rt_feeder_A // self.cfg.n_feeders_A
            for i in range(self.cfg.n_feeders_A):
                t = self.cfg.base_wait_feeder_A + (i * stagger)
                self.add_event(
                    make_wait_src(
                        route_id='A', vessel_id=f'feeder {i+1}', wait_time=t
                    )
                )

        # Route B Feeders
        if self.cfg.n_feeders_B > 0:
            stagger = self.cfg.rt_feeder_B // self.cfg.n_feeders_B
            for i in range(self.cfg.n_feeders_B):
                t = self.cfg.base_wait_feeder_B + (i * stagger)
                self.add_event(
                    make_wait_src(
                        route_id='B', vessel_id=f'feeder {i+1}', wait_time=t
                    )
                )

        # Route A Mainlands
        if self.cfg.n_mainlands_A > 0:
            stagger = self.cfg.rt_mainland_A // self.cfg.n_mainlands_A
            for i in range(self.cfg.n_mainlands_A):
                t = self.cfg.base_wait_mainland_A + (i * stagger)
                self.add_event(
                    make_wait_interm(
                        route_id='A', vessel_id=f'mainland {i+1}', wait_time=t
                    )
                )

        # Route B Mainlands
        if self.cfg.n_mainlands_B > 0:
            stagger = self.cfg.rt_mainland_B // self.cfg.n_mainlands_B
            for i in range(self.cfg.n_mainlands_B):
                wait_time = self.cfg.base_wait_mainland_B + (i * stagger)
                self.add_event(
                    make_wait_interm(
                        route_id='B', vessel_id=f'mainland {i+1}', wait_time=t
                    )
                )

        # Interflood Recovery (Surge Feeders)
        interflood_end_wait_time = self.days_until(self.cfg.interflood_end)
        if interflood_end_wait_time < 0:
            interflood_end_wait_time = (
                self.days_until(('December', 31)) +
                days_until(1, 1, self.cfg.interflood_end) + 1
            )

        if self.cfg.n_post_interflood_A > 0:
            # stagger = self.cfg.rt_feeder_A // self.cfg.n_post_interflood_A
            for i in range(self.cfg.n_post_interflood_A):
                # t = interflood_end_wait_time + (i * stagger)
                t = interflood_end_wait_time
                self.add_event(
                    make_wait_src(
                        route_id='A', vessel_id=f'feeder {i+1}*', wait_time=t
                    )
                )

        if self.cfg.n_post_interflood_B > 0:
            # stagger = self.cfg.rt_feeder_B // self.cfg.n_post_interflood_B
            for i in range(self.cfg.n_post_interflood_B):
                # t = interflood_end_wait_time + (i * stagger)
                t = interflood_end_wait_time
                self.add_event(
                    make_wait_src(
                        route_id='B', vessel_id=f'feeder {i+1}*', wait_time=t
                    )
                )

        # Route C Feeders
        alt_route_wait_time = self.days_until(self.cfg.alt_route_start)
        if alt_route_wait_time < 0:
            alt_route_wait_time = (
                self.days_until(('December', 31)) +
                days_until(1, 1, self.cfg.alt_route_start) + 1
            )

        if self.cfg.n_feeders_C > 0:
            stagger = self.cfg.rt_feeder_C // self.cfg.n_feeders_C
            for i in range(self.cfg.n_feeders_C):
                t = alt_route_wait_time + (i * stagger)
                # t = alt_route_wait_time
                self.add_event(
                    make_wait_src(
                        route_id='C', vessel_id=f'feeder {i+1}', wait_time=t
                    )
                )

    def get_timestamp(self):
        month_name = MONTH_NAMES[self.month]
        return month_name + ("  " if self.day < 10 else " ") + str(self.day)

    def in_period(self, period_start: tuple[str, int], period_end: tuple[str, int]):
        return in_period(
            self.month, self.day, period_start, period_end
        )

    def days_until(self, period_boundary: tuple[str, int]):
        return days_until(
            self.month, self.day, period_boundary
        )

    def route_A_local_duration(self):
        return self.cfg.route_A_local_distr(
            self.month, self.day
        )

    def route_A_global_duration(self):
        return self.cfg.route_A_global_distr(
            self.month, self.day
        )

    def route_B_local_duration(self):
        return self.cfg.route_B_local_distr(
            self.month, self.day
        )

    def route_B_global_duration(self):
        return self.cfg.route_B_global_distr(
            self.month, self.day
        )

    def route_B_transportation_duration(self):
        return self.cfg.route_B_transportation_distr(
            self.month, self.day
        )

    def route_C_duration(self):
        return self.cfg.route_C_distr(
            self.month, self.day
        )

    def add_event(self, event_lmbd):
        self.events_to_add.append(event_lmbd)

    def add_callback(self, event_callback):
        self.callbacks.append(event_callback)

    def simulate_day(self, verbose=False, output_handle=None):
        if verbose:
            if output_handle is None:
                print("***", self.get_timestamp(), "***")
            else:
                output_handle.write("*** " + self.get_timestamp() + " ***\n")

        n_avail_at_source = self.n_unloaded_at_source
        if n_avail_at_source < self.min_unloaded_at_source:
            self.min_unloaded_at_source = n_avail_at_source
            self.min_unloaded_timestamp = self.get_timestamp()

        events_to_delete = []
        for event_id, event in self.events.items():
            if finished := event():
                # finished event may result in n_unloaded_at_source increasing,
                # but we need some time to transport unloaded containers back,
                # so we save n_avail_at_source before any events start/end
                events_to_delete.append(event_id)

        if events_to_delete:
            for event_id in events_to_delete:
                if verbose:
                    if output_handle is None:
                        print("Event   ended >", event_id)
                    else:
                        output_handle.write("Event   ended > " + event_id)
                del self.events[event_id]

        if self.events_to_add:
            for event_lmbd in self.events_to_add:
                event = event_lmbd(self)
                self.events[event.id] = event
                if verbose:
                    if output_handle is None:
                        print("Event started >", event.id)
                    else:
                        output_handle.write("Event started > " + event.id)

            self.events_to_add.clear()

        if self.callbacks:
            for event_callback in self.callbacks:
                event_callback(self)

            self.callbacks.clear()

        days_in_month = MONTH_DAYS[self.month]
        monthly_prod_rate = self.cfg.monthly_prod_rate[self.month]
        daily_load = monthly_prod_rate // days_in_month
        remainder = monthly_prod_rate - days_in_month * daily_load
        n_to_load = (  # distribute remainder across the days
            daily_load + 1 if self.day + remainder > days_in_month else
            daily_load
        )

        if n_avail_at_source >= n_to_load:
            self.n_unloaded_at_source -= n_to_load
            # TODO: find better condition for determining if production rate is increased
            # in order to match alternative route
            base_prod_rate = min(self.cfg.monthly_prod_rate)
            if monthly_prod_rate > base_prod_rate:
                # TODO: make this formula more reliable, for now it is applicable only when
                # production rate is increased during 6 months as described below
                alt_load = round((2 * self.cfg.alt_share / (1 + self.cfg.alt_share)) * n_to_load)
                main_load = n_to_load - alt_load
                main_load_A = round(self.cfg.main_share_A * main_load)

                self.n_loaded_at_source_for_A += main_load_A
                self.n_loaded_at_source_for_B += main_load - main_load_A
                self.n_loaded_at_source_for_C += alt_load
            else:
                main_load_A = round(self.cfg.main_share_A * n_to_load)

                self.n_loaded_at_source_for_A += main_load_A
                self.n_loaded_at_source_for_B += n_to_load - main_load_A
        else:
            return True  # run failed

        if verbose:
            load_A = self.n_loaded_at_source_for_A
            load_B = self.n_loaded_at_source_for_B
            load_C = self.n_loaded_at_source_for_C
            n_loaded = load_A + load_B + load_C

            delivered_A = self.n_delivered_from_A
            delivered_B = self.n_delivered_from_B
            delivered_C = self.n_delivered_from_C
            n_delivered = delivered_A + delivered_B + delivered_C

            if output_handle is None:
                print(
                    f"S: n_unloaded = {self.n_unloaded_at_source}, n_loaded = {n_loaded}",
                    f"(with A = {load_A}, B = {load_B}, C = {load_C})"
                )
                print(f"A: n_unloaded = {self.n_unloaded_at_A}, n_loaded = {self.n_loaded_at_A}")
                print(f"B: n_unloaded = {self.n_unloaded_at_B}, n_loaded = {self.n_loaded_at_P}")
                print(
                    f"D: n_unloaded = {self.n_unloaded_at_destination}, n_delivered = {n_delivered}",
                    f"(with A = {delivered_A}, B = {delivered_B}, C = {delivered_C})"
                )
            else:
                output_handle.write(
                    f"S: n_unloaded = {self.n_unloaded_at_source}, n_loaded = {n_loaded} "
                    f"(with A = {load_A}, B = {load_B}, C = {load_C})\n"
                    f"A: n_unloaded = {self.n_unloaded_at_A}, n_loaded = {self.n_loaded_at_A}\n"
                    f"B: n_unloaded = {self.n_unloaded_at_B}, n_loaded = {self.n_loaded_at_P}\n"
                    f"D: n_unloaded = {self.n_unloaded_at_destination}, n_delivered = {n_delivered}"
                    f" (with A = {delivered_A}, B = {delivered_B}, C = {delivered_C})\n"
                )

        self.day += 1
        if self.day > days_in_month:
            self.month = (self.month + 1) % 12
            self.day = 1

        return False

    def simulate(
        self,
        seed=1,
        verbose_start=None,
        verbose_end=None,
        print_final_stats=False,
        output_file='stdout',
    ):
        np.random.seed(seed)

        output_handle = None if output_file == 'stdout' else open(output_file, 'w')

        failed = False
        for i in range(self.cfg.simulation_period):
            verbose = (
                verbose_start is not None and verbose_end is not None and
                self.in_period(verbose_start, verbose_end)
            ) or (print_final_stats and i + 1 == self.cfg.simulation_period)

            if failed := self.simulate_day(verbose, output_handle):
                break

        if print_final_stats:
            if failed:
                if output_handle is None:
                    print("!!! RUN FAILED !!!")
                else:
                    output_handle.write("!!! RUN FAILED !!!\n")

            for event_id in self.events.keys():
                if output_handle is None:
                    print("Event hanging >", event_id)
                else:
                    output_handle.write("Event hanging > " + event_id)

            n_delivered = (
                self.n_delivered_from_A + self.n_delivered_from_B + self.n_delivered_from_C
            )

            if output_handle is None:
                print("*** Final stats ***")
                print("Delivered to destination:", n_delivered)
                print(
                    "Minimum number of available containers:", self.min_unloaded_at_source,
                    "(" + self.min_unloaded_timestamp + ")",
                )
            else:
                output_handle.write(\
                    f"*** Final stats ***\n"
                    f"Delivered to destination: {n_delivered}\n"
                    f"Minimum number of available containers: "
                    f"{self.min_unloaded_at_source} ({self.min_unloaded_timestamp})\n"
                )

        if output_handle is not None:
            output_handle.close()

        return failed


class Src2Interm_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        n_load: int,
        route_duration: int,
    ):
        self.env = env

        assert route_id in ('A', 'B')

        if route_id == 'A':
            self.env.n_loaded_at_source_for_A -= n_load
        else:
            self.env.n_loaded_at_source_for_B -= n_load

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.n_load = n_load
        self.route_duration = route_duration
        self.day_counter = 0

        timestamp = self.env.get_timestamp()
        self.id = f"{timestamp}: S -> {route_id} ({vessel_id}), n = {n_load}, t = {route_duration}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.route_duration:
            return False

        if self.route_id == 'A':
            # self.env.n_loaded_at_A += self.n_load
            def _unload_at_A(env):
                env.n_loaded_at_A += self.n_load

            self.env.add_callback(_unload_at_A)
        else:
            transportation_time = self.env.route_B_transportation_duration()
            self.env.add_event(
                lambda env: TransportInterm_Event(
                    env, self.n_load, transportation_time, is_loaded=True
                )
            )

        # TODO: probably fix, taken from excel table where pause is 4 days, but
        # since we will wait at both points divide it by 2 for consistency?
        self.env.add_event(
            lambda env: WaitAtInterm_Event(
                env, self.route_id, self.vessel_id,
                wait_time=2, is_local_section=True,
            )
        )

        return True


class Src2Dest_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        n_load: int,
        route_duration: int,
    ):
        self.env = env

        assert route_id == 'C'

        self.env.n_loaded_at_source_for_C -= n_load

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.n_load = n_load
        self.route_duration = route_duration
        self.day_counter = 0

        timestamp = self.env.get_timestamp()
        self.id = f"{timestamp}: S -> D ({vessel_id}), n = {n_load}, t = {route_duration}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.route_duration:
            return False

        def _unload_at_dest(env):
            env.n_unloaded_at_destination += self.n_load

        self.env.add_callback(_unload_at_dest)
        self.env.n_delivered_from_C += self.n_load

        # TODO: fix, for now made it to match stop time at intermediate points
        self.env.add_event(
            lambda env: WaitAtDest_Event(
                env, self.route_id, self.vessel_id, wait_time=2
            )
        )

        return True


class Interm2Src_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        n_load: int,
        route_duration: int,
    ):
        self.env = env

        assert route_id in ('A', 'B')

        if route_id == 'A':
            self.env.n_unloaded_at_A -= n_load
        else:
            self.env.n_unloaded_at_B -= n_load

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.n_load = n_load
        self.route_duration = route_duration
        self.day_counter = 0

        timestamp = self.env.get_timestamp()
        self.id = f"{timestamp}: {route_id} -> S ({vessel_id}), n = {n_load}, t = {route_duration}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.route_duration:
            return False

        def _unload_at_src(env):
            env.n_unloaded_at_source += self.n_load

        self.env.add_callback(_unload_at_src)

        # TODO: see comment for Src2Interm_Event
        self.env.add_event(
            lambda env: WaitAtSrc_Event(
                env, self.route_id, self.vessel_id, wait_time=2
            )
        )

        return True


class Interm2Dest_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        n_load: int,
        route_duration: int,
    ):
        self.env = env

        assert route_id in ('A', 'B')

        if route_id == 'A':
            self.env.n_loaded_at_A -= n_load
        else:
            self.env.n_loaded_at_P -= n_load

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.n_load = n_load
        self.route_duration = route_duration
        self.day_counter = 0

        timestamp = self.env.get_timestamp()
        self.id = f"{timestamp}: {route_id} -> D ({vessel_id}), n = {n_load}, t = {route_duration}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.route_duration:
            return False

        def _unload_at_dest(env):
            env.n_unloaded_at_destination += self.n_load

        self.env.add_callback(_unload_at_dest)
        if self.route_id == 'A':
            self.env.n_delivered_from_A += self.n_load
        else:
            self.env.n_delivered_from_B += self.n_load

        # TODO: wait time is maximum 1 day from schedules I've seen,
        # probably a busy port, but better look into it...
        self.env.add_event(
            lambda env: WaitAtDest_Event(
                env, self.route_id, self.vessel_id, wait_time=1
            )
        )

        return True


class Dest2Interm_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        n_load: int,
        route_duration: int,
    ):
        self.env = env

        assert route_id in ('A', 'B')

        self.env.n_unloaded_at_destination -= n_load

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.n_load = n_load
        self.route_duration = route_duration
        self.day_counter = 0

        timestamp = self.env.get_timestamp()
        self.id = f"{timestamp}: D -> {route_id} ({vessel_id}), n = {n_load}, t = {route_duration}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.route_duration:
            return False

        if self.route_id == 'A':
            def _unload_empties_at_A(env):
                env.n_unloaded_at_A += self.n_load

            self.env.add_callback(_unload_empties_at_A)
        else:
            transportation_time = self.env.route_B_transportation_duration()
            self.env.add_event(
                lambda env: TransportInterm_Event(
                    env, self.n_load, transportation_time, is_loaded=False
                )
            )

        # TODO: see comment for Src2Dest_Event
        self.env.add_event(
            lambda env: WaitAtInterm_Event(
                env, self.route_id, self.vessel_id,
                wait_time=2, is_local_section=False,
            )
        )

        return True


class Dest2Src_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        n_load: int,
        route_duration: int,
    ):
        if route_id != 'C':
            raise ValueError(
                f"This event is tied to route C, but it originated from route {route_id}"
            )

        self.env = env
        self.env.n_unloaded_at_destination -= n_load

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.n_load = n_load
        self.route_duration = route_duration
        self.day_counter = 0

        timestamp = self.env.get_timestamp()
        self.id = f"{timestamp}: D -> S ({vessel_id}), n = {n_load}, t = {route_duration}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.route_duration:
            return False

        def _unload_at_src(env):
            env.n_unloaded_at_source += self.n_load

        self.env.add_callback(_unload_at_src)

        # TODO: see comment for Src2Dest_Event
        self.env.add_event(
            lambda env: WaitAtSrc_Event(
                env, self.route_id, self.vessel_id, wait_time=2
            )
        )

        return True


class WaitAtInterm_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        wait_time: int,
        is_local_section: bool,
    ):
        self.env = env

        assert route_id in ('A', 'B')

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.wait_time = wait_time
        self.is_local_section = is_local_section
        self.day_counter = 0
        self.route_duration = 0

        timestamp = self.env.get_timestamp()
        point = 'A' if route_id == 'A' else 'P'
        self.id = f"{timestamp}: ~{point} ({vessel_id}), t = {wait_time}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.wait_time:
            return False

        if self.is_local_section:
            if self.route_duration == 0:  # sample once
                self.route_duration = (
                    self.env.route_A_local_duration() if self.route_id == 'A' else
                    self.env.route_B_local_duration()
                )

            days_until_interflood = self.env.days_until(self.env.cfg.interflood_start)
            in_interflood_period = self.env.in_period(
                self.env.cfg.interflood_start, self.env.cfg.interflood_end
            )
            if (
                days_until_interflood >= 0 and
                self.route_duration >= days_until_interflood
            ) or in_interflood_period:
                return False  # wait until it ends

            # ----------------------------------------------------------- #
            if self.env.cfg.disable_underloaded_I2S:
                def _interm2src_or_wait_event(env):
                    n_unloaded = (
                        env.n_unloaded_at_A if self.route_id == 'A' else
                        env.n_unloaded_at_B
                    )
                    capacity = env.cfg.max_feeder_vessel_capacity
                    n_required = round(env.cfg.underload_share_I2S * capacity)

                    if n_unloaded <= n_required:
                        return WaitAtInterm_Event(
                            env, self.route_id, self.vessel_id,
                            wait_time=1, is_local_section=True,
                        )

                    return Interm2Src_Event(
                        env,
                        self.route_id,
                        self.vessel_id,
                        min(n_unloaded, capacity),
                        self.route_duration,
                    )

                self.env.add_event(_interm2src_or_wait_event)
            else:
                self.env.add_event(
                    lambda env: Interm2Src_Event(
                        env,
                        self.route_id,
                        self.vessel_id,
                        min(
                            (
                                env.n_unloaded_at_A if self.route_id == 'A' else
                                env.n_unloaded_at_B
                            ),
                            env.cfg.max_feeder_vessel_capacity
                        ),
                        self.route_duration,
                    )
                )
        else:
            if self.route_duration == 0:  # sample once
                self.route_duration = (
                    self.env.route_A_global_duration() if self.route_id == 'A' else
                    self.env.route_B_global_duration()
                )

            # ----------------------------------------------------------- #
            if self.env.cfg.disable_underloaded_I2D:
                def _interm2dest_or_wait_event(env):
                    n_loaded = (
                        env.n_loaded_at_A if self.route_id == 'A' else
                        env.n_loaded_at_P
                    )
                    capacity = env.cfg.max_mainland_vessel_capacity
                    n_required = round(env.cfg.underload_share_I2D * capacity)

                    if n_loaded <= n_required:
                        return WaitAtInterm_Event(
                            env, self.route_id, self.vessel_id,
                            wait_time=1, is_local_section=False,
                        )

                    return Interm2Dest_Event(
                        env,
                        self.route_id,
                        self.vessel_id,
                        min(n_loaded, capacity),
                        self.route_duration,
                    )

                self.env.add_event(_interm2dest_or_wait_event)
            else:
                self.env.add_event(
                    lambda env: Interm2Dest_Event(
                        env,
                        self.route_id,
                        self.vessel_id,
                        min(
                            (
                                env.n_loaded_at_A if self.route_id == 'A' else
                                env.n_loaded_at_P
                            ),
                            env.cfg.max_mainland_vessel_capacity
                        ),
                        self.route_duration,
                    )
                )

        return True


class WaitAtSrc_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        wait_time: int,
    ):
        self.env = env

        assert route_id in ('A', 'B', 'C')

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.wait_time = wait_time
        self.day_counter = 0
        self.route_duration = 0

        timestamp = self.env.get_timestamp()
        self.id = f"{timestamp}: ~S ({vessel_id}, route {route_id}), t = {wait_time}"

    def __call__(self):
        self.day_counter += 1
        
        if self.day_counter < self.wait_time:
            return False

        if self.route_id == 'C':
            days_until_alt_route = self.env.days_until(self.env.cfg.alt_route_start)
            if days_until_alt_route > 0:
                return False  # wait until it starts

            if self.route_duration == 0:  # sample once
                self.route_duration = self.env.route_C_duration()

            days_until_alt_route = self.env.days_until(self.env.cfg.alt_route_end)
            if (
                days_until_alt_route >= 0 and
                self.route_duration < days_until_alt_route
            ):
                # ----------------------------------------------------------- #
                if self.env.cfg.disable_underloaded_S2D:
                    def _src2dest_or_wait_event(env):
                        n_loaded = env.n_loaded_at_source_for_C
                        capacity = env.cfg.max_feeder_vessel_capacity
                        n_required = round(env.cfg.underload_share_S2D * capacity)

                        if n_loaded <= n_required:
                            return WaitAtSrc_Event(
                                env, self.route_id, self.vessel_id, wait_time=1
                            )

                        return Src2Dest_Event(
                            env,
                            self.route_id,
                            self.vessel_id,
                            min(n_loaded, capacity),
                            self.route_duration,
                        )

                    self.env.add_event(_src2dest_or_wait_event)
                else:
                    self.env.add_event(
                        lambda env: Src2Dest_Event(
                            env,
                            self.route_id,
                            self.vessel_id,
                            min(
                                env.n_loaded_at_source_for_C,
                                env.cfg.max_feeder_vessel_capacity
                            ),
                            self.route_duration,
                        )
                    )
            else:
                return False  # wait for the next season
        else:
            if self.route_duration == 0:  # sample once
                self.route_duration = (
                    self.env.route_A_local_duration() if self.route_id == 'A' else
                    self.env.route_B_local_duration()
                )

            days_until_interflood = self.env.days_until(self.env.cfg.interflood_start)
            in_interflood_period = self.env.in_period(
                self.env.cfg.interflood_start, self.env.cfg.interflood_end
            )
            if (
                days_until_interflood >= 0 and
                self.route_duration >= days_until_interflood
            ) or in_interflood_period:
                return False  # wait until it ends

            # ----------------------------------------------------------- #
            if self.env.cfg.disable_underloaded_S2I:
                def _src2interm_or_wait_event(env):
                    n_loaded = (
                        env.n_loaded_at_source_for_A if self.route_id == 'A' else
                        env.n_loaded_at_source_for_B
                    )
                    capacity = env.cfg.max_feeder_vessel_capacity
                    n_required = round(env.cfg.underload_share_S2I * capacity)

                    if n_loaded <= n_required:
                        return WaitAtSrc_Event(
                            env, self.route_id, self.vessel_id, wait_time=1
                        )

                    return Src2Interm_Event(
                        env,
                        self.route_id,
                        self.vessel_id,
                        min(n_loaded, capacity),
                        self.route_duration,
                    )

                self.env.add_event(_src2interm_or_wait_event)
            else:
                self.env.add_event(
                    lambda env: Src2Interm_Event(
                        env,
                        self.route_id,
                        self.vessel_id,
                        min(
                            (
                                env.n_loaded_at_source_for_A if self.route_id == 'A' else
                                env.n_loaded_at_source_for_B
                            ),
                            env.cfg.max_feeder_vessel_capacity
                        ),
                        self.route_duration,
                    )
                )

        return True


class WaitAtDest_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        route_id: str,
        vessel_id: str,
        wait_time: int,
    ):
        self.env = env

        assert route_id in ('A', 'B', 'C')

        self.route_id = route_id
        self.vessel_id = vessel_id
        self.wait_time = wait_time
        self.day_counter = 0
        self.route_duration = 0

        timestamp = self.env.get_timestamp()
        self.id = f"{timestamp}: ~D ({vessel_id}, route {route_id}), t = {wait_time}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.wait_time:
            return False

        if self.route_id == 'C':
            if self.route_duration == 0:  # sample once
                self.route_duration = self.env.route_C_duration()

            days_until_alt_route = self.env.days_until(self.env.cfg.alt_route_end)
            if (
                days_until_alt_route >= 0 and
                self.route_duration < days_until_alt_route
            ):
                # ----------------------------------------------------------- #
                if self.env.cfg.disable_underloaded_D2S:
                    def _dest2src_or_wait_event(env):
                        n_unloaded = env.n_unloaded_at_destination
                        capacity = env.cfg.max_feeder_vessel_capacity
                        n_required = round(env.cfg.underload_share_D2S * capacity)

                        if n_unloaded <= n_required:
                            return WaitAtDest_Event(
                                env, self.route_id, self.vessel_id, wait_time=1
                            )

                        return Dest2Src_Event(
                            env,
                            self.route_id,
                            self.vessel_id,
                            min(n_unloaded, capacity),
                            self.route_duration,
                        )
    
                    self.env.add_event(_dest2src_or_wait_event)
                else:
                    self.env.add_event(
                        lambda env: Dest2Src_Event(
                            env,
                            self.route_id,
                            self.vessel_id,
                            min(
                                env.n_unloaded_at_destination,
                                env.cfg.max_feeder_vessel_capacity
                            ),
                            self.route_duration,
                        )
                    )
            else:
                # TODO: for now waits until the next season if either there is not
                # enough time to return or the route is already closed
                return False
        else:
            if self.route_duration == 0:
                self.route_duration = (
                    self.env.route_A_global_duration() if self.route_id == 'A' else
                    self.env.route_B_global_duration()
                )

            # ----------------------------------------------------------- #
            if self.env.cfg.disable_underloaded_D2I:
                def _dest2interm_or_wait_event(env):
                    n_unloaded = env.n_unloaded_at_destination
                    capacity = env.cfg.max_mainland_vessel_capacity
                    n_required = round(env.cfg.underload_share_D2I * capacity)

                    if n_unloaded <= n_required:
                        return WaitAtDest_Event(
                            env, self.route_id, self.vessel_id, wait_time=1
                        )

                    return Dest2Interm_Event(
                        env,
                        self.route_id,
                        self.vessel_id,
                        min(n_unloaded, capacity),
                        self.route_duration,
                    )

                self.env.add_event(_dest2interm_or_wait_event)
            else:
                self.env.add_event(
                    lambda env: Dest2Interm_Event(
                        env,
                        self.route_id,
                        self.vessel_id,
                        min(
                            env.n_unloaded_at_destination,
                            env.cfg.max_mainland_vessel_capacity
                        ),
                        self.route_duration,
                    )
                )

        return True


class TransportInterm_Event:

    def __init__(
        self,
        env: SimulationEnvironment,
        n_load: int,
        transportation_time: int,
        is_loaded: bool,
    ):
        self.env = env

        self.n_load = n_load
        self.transportation_time = transportation_time
        self.is_loaded = is_loaded
        self.day_counter = 0

        timestamp = self.env.get_timestamp()
        transition = "B -> P" if is_loaded else "P -> B"
        self.id = f"{timestamp}: {transition}, n = {n_load}, t = {transportation_time}"

    def __call__(self):
        self.day_counter += 1

        if self.day_counter < self.transportation_time:
            return False

        if self.is_loaded:
            # self.env.n_loaded_at_P += self.n_load
            def _unload_at_P(env):
                env.n_loaded_at_P += self.n_load

            self.env.add_callback(_unload_at_P)
        else:
            # self.env.n_unloaded_at_B += self.n_load
            def _unload_at_B(env):
                env.n_unloaded_at_B += self.n_load

            self.env.add_callback(_unload_at_B)

        return True


def run_single_simulation(args):
    """Runs one simulation year. Designed to be multiprocessing-friendly."""
    cfg, seed = args

    env = SimulationEnvironment(cfg)
    env.setup_fleet()

    failed = env.simulate(seed=seed)

    return {
        'seed': seed,
        'failed': failed,
        'min_containers': env.min_unloaded_at_source,
        'total_delivered': (
            env.n_delivered_from_A + env.n_delivered_from_B + env.n_delivered_from_C
        ),
        'delivered_A': env.n_delivered_from_A,
        'delivered_B': env.n_delivered_from_B,
        'delivered_C': env.n_delivered_from_C,
    }


def evaluate_config(cfg: SimulationConfig, n_sims: int):
    """Runs Monte Carlo for a specific fleet cfguration and returns summary stats."""

    # Generate unique seeds
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1000000, size=n_sims)

    # Use ProcessPoolExecutor to run simulations in parallel (much faster)
    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(run_single_simulation, [(cfg, seed) for seed in seeds])
        )

    df = pd.DataFrame(results)

    return {
        'start_date': (
            (
                cfg.start_month if isinstance(cfg.start_month, str) else
                MONTH_NAMES[cfg.start_month]
            ) + " " + str(cfg.start_day)
        ),

        'n_containers': cfg.n_containers,

        'n_main_A': cfg.n_mainlands_A,
        'n_main_B': cfg.n_mainlands_B,
        'n_feeder': f'{cfg.n_feeders_A}+{cfg.n_post_interflood_A}',

        'failure_rate': df['failed'].mean(),

        'avg_delivered': df['total_delivered'].mean(),
        # 'avg_delivered*': df['total_delivered'][~df['failed']].mean(),

        # 5-th percentile represents the "worst case" inventory drop across 95% of scenarios
        'worst_case_min_inv': np.percentile(df['min_containers'], 5),
        # 'worst_case_min_inv*': np.percentile(df['min_containers'][~df['failed']], 5),
    }


if __name__ == '__main__':
    # How events work: most of them are looped sequence with exception for
    # interflood pause and when alternative route becomes unavailable
    # route A (local):  -> WaitAtSrc -> Src2Interm -> WaitAtInterm -> Interm2Src ->
    # route A (global): -> WaitAtInterm -> Interm2Dest -> WaitAtDest -> Dest2Interm ->
    # route B (local):  -> WaitAtSrc -> Src2Interm -> WaitAtInterm -> Interm2Src ->
    #                                       |-> TransportInterm (from B to P)
    # route B (global): -> WaitAtInterm -> Interm2Dest -> WaitAtDest -> Dest2Interm ->
    #                                        TransportInterm (from P to B) <-|
    # route C: -> WaitAtSrc -> Src2Dest -> WaitAtDest -> Dest2Src ->
    cfg = SimulationConfig(
        simulation_period=(365 + 60),
        start_month='February', start_day=2,
        interflood_start=('May', 10),
        interflood_end=('June', 30),
        alt_route_start=('July', 1),
        alt_route_end=('November', 10),
        monthly_prod_rate=[
            3000, 3000, 3000, 3000, 3000, 3666, 3666, 3667, 3667, 3667, 3667, 3000
        ],
        n_containers=30_000,
        n_feeders_A=2, n_feeders_B=2, n_feeders_C=4,
        n_mainlands_A=6, n_mainlands_B=4,
        n_post_interflood_A=3, n_post_interflood_B=2,
        # if we disable it, then everything colapses after interflood ends...
        disable_underloaded_I2S=False,
        underload_share_I2D=0.8,
        underload_share_S2I=0.5,
    )

    # print(evaluate_config(cfg, n_sims=10_000))
    # ---- OR ---- #
    env = SimulationEnvironment(cfg)
    env.setup_fleet()  # or custom setup if neccessary
    env.simulate(
        seed=1,
        verbose_start=('February', 2),
        verbose_end=('February', 3),
        print_final_stats=True,
    )
