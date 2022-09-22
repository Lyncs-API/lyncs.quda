"""
Implementation of the HMC algorithm
"""

from dataclasses import dataclass, field
from math import prod
from numpy import exp
from random import random
from argparse import Namespace
import click
from tqdm import tqdm
from lyncs_quda import gauge_field, momentum, lib, MPI
from aim import Run
import time


@dataclass
class HMCHelper:
    beta: float = 5
    lattice: tuple = (4, 4, 4, 4)
    # Log
    last_action: float = field(init=False, default=None)

    @property
    def ndims(self):
        return len(self.lattice)

    @property
    def plaq_coeff(self):
        "Plaquette coefficient"
        return -self.beta / 6  # = -1/g_0^2

    @property
    def plaq_paths(self):
        "List of plaquette paths"
        return tuple(
            (mu, nu, -mu, -nu)
            for mu in range(1, self.ndims)
            for nu in range(mu + 1, self.ndims + 1)
        )

    @property
    def coeffs(self):
        "For now only plaquettes, but can be extended"
        return self.plaq_coeff

    @property
    def paths(self):
        "For now only plaquettes, but can be extended"
        return self.plaq_paths

    def action(self, field):
        "Returns the action computed on field"
        self.last_action = 2 * float(
            field.compute_paths(self.paths, self.coeffs).reduce(mean=False)
        )
        return self.last_action

    def unity_gauge(self):
        "Returns a unity geuge"
        out = gauge_field(self.lattice)
        out.unity()
        return out

    def random_gauge(self):
        "Returns a random mom"
        out = gauge_field(self.lattice)
        out.gaussian(10)
        return out

    def random_mom(self):
        "Returns a random momenutm"
        out = momentum(self.lattice)
        out.gaussian(1 / 2**0.5)
        return out

    def update_field(self, field, mom, coeff):
        "Updates the Gauge field"
        return field.update_gauge(mom, coeff)

    def update_mom(self, field, mom, coeff):
        "Updates the momentum"
        return field.compute_paths(
            self.paths, self.coeffs, out=mom, add_coeff=coeff, force=True
        )

    def update(self, field, mom, fcoeff, mcoeff):
        if fcoeff != 0:
            field = self.update_field(field, mom, fcoeff)
        if mcoeff != 0:
            mom = self.update_mom(field, mom, mcoeff)
        return field, mom

    def mom2(self, mom):
        "Norm squared of the momentum half"
        return mom.norm2() / 2

    def hamiltonian(self, field, mom):
        mom2 = self.mom2(mom)
        action = self.action(field)
        return mom2 + action


@dataclass
class Integrator:
    steps: int
    time: float = 1.0
    order: int = field(init=False, default=1)
    pos_coeffs: tuple = field(init=False, default=(0, 1))
    mom_coeffs: tuple = field(init=False, default=(0.5, 0.5))

    def __call__(self, field, mom, helper):

        # Use negative time for reversed integration
        dtime = self.time / self.steps
        for step in range(self.steps):

            for fcoeff, mcoeff in zip(self.pos_coeffs, self.mom_coeffs):
                field, mom = helper.update(field, mom, fcoeff * dtime, mcoeff * dtime)

        return field, mom


HMC_INTEGRATORS = {}


def register(cls):
    HMC_INTEGRATORS[cls.__name__] = cls
    return cls


@register
class LeapFrog(Integrator):
    "Leap frog integration scheme"


@register
@dataclass
class MN2(Integrator):
    "Second order minimal norm integration scheme in position version"
    alpha: float = 0
    order: int = field(init=False, default=2)

    def __post_init__(self):
        self.pos_coeffs = (0, 0.5, 0.5)
        self.mom_coeffs = (self.alpha, 1 - 2 * self.alpha, self.alpha)


@register
@dataclass
class MN2p(Integrator):
    "Second order minimal norm integration scheme in velocity version"
    alpha: float = 0
    order: int = field(init=False, default=2)

    def __post_init__(self):
        self.pos_coeffs = (self.alpha, 1 - 2 * self.alpha, self.alpha)
        self.mom_coeffs = (0.5, 0.5, 0)


_OMF4 = (
    0.2539785108410595,
    -0.03230286765269967,
    0.08398315262876693,
    0.6822365335719091,
)


@register
@dataclass
class OMF4(Integrator):
    "Fourth order integration scheme in velocity version"
    order: int = field(init=False, default=4)
    pos_coeffs: tuple = field(
        init=False,
        default=(
            0,
            _OMF4[0],
            _OMF4[1],
            1 - 2 * (_OMF4[0] + _OMF4[1]),
            _OMF4[1],
            _OMF4[0],
        ),
    )
    mom_coeffs: tuple = field(
        init=False,
        default=(
            _OMF4[2],
            _OMF4[3],
            0.5 - (_OMF4[2] + _OMF4[3]),
            0.5 - (_OMF4[2] + _OMF4[3]),
            _OMF4[3],
            _OMF4[2],
        ),
    )


@dataclass
class HMC:
    helper: HMCHelper
    integrate: Integrator

    # Metrics
    accepted: int = field(init=False, default=0)
    steps: int = field(init=False, default=0)
    denergy: float = field(init=False, default=0)
    last_accepted: bool = field(init=False, default=False)
    last_action: float = field(init=False, default=0)

    def __call__(self, field):
        mom = self.helper.random_mom()
        energy = self.helper.hamiltonian(field, mom)
        action = self.helper.last_action
        field1, mom1 = self.integrate(field, mom, self.helper)
        energy1 = self.helper.hamiltonian(field1, mom1)
        action1 = self.helper.last_action
        self.denergy = energy1 - energy

        self.steps += 1
        self.last_accepted = random() < exp(-self.denergy)
        self.accepted += self.last_accepted

        if self.last_accepted:
            self.last_action = action1
            return field1
        self.last_action = action
        return field

    @property
    def last_plaquette(self):
        return (
            -3
            * self.last_action
            / (self.helper.beta * prod(self.helper.lattice) * len(self.helper.paths))
        )

    @property
    def stats(self):
        return {
            "action": self.last_action,
            "plaquette": self.last_plaquette,
            "dH": self.denergy,
            "exp(-dH)": exp(-self.denergy),
            "acc. rate": self.accepted / self.steps,
            "accepted": self.last_accepted,
        }


@click.command()
@click.option("--beta", type=float, default=5, help="target action's beta")
@click.option("--lattice-size", type=int, default=16, help="Size of hypercubic lattice")
@click.option(
    "--lattice-dims",
    nargs=4,
    type=int,
    default=(0, 0, 0, 0),
    help="Size of asymmetric lattice",
)
@click.option(
    "--procs",
    nargs=4,
    default=(1, 1, 1, 1),
    type=int,
    help="Cartesian topology of the communicator",
)
@click.option(
    "--integrator",
    default="OMF4",
    type=click.Choice(HMC_INTEGRATORS),
    help="Integrator for HMC",
)
@click.option(
    "--t-steps",
    type=int,
    default=3,
    help="Number of time steps trajectory",
)
@click.option(
    "--n-trajs",
    type=int,
    default=100,
    help="Number of trajectories",
)
@click.option(
    "--start",
    type=click.Choice(["unity", "random"]),
    default="random",
    help="Initial field",
)
def main(**kwargs):
    args = Namespace(**kwargs)

    lattice = (
        args.lattice_dims if prod(args.lattice_dims) != 0 else (args.lattice_size,) * 4
        )
    lib.set_comm(procs=args.procs)
    helper = HMCHelper(args.beta, lattice)
    integr = HMC_INTEGRATORS[args.integrator]
    integr = integr(args.t_steps)
    hmc = HMC(helper, integr)

    if args.start == "random":
        field = helper.random_gauge()
    elif args.start == "unity":
        field = helper.random_unity()
    else:
        raise ValueError("Unknown start")

    dname = "/cyclamen/home/syamamoto/Lattice2022/"
    fname = (
        "".join(tuple(map(str, args.procs + lattice)))
        + f"_beta{args.beta}_{args.integrator}_tsteps{args.t_steps}_ntraj_{args.n_trajs}with{args.start}"
    )
    fp = open(dname + fname, "w")

    # run = Run(repo='/cyclamen/home/syamamoto/Lattice2022/aim', run_hash=hash_id, experiment=experiment, system_tracking_interval=1)
    run = Run(
        repo="/cyclamen/home/syamamoto/Lattice2022/aim",
        experiment=experiment,
        system_tracking_interval=1,
    )
    run["beta"] = args.beta
    run["num_ranks"] = prod(args.procs)

    with tqdm(range(args.n_trajs)) as pbar:
        for step in pbar:
            field = hmc(field)
            pbar.set_description(f"plaq: {hmc.last_plaquette}")

            for key, val in hmc.stats.items():
                run.track(val, name=key)

            print(" ".join(list(map(str, hmc.stats.values()))), file=fp)
    fp.close()


if __name__ == "__main__":
    main()
