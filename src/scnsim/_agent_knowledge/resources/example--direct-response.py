"""Complete Direct usage candidate; execute explicitly, never during resource reads.

This writes workspaces/agent-direct below the working directory and launches
the real Julia backend. It contains no stored or synthetic solver outputs.
"""

# %% Define the physical model, then expose the resonator's assembly boundary.
from scnsim import CircuitPlan, components, units as u

plan = CircuitPlan(id="primitive_resonator")
resonator = plan.subsystem(id="resonator")

capacitor = resonator.add(
    components.capacitor(id="capacitor", capacitance=110.0 * u.fF)
)
inductor = resonator.add(
    components.inductor(id="inductor", inductance=5.8 * u.nH)
)
resonator_bus = resonator.bus(id="terminal")
parallel_lc = resonator.parallel(
    id="parallel_lc",
    start=resonator_bus,
    branches=((capacitor,), (inductor,)),
    end=resonator.ground,
)
terminal = resonator.expose_pin(id="terminal", at=resonator_bus)

# %% Assemble the coupling and terminated measurement Port in the parent.
signal_bus = plan.bus(id="signal_boundary")
resonator_root_bus = plan.bus(id="resonator_node")
coupling_cap = plan.add(
    components.capacitor(id="coupling_cap", capacitance=6.0 * u.fF)
)
coupling = plan.series(
    id="coupling",
    start=signal_bus,
    elements=(coupling_cap,),
    end=resonator_root_bus,
)
plan.link(
    id="resonator_terminal",
    endpoints=(resonator_root_bus, terminal),
)
signal_port = plan.add_port(
    id="signal_in",
    at=signal_bus,
    role="terminated",
    reference_impedance=50.0 * u.ohm,
)
resonator_node = resonator_root_bus.node

# %% Ask a frequency-response question without changing the authored circuit.
from scnsim import CircuitRun, DirectSolveSpec

run = CircuitRun(plan=plan, workspace="workspaces/agent-direct")
original = run.original
direct_spec = DirectSolveSpec(
    frequencies=[5.5, 6.0, 6.5, 7.0] * u.GHz,
)
direct = run.solve(original, direct_spec)

# %% Inspect actual result coordinates and complex scattering data.
print("Frequencies:", direct.frequencies.to(u.GHz))
print("S channels:", direct.s.view.coordinates)
scattering = direct.s.view.matrix.magnitude
print("S shape (frequency, output, input):", scattering.shape)
print("S real:", scattering.real)
print("S imaginary:", scattering.imag)

# %% Retrieve this exact stored success; resolve does not launch another solve.
resolved = run.resolve(original, direct_spec)
if resolved.identity != direct.identity:
    raise RuntimeError("Resolved result did not preserve the exact result identity")
print("Exact stored result:", resolved.identity.result_sha256)

# %% Invalid input is rejected before any new calculation is requested.
try:
    DirectSolveSpec(frequencies=[-1.0] * u.GHz)
except ValueError as error:
    print("Rejected invalid frequency:", error)
else:
    raise RuntimeError("DirectSolveSpec unexpectedly accepted a negative frequency")
