"""Declared objective loss on the single native physical Quantity callable.

The coordinator supplies finalized targets/scales and owns search/continuation.
One call evaluates one physical point and the complete requested JAX transform. Numerical derivatives
remain physical SI derivatives; optimizer-coordinate transformations are outside
this module. No native factors, points or derivative tapes survive the call.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple, Any
from threading import RLock
import hashlib
import json

from ..canonical import canonical_json_bytes, float64_from_hex
from .quantity_program import compile_quantity_program
from ..numerics.objective_loss import compile_objective_loss
from ..numerics.program_models import ProgramOutcome
from ..numerics.evidence import require_primal


class ObjectiveOutcome(NamedTuple):
    primal: ProgramOutcome
    values: tuple
    normalized_residuals: tuple
    contributions: tuple
    total_cost: Any
    gradient: Any
    aux: Any


@dataclass(frozen=True, slots=True, eq=False)
class ObjectiveProgram:
    quantity_program: object
    loss: object
    identity_bytes: bytes
    _native_functions: dict = field(default_factory=dict, repr=False, compare=False)
    _function_lock: Any = field(default_factory=RLock, repr=False, compare=False)

    @property
    def program(self): return self.quantity_program.program
    @property
    def precision(self): return self.program.precision
    @property
    def output_layout(self): return self.quantity_program.output_layout
    @property
    def program_sha256(self): return hashlib.sha256(self.identity_bytes).hexdigest()

    def inputs(self, parameter_set, *, branch_starts_rad_s=None):
        return self.quantity_program.inputs(parameter_set, branch_starts_rad_s=branch_starts_rad_s)

    def prepare_batch(self, inputs_sequence, *, sparse_runner=None, program_owner=None, affine_evidence=None):
        # Capability acquisition/classification only; no baseline solve or
        # optimizer-specific physics. The operation owns these static packets.
        observations = ((None,) * len(inputs_sequence) if affine_evidence is None else affine_evidence)
        return tuple(self.quantity_program.native_program(inputs, program_owner=program_owner,
                                                           admit_core=False, affine_evidence=observation)
                     for inputs, observation in zip(inputs_sequence, observations, strict=True))

    def _native_frame(self, inputs, prepared=None, *, program_owner=None, affine_evidence=None):
        import jax
        import jax.numpy as jnp
        native = (self.quantity_program.native_program(inputs, program_owner=program_owner,
                                                       admit_core=False, affine_evidence=affine_evidence)
                  if prepared is None else prepared)
        # Physical classification uses the untouched host binary64 inputs.
        # Create device operands with x64 enabled BEFORE native tracing, even
        # for selected32 arithmetic; the native core owns that later cast.
        with jax.enable_x64(True):
            source_indices = jnp.asarray(native.input_layout['omega']['source_indices'], dtype=jnp.int32)
            arguments = (jnp.asarray(inputs.theta, dtype=jnp.float64),
                         jnp.asarray(inputs.omega_rad_s)[source_indices],
                         jnp.asarray(inputs.branch_starts_rad_s))
        return native, arguments

    def coefficient_operands(self):
        """Actual execution coefficients, distinct from compilation identity."""
        import jax.numpy as jnp
        return tuple(jnp.asarray(values, dtype=jnp.float64)
                     for values in self.loss.runtime_operands())

    def execution_variant(self, native, arguments, *, derivative, device, completed=False):
        """The same composed callable/key/signature consumed by warmup and execute."""
        # Execution wraps this pure (key, callable, operands) recipe in its
        # immutable plan; compilation has no upward execution dependency.
        transform = 'vjp' if derivative else 'checked_value'
        if completed:
            import jax
            from ..numerics.native_program import bind_completed_branches
            arguments = (arguments[0], arguments[1], bind_completed_branches(arguments[2]), *arguments[3:])
            transform = ('checked_completed_objective_vjp' if derivative else 'checked_completed_objective_value',
                         str(jax.tree.structure(arguments[3])))
        key = (native.variant_key('value', device=device, operands=arguments,
                                  transform=transform),
               ('objective_loss', self.loss.structural_record()))
        return key, self._function(native, derivative=derivative, completed=completed), tuple(arguments)

    def _invoke(self, native, arguments, *, derivative, native_executor,
                program_owner, device, completed=None):
        from contextlib import nullcontext
        if completed is None:
            arguments = (*arguments, self.coefficient_operands())
        else:
            packets, original_errors = completed
            from ..numerics.native_program import bind_completed_branches
            packets = bind_completed_branches(packets)
            arguments = (arguments[0], arguments[1], packets, original_errors, self.coefficient_operands())
        key, function, operands = self.execution_variant(native, arguments, derivative=derivative,
                                                         device=device, completed=completed is not None)
        borrowed = (nullcontext() if program_owner is None else
                    program_owner.hold_exact_core(native.core))
        with borrowed:
            return native_executor(function,
                *operands, precision=self.precision, compile_key=key,
                owner=program_owner,
                phase='physical_objective_value_gradient' if derivative else 'physical_objective_value',
                details={'objective_count': len(self.loss.objectives)})

    def _function(self, native, *, derivative, completed=False):
        """Stable same-frame loss body, with no graph interpreter or live point."""
        import jax
        from ..numerics.native_sparse import checked_native
        key = native.structural_identity, derivative, completed
        with self._function_lock:
            function = self._native_functions.get(key)
            if function is None:
                raw = native.completed_numeric_function if completed else native.numeric_function
                loss = self.loss.algebra_function()
                output_pullback = self.loss.output_pullback_function()

                def result_loss(result, coefficients):
                    if derivative:
                        payload, covectors = output_pullback(result[1].output_values, coefficients)
                        return payload.total_cost, (result, payload, covectors)
                    payload = loss(result[1].output_values, coefficients)
                    return payload.total_cost, (result, payload)

                if completed:
                    def body(theta, omega, packets, original_errors, coefficients):
                        # The native producer rebuilds candidate(theta), consumes
                        # actual same-point packets with implicit root rules and
                        # archives their original selected checked Errors.
                        return result_loss(raw(theta, omega, packets, original_errors), coefficients)
                else:
                    def body(theta, omega, seeds, coefficients):
                        return result_loss(raw(theta, omega, seeds), coefficients)
                function = checked_native(jax.value_and_grad(body, argnums=0, has_aux=True)
                                          if derivative else body)
                self._native_functions[key] = function
            return function

    def value(self, inputs, sparse_runner=None, *, prepared=None, native_executor,
              program_owner=None, device, compact_evidence=False, completed=None, affine_evidence=None):
        import jax
        with jax.enable_x64(True):
            from ..numerics.evidence import project_native_outcome
            native, arguments = self._native_frame(inputs, prepared, program_owner=program_owner, affine_evidence=affine_evidence)
            error, (_, (result, payload)) = self._invoke(native, arguments, derivative=False,
                native_executor=native_executor, program_owner=program_owner, device=device,
                completed=completed)
            primal = project_native_outcome(native, result, error, compact_evidence=compact_evidence)
            require_primal(primal)
            return ObjectiveOutcome(primal, payload.values, payload.normalized_residuals,
                                    payload.contributions, payload.total_cost, None, primal.aux)

    def value_and_gradient(self, inputs, *, active_keys, sparse_runner=None,
                           native_executor, program_owner=None, device, native_ad_runner=None, compact_evidence=False, completed=None, affine_evidence=None):
        """Project one certified complete-cost derivative from the shared loss.

        Response work requires native_ad_runner(native, arguments, loss,
        active_keys=...) -> (DerivativeOutcome, LossOutcome). The derivative
        uses the complete physical theta axis; this method selects active axes
        in declaration order. The injected host runner owns bounded response
        residual/replay and certificate projection, never another loss model.
        Root-only work uses the same native joint callable directly. Optional
        completed=(ordered packets, original selected checked Errors) is an
        operation-owned same-point operand; candidate(theta) and local implicit
        derivatives remain traced, and no completed arrays enter static caches.
        """
        import jax
        with jax.enable_x64(True):
            import jax.numpy as jnp
            from ..numerics.evidence import project_native_derivative_outcome
            native, arguments = self._native_frame(inputs, program_owner=program_owner, affine_evidence=affine_evidence)
            if native.work_layout:
                if native_ad_runner is None:
                    raise TypeError('response Objective AD requires the bounded native_ad_runner')
                completed_options = {} if completed is None else {
                    "completed_branches": completed[0], "completed_branch_errors": completed[1],
                }
                derivative, payload = native_ad_runner(
                    native, arguments, self.loss, active_keys=active_keys,
                    compact_evidence=compact_evidence, **completed_options)
            else:
                error, ((_, (result, payload, covectors)), full_gradient) = self._invoke(native, arguments, derivative=True,
                    native_executor=native_executor, program_owner=program_owner, device=device,
                    completed=completed)
                # B owns precise declared derivative demand and transformed equation
                # certificates. Loss-covector zero is not a fabricated derivative.
                derivative = project_native_derivative_outcome(native, (result, full_gradient), error,
                    output_covectors=native.output_covector_rows(covectors), compact_evidence=compact_evidence)
            require_primal(derivative.primal)
            indices = self.program.model_program.fields.layout.input_indices(active_keys)
            gradient = (derivative.derivative[jnp.asarray(indices, dtype=jnp.int32)]
                        if derivative.derivative is not None else None)
            return ObjectiveOutcome(derivative.primal, payload.values, payload.normalized_residuals,
                                    payload.contributions, payload.total_cost, gradient, derivative.aux)


def _smooth_operand(quantity):
    if quantity['type'] == 'quantity_absolute':
        return quantity['operand']
    if quantity.get('projection') == 'magnitude':
        return dict(quantity, projection=None)
    return None


def compile_objective_quantities(plan_document, optimization_spec, parameter_set, *, default_view=None, mesh=None,
                              bounds=None, precision='float64', authorized=frozenset(), program_owner=None, admit_program=False):
    objectives = tuple(optimization_spec['objectives'])
    declarations, modes = {}, []
    for objective in objectives:
        quantity = objective['quantity']
        raw = _smooth_operand(quantity)
        if (raw is not None and objective['comparison'] == 'target'
                and float64_from_hex(objective['target']['si_value_f64']) == 0):
            declarations[objective['id']] = raw
            modes.append('squared_magnitude_zero_target')
        else:
            declarations[objective['id']] = quantity
            modes.append('value')
    quantity_program = compile_quantity_program(plan_document, declarations, parameter_set,
                                                default_view=default_view, mesh=mesh,
                                                bounds=bounds, precision=precision, authorized=authorized, program_owner=program_owner, admit_program=admit_program)
    return quantity_program, tuple(modes)


def compile_objective_program(plan_document, optimization_spec, parameter_set, *, default_view=None, mesh=None,
                              bounds=None, precision='float64', authorized=frozenset(),
                              program_owner=None, admit_program=False, require_resolved_scales=True):
    quantity_program, modes = compile_objective_quantities(
        plan_document, optimization_spec, parameter_set, default_view=default_view, mesh=mesh,
        bounds=bounds, precision=precision, authorized=authorized,
        program_owner=program_owner, admit_program=admit_program)
    objectives = tuple(optimization_spec['objectives'])
    loss = compile_objective_loss(objectives, operand_modes=modes,
                                  require_resolved_scales=require_resolved_scales)
    identity = {'schema': 'scnsim.objective_program', 'schema_version': 1,
                'quantity_program': json.loads(quantity_program.identity_bytes),
                'objectives': objectives, 'operand_modes': modes}
    return ObjectiveProgram(quantity_program, loss, canonical_json_bytes(identity))
