"""Immutable physical declarations and factorized output layout.

This compilation carrier owns no numerical evaluator, factor, coefficient
point, tape or executable. Weak native indices provide admission coordination;
actual compiled ownership belongs to the bounded Run/operation CompileOwner.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from math import prod
from threading import RLock
from weakref import WeakValueDictionary
from .program import OutputAxis
from .output_layout import OutputBlock, OutputLayout
from ..numerics.program_models import ObservableDescriptor


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PhysicalLayout:
    model_program: Any
    views: tuple
    observables: tuple[ObservableDescriptor, ...]
    precision: str
    output_layout: OutputLayout
    output_expressions: tuple
    _native_providers: Any
    _native_cores: Any
    _function_lock: Any

    def __init__(self, model_program, views, observables, *, precision='float64', output_expressions=()):
        if precision not in ('float32', 'float64'):
            raise ValueError('precision must be float32 or float64')
        views, observables = tuple(views), tuple(observables)
        if any(view.model_program is not model_program for view in views):
            raise ValueError('all Views must bind this same ModelProgram')
        supported = {'S', 'Y', 'Z', 'operator', 'response_element', 'diagonal_root',
                     'operator_element_root', 'hybridized_pole', 'transfer_zero', 'residue_coupling',
                     'static_capacitance', 'static_capacitance_element'}
        for descriptor in observables:
            if descriptor.kind not in supported:
                raise TypeError(f'unknown declared native numerical route: {descriptor.kind}')
            if descriptor.kind == 'response_element' and len(descriptor.indices) != 3:
                raise ValueError('response_element indices are (family index S/Y/Z, output, input)')
        axes, blocks, offset = [], [], 0
        for ordinal, descriptor in enumerate(observables):
            view = views[descriptor.view_index]
            count = descriptor.frequency_slice[1] - descriptor.frequency_slice[0]
            scalar = descriptor.kind in ('response_element', 'diagonal_root', 'operator_element_root', 'hybridized_pole', 'transfer_zero', 'residue_coupling')
            shape = () if scalar and (count == 1 or descriptor.branch_indices) else (count,) if scalar else (count, len(view.terminal_ids), len(view.terminal_ids))
            if descriptor.kind.startswith('static_capacitance'):
                shape = ((len(descriptor.retained_indices),) * 2
                         if descriptor.kind == 'static_capacitance' else ())
            key = descriptor.quantity_key or str(ordinal)
            components = ('real',) if descriptor.projection is not None else ('real', 'imag')
            blocks.append(OutputBlock(ordinal, key, shape, components, descriptor.si_unit, offset))
            offset += prod(shape) * len(components)
            for component in components:
                axes.append(OutputAxis(key, component, descriptor.si_unit,
                                       tuple(view.terminal_ids), shape))
        output_expressions = tuple(output_expressions)
        if output_expressions:
            axes, blocks, offset = [], [], 0
            for ordinal, expression in enumerate(output_expressions):
                components = ('real', 'imag') if expression.complex_output else ('real',)
                blocks.append(OutputBlock(ordinal, expression.quantity_key, expression.shape, components, expression.si_unit, offset))
                offset += prod(expression.shape) * len(components)
                for component in components:
                    axes.append(OutputAxis(expression.quantity_key, component, expression.si_unit, expression.coordinates, expression.shape))
        for name, value in (('model_program', model_program), ('views', views), ('observables', observables),
                            ('precision', precision), ('output_expressions', output_expressions), ('output_layout', OutputLayout.from_blocks(axes, blocks)),
                            ('_native_providers', WeakValueDictionary()), ('_native_cores', WeakValueDictionary()), ('_function_lock', RLock())):
            object.__setattr__(self, name, value)

    def _capabilities(self, inputs, *, affine_evidence=None):
        # Original binary64 physical classification precedes selected casting.
        fields = self.model_program.physical_fields(inputs.theta, affine_evidence=affine_evidence)
        return tuple(view.capability(fields) if any(
            item.view_index == index and not item.kind.startswith('static_capacitance')
            for item in self.observables) else False
            for index, view in enumerate(self.views))
