"""Canonical Quantity declarations lowered into one physical arithmetic program.

This frontend owns mesh planning, expression units and real output coordinates.
It retains declaration bytes and static programs, never a candidate point, native
factor, Workspace, Run or derivative tape. Runtime operands are packed afresh.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import hashlib
import numpy as np

from ..canonical import canonical_json_bytes, float64_from_hex
from .compiler import parameter_key
from .mesh import quantity
from .program import MeshPlan
from .model_program import compile_model_program, plan_mesh
from .view_program import compile_view_program
from ..numerics.program_models import ObservableDescriptor, ProgramInputs
from .physical_layout import PhysicalLayout


@dataclass(frozen=True, slots=True)
class QuantityLeaf:
    key: str
    observable_index: int
    si_unit: str
    axes_bytes: bytes
    declaration_bytes: bytes

    @property
    def axes(self):
        return json.loads(self.axes_bytes)


@dataclass(frozen=True, slots=True)
class OutputExpression:
    quantity_key: str
    indices: tuple[int, ...]
    tree: tuple
    si_unit: str
    shape: tuple[int, ...]
    coordinates: tuple[str, ...]
    complex_output: bool
    has_absolute: bool

    def _evaluate(self, values):
        import jax.numpy as jnp
        # Tree leaf ordinals are local to this expression; callers supply
        # the shared frame's complete ordered observable values.
        values = tuple(values[index] for index in self.indices)
        def visit(node):
            kind = node[0]
            if kind == 'leaf':
                return values[node[1]] * node[2], jnp.asarray(False)
            if kind == 'absolute':
                value, unavailable = visit(node[1])
                return jnp.abs(value), unavailable | jnp.any(value == 0)
            if kind == 'difference':
                left, a = visit(node[1]); right, b = visit(node[2])
                return left - right, a | b
            value, unavailable = visit(node[1][0])
            for term in node[1][1:]:
                part, current = visit(term)
                value, unavailable = value + part, unavailable | current
            return value, unavailable
        return visit(self.tree)

    def apply(self, *values):
        return self._evaluate(values)[0]

    def nondifferentiable(self, *values):
        return self._evaluate(values)[1]


@dataclass(frozen=True, slots=True)
class QuantityProgram:
    program: PhysicalLayout
    declarations: tuple[tuple[str, bytes], ...]
    leaves: tuple[QuantityLeaf, ...]
    frequencies_rad_s: tuple[complex, ...]
    branch_starts_rad_s: tuple[complex, ...]
    identity_bytes: bytes
    observable_metadata_bytes: tuple[bytes, ...] = ()

    def native_program(self, inputs, *, program_owner=None, admit_core=False, affine_evidence=None):
        """Admit one static native partition after actual physical classification.

        Inputs are used only during acquisition, never retained by the admitted
        function. Concurrent callers use the atomically admitted same callable.
        """
        from ..numerics.native_program import compile_native_program, acquire_native_core
        capabilities = self.program._capabilities(inputs, affine_evidence=affine_evidence)
        key = ('native_program', capabilities)
        with self.program._function_lock:
            admitted = self.program._native_providers.get(key)
            existing_core = self.program._native_cores.get(capabilities)
        core = acquire_native_core(self, capabilities, owner=program_owner, admit=admit_core,
                                   existing=existing_core)
        with self.program._function_lock:
            if program_owner is None:
                core = self.program._native_cores.setdefault(capabilities, core)
            else:
                self.program._native_cores[capabilities] = core
        if admitted is None or admitted.core is not core:
            built = compile_native_program(self, inputs, capabilities=capabilities, core=core)
            with self.program._function_lock:
                previous = self.program._native_providers.get(key)
                same_core = previous is not None and (previous.core is core or
                    (program_owner is None and previous.core.structural_identity == core.structural_identity))
                if same_core:
                    admitted = previous
                else:
                    self.program._native_providers[key] = built
                    admitted = built
        return admitted

    @property
    def output_layout(self):
        return self.program.output_layout

    @property
    def identity(self):
        return json.loads(self.identity_bytes)

    @property
    def program_sha256(self):
        return hashlib.sha256(self.identity_bytes).hexdigest()

    def projected_output_layout(self):
        """Detached static primal output shapes/dtypes from expression lowering.

        This is the same ordered projection authority used by native execution,
        not an eval_shape/solve or an inference from packed real row counts.
        Original binary64 parameter operands do not widen selected output dtype.
        """
        real = 'float32' if self.program.precision == 'float32' else 'float64'
        complex_dtype = 'complex64' if self.program.precision == 'float32' else 'complex128'
        return tuple({'quantity_key': expression.quantity_key,
                      'shape': expression.shape,
                      'dtype': complex_dtype if expression.complex_output else real,
                      'si_unit': expression.si_unit}
                     for expression in self.program.output_expressions)

    def quantity_metadata(self):
        return tuple({'key': leaf.key, 'observable_index': leaf.observable_index,
                      'si_unit': leaf.si_unit, 'axes': leaf.axes,
                      'declaration': json.loads(leaf.declaration_bytes)} for leaf in self.leaves)

    def require_primal(self, outcome):
        from ..numerics.evidence import require_primal
        return require_primal(outcome)

    def inputs(self, parameter_set, *, branch_starts_rad_s=None):
        # Preserve original binary64 effective operands for physical validation;
        # The native core casts only at its selected numerical boundary.
        theta = self.program.model_program.fields.layout.pack(parameter_set, dtype=np.float64)
        starts = self.branch_starts_rad_s if branch_starts_rad_s is None else branch_starts_rad_s
        return ProgramInputs(theta, np.asarray(self.frequencies_rad_s, dtype=np.complex128),
                             np.asarray(starts, dtype=np.complex128))


    def point_metadata(self, inputs):
        """Detached actual-point model/View metadata, without numerical solves.

        Accept the same complete ProgramInputs used for the outcome. Physical
        validation is owned by evaluation; this projects the shared field and
        routing expressions rather than rebuilding a numeric model or View.
        """
        model = self.program.model_program
        fields = model.fields.apply(np.asarray(inputs.theta, dtype=np.float64), xp=np)
        rd = np.float32 if self.program.precision == 'float32' else np.float64
        reference = np.asarray(model.port_resistance, dtype=rd)
        # Probe compensation changes the physical load mask, not the declared
        # logical Port reference resistance used for normalization.
        static_only = tuple(not any(item.view_index == index and not item.kind.startswith('static_capacitance')
                                    for item in self.program.observables)
                            for index in range(len(self.program.views)))
        views = tuple(dict(view_index=index, lineage=(dict(declaration=json.loads(view.declaration_bytes),
                           transforms='runtime_coordinate_transform', port_realizable=False)
                           if static_only[index] else view.lineage(fields)),
                           terminal_ids=list(view.terminal_ids), node_ids=list(view.node_ids),
                           probe_loads={port: 'compensated' if mask == 0 else 'raw'
                                        for port, mask in zip(model.port_ids, view.port_mask, strict=True)},
                           reference_coordinate_ids=list(model.port_ids),
                           reference_impedances_ohm=[float(value) for value in np.diag(reference)],
                           load_mask=[float(value) for value in view.port_mask])
                      for index, view in enumerate(self.program.views))
        return dict(discretization=model.point_discretization(fields), views=views,
                    root_search_requests=self.root_search_metadata(),
                    observables=tuple(dict(observable_index=index, view_index=item.view_index)
                                      for index, item in enumerate(self.program.observables)))

    def root_search_metadata(self):
        """Declared extent from original Hints, independent of tracked anchors.

        This records search extent and the actual finite mesh authority. It
        does not override an authored resolution or assert continuum coverage.
        """
        from ..numerics.root_search import complex_frequency, resolve_region
        rows = {}
        for row in self.observable_metadata():
            start,spec = row['start_index'],row['spec']
            if start is None or 'search' not in spec:
                continue
            hint = complex_frequency(spec['root_hint'] if 'root_hint' in spec else spec['anchor'])
            real,imag = resolve_region(spec['search'],hint)
            rows[start] = dict(start_index=start,declaration=spec['search'],
                authored_hint_hz=hint,real_frequency_interval_hz=real,
                imag_frequency_interval_hz=imag,
                mesh_strategy=self.program.model_program.mesh.strategy)
        return tuple(rows.values())

    def observable_metadata(self):
        """Detached ordered leaf declarations, parallel to program.observables."""
        return tuple(json.loads(row) for row in self.observable_metadata_bytes)

    def root_metadata(self):
        """One ordered root-row authority for declaration and numerical consumers."""
        rows = {}
        for descriptor,metadata in zip(self.program.observables,self.observable_metadata(),strict=True):
            if descriptor.branch_indices:
                index = descriptor.branch_indices[0]
                rows.setdefault(index, {'start_index': index, 'kind': descriptor.kind,
                                        'view_index': descriptor.view_index,
                                        'quantity_key': descriptor.quantity_key,
                                        'indices': descriptor.indices,
                                        'spec': metadata['spec'], 'view': metadata['view']})
        return tuple(rows[index] for index in sorted(rows))


    def direction_inputs(self, inputs, encoded_direction):
        rows = encoded_direction['direction'] if isinstance(encoded_direction, dict) else encoded_direction
        directions = {parameter_key(row['parameter']): quantity(row['value']) for row in rows}
        axes = self.program.model_program.fields.layout.axes
        theta = np.zeros_like(inputs.theta)
        for index, axis in enumerate(axes):
            if axis.key in directions:
                if axis.matrix_field is not None:
                    raise TypeError('whole-RLGC directional input requires its declared structured operand')
                theta[index] = directions[axis.key]
        return ProgramInputs(theta, np.zeros_like(inputs.omega_rad_s), np.zeros_like(inputs.branch_starts_rad_s))

    def weight_vector(self, encoded_weights):
        rows = encoded_weights['weights'] if isinstance(encoded_weights, dict) else encoded_weights
        weights = np.zeros(len(self.output_layout.rows), dtype=np.float64)
        for item in rows:
            coordinate = item['output']
            position = self.output_layout.row_index(coordinate['quantity_label'],
                tuple(coordinate['index']), coordinate['component'])
            weights[position] = quantity(item['value'])
        return weights


def _frequency_grid(record):
    # Both current producers preserve ordered scalar envelopes: public
    # sensitivity declares an array envelope, prepared Direct declares rows.
    if isinstance(record, dict):
        if record['type'] != 'quantity_array':
            raise TypeError('frequency declaration requires its canonical quantity array')
        if tuple(record['shape']) != (len(record['values']),):
            raise ValueError('frequency array shape disagrees with its ordered scalar values')
        return tuple(record['values'])
    return tuple(record)


def _unit_factor(source, target):
    if source == target:
        return 1.0
    if source == 'radian / second' and target == 'hertz':
        return 1.0 / (2.0 * math.pi)
    if source == 'hertz' and target == 'radian / second':
        return 2.0 * math.pi
    raise ValueError('Quantity expression terms do not share a convertible public unit convention')


def compile_quantity_program(plan_document, quantity_declarations, parameter_set, *,
                             default_view=None, mesh=None, bounds=None,
                             precision='float64', authorized=frozenset(), program_owner=None, admit_program=False):
    """Compile static declarations once; point values remain runtime operands."""
    mesh_plan = _compile_mesh(plan_document, parameter_set, mesh, bounds, authorized)
    # The declared topology/mesh and operand schema suffice for static lookup.
    # Runtime scalar/matrix coefficient bytes never enter this acquisition key.
    bindings = []
    for row in parameter_set['bindings']:
        value = row['value']
        if value['type'] == 'rlgc':
            metadata = dict(value)
            for name in ('resistance', 'inductance', 'conductance', 'capacitance'):
                field = name + '_per_length'
                metadata[field] = {key: item for key, item in value[field].items()
                                   if key != 'values_f64'}
        else:
            metadata = {key: item for key, item in value.items()
                        if key != 'si_value_f64'}
        bindings.append({'parameter': row['parameter'], 'value_schema': metadata})
    acquisition_key = ('quantity_program_declaration', canonical_json_bytes({
        'plan': plan_document, 'quantities': quantity_declarations,
        'default_view': default_view, 'precision': precision,
        'mesh': {'sections': [[list(path), count] for path, count in mesh_plan.sections],
                 'strategy': mesh_plan.strategy, 'derivations': json.loads(mesh_plan.derivation_bytes)},
        'bindings': bindings, 'authorized': sorted(authorized)}))
    if program_owner is not None:
        admitted = program_owner.get(acquisition_key)
        if admitted is not None:
            return admitted
    model = compile_model_program(plan_document, parameter_set, mesh=mesh_plan,
                                  authorized=authorized)
    views, view_indices, descriptors, frequencies, expressions, leaves, starts = [], {}, [], [], [], [], []
    branch_indices = {}
    observable_metadata = []
    declarations = tuple((key, canonical_json_bytes(value)) for key, value in quantity_declarations.items())

    def leaf(node, key, *, hidden=False):
        declaration = node.get('view', default_view)
        if declaration is None:
            raise TypeError('Quantity declaration requires a bound View')
        encoded = canonical_json_bytes(declaration)
        if encoded not in view_indices:
            view_indices[encoded] = len(views)
            views.append(compile_view_program(model, declaration))
        view_index = view_indices[encoded]
        view = views[view_index]
        kind, spec = node['type'], node['spec']
        indices, branches, projection = (), (), node.get('projection')
        dependencies, frequency_mode = (), None
        zero_charge_indices, retained_indices = (), ()
        coordinates = view.terminal_ids
        if kind in ('static_capacitance_projection', 'static_capacitance_element_projection'):
            descriptor_kind = spec['type']
            # The View alone owns the ordered partition. For this Spec the
            # complement carries q_e=0; PTC's resistive load mask is inert for C.
            retained_indices = view.selected if view.retained else tuple(range(len(view.node_ids)))
            zero_charge_indices = view.eliminated if view.retained else ()
            coordinates = tuple(view.node_ids[i] for i in retained_indices)
            grid, unit, complex_output = (), 'farad', False
            if kind == 'static_capacitance_projection':
                shape = (len(coordinates), len(coordinates))
            else:
                indices = (coordinates.index(spec['row']), coordinates.index(spec['column']))
                shape = ()
        elif kind == 'matrix_projection':
            family = projection
            grid = _frequency_grid(spec['frequencies'])
            unit = {'S': 'dimensionless', 'Y': 'siemens', 'Z': 'ohm'}[family]
            descriptor_kind, projection = family, None
            shape, complex_output = (len(grid), len(view.terminal_ids), len(view.terminal_ids)), True
        elif kind == 'operator_projection':
            if projection != 'matrix':
                raise TypeError('Operator projection requires the declared matrix selector')
            grid, descriptor_kind, projection = _frequency_grid(spec['frequencies']), 'operator', None
            unit = 'siemens / second'
            shape, complex_output = (len(grid), len(view.terminal_ids), len(view.terminal_ids)), True
        elif kind == 'response_element_projection':
            grid, descriptor_kind = (spec['frequency'],), 'response_element'
            family = spec['family']
            indices = (('S', 'Y', 'Z').index(family), view.terminal_ids.index(spec['output_coordinate']),
                       view.terminal_ids.index(spec['input_coordinate']))
            unit = {'S': 'dimensionless', 'Y': 'siemens', 'Z': 'ohm'}[family]
            shape, complex_output = (), projection is None
        elif kind in ('diagonal_root_projection', 'operator_element_root_projection',
                      'hybridized_pole_projection', 'transfer_zero_projection'):
            descriptor_kind = spec['type']
            anchor = spec['root_hint'] if descriptor_kind in ('diagonal_root', 'operator_element_root') else spec['anchor']
            if anchor['type'] == 'complex_quantity_f64':
                start_omega = complex(float64_from_hex(anchor['real_si_f64']), float64_from_hex(anchor['imag_si_f64'])) * (2 * math.pi)
            else:
                start_omega = complex(quantity(anchor) * (2 * math.pi))
            binding = canonical_json_bytes({'spec': spec, 'view': declaration})
            if binding not in branch_indices:
                branch_indices[binding] = len(starts)
                starts.append(start_omega)
            branches = (branch_indices[binding],)
            if descriptor_kind == 'diagonal_root':
                indices = (view.terminal_ids.index(spec['coordinate']),)
            elif descriptor_kind == 'operator_element_root':
                indices = (view.terminal_ids.index(spec['row']), view.terminal_ids.index(spec['column']))
            elif descriptor_kind == 'transfer_zero':
                indices = (('S', 'Y', 'Z').index(spec['family']), view.terminal_ids.index(spec['output_coordinate']),
                           view.terminal_ids.index(spec['input_coordinate']))
            else:
                if tuple(spec['coordinates']) != view.terminal_ids:
                    raise ValueError('HybridizedPole coordinates disagree with its ordered retained View')
            grid, shape, complex_output, unit = (), (), projection is None, 'hertz'
        elif kind == 'residue_coupling_projection':
            dependencies = tuple(leaf({'type': branch['type'] + '_projection', 'spec': branch,
                                       'view': declaration, 'projection': None}, key + ':branch:' + role, hidden=True)[0]
                                 for role, branch in (('a', spec['branch_a']), ('b', spec['branch_b'])))
            descriptor_kind, shape, unit = 'residue_coupling', (), 'radian / second'
            complex_output = projection is None
            frequency_mode = 'complex_root_midpoint' if spec['frequency'] == 'complex_root_midpoint' else 'fixed'
            grid = () if frequency_mode == 'complex_root_midpoint' else (spec['frequency'],)
        else:
            raise TypeError(f'Quantity frontend numerical route is not integrated: {kind}')
        start = len(frequencies)
        frequencies.extend(complex(quantity(value) * (2.0 * math.pi)) for value in grid)
        index = len(descriptors)
        axes = (('frequency', tuple(float(quantity(value)).hex() for value in grid)),
                ('output_coordinate', view.terminal_ids), ('input_coordinate', view.terminal_ids)) if shape else ()
        if descriptor_kind.startswith('static_capacitance'):
            axes = (('row_coordinate', coordinates), ('column_coordinate', coordinates)) if shape else ()
        descriptors.append(ObservableDescriptor(descriptor_kind, view_index, (start, len(frequencies)),
                                                indices=indices, branch_indices=branches, projection=projection,
                                                quantity_key=key, si_unit=unit, axes=axes,
                                                dependency_indices=dependencies, frequency_mode=frequency_mode,
                                                zero_charge_indices=zero_charge_indices, retained_indices=retained_indices))
        raw_unit = 'radian / second' if branches or descriptor_kind == 'residue_coupling' else unit
        locator = ({'kind': 'branch_results', 'start_index': branches[0]} if branches else
                   {'kind': 'quantity_results', 'observable_index': index} if descriptor_kind == 'residue_coupling' else
                   {'kind': 'observable_raw_values', 'observable_index': index})
        observable_metadata.append(canonical_json_bytes({
            'observable_index': index, 'quantity_key': key, 'kind': descriptor_kind,
            'selector': node['type'], 'projection': node.get('projection'),
            'spec': spec, 'view': declaration, 'declaration': node,
            'si_unit': unit, 'raw_si_unit': raw_unit, 'hidden': hidden,
            'start_index': branches[0] if branches else None, 'primal_locator': locator,
            **({'row_coordinates': list(coordinates), 'column_coordinates': list(coordinates),
                'zero_charge': [view.node_ids[i] for i in zero_charge_indices],
                'coordinate_order': list(view.node_ids),
                'original_coordinate_order': list(model.node_ids),
                'coordinate_transform_locator': {'kind': 'static_facts', 'observable_index': index},
                'finite_mesh': json.loads(mesh_plan.derivation_bytes)}
               if descriptor_kind.startswith('static_capacitance') else {})}))
        return index, unit, shape, complex_output, coordinates, axes

    for key, encoded in declarations:
        node = json.loads(encoded)
        local_indices = []
        metadata = []
        def lower(current, target_unit=None):
            kind = current['type']
            if kind not in ('quantity_sum', 'quantity_difference', 'quantity_absolute'):
                index, unit, shape, complex_output, coordinates, axes = leaf(current, key)
                local = len(local_indices); local_indices.append(index)
                metadata.append((unit, shape, complex_output, coordinates, axes))
                return ('leaf', local, _unit_factor(unit, target_unit or unit)), unit
            if kind == 'quantity_absolute':
                operand, unit = lower(current['operand'], target_unit)
                return ('absolute', operand), unit
            if kind == 'quantity_difference':
                left, unit = lower(current['left'], target_unit)
                right, _ = lower(current['right'], target_unit or unit)
                return ('difference', left, right), unit
            first, unit = lower(current['terms'][0], target_unit)
            terms = [first]
            terms.extend(lower(term, target_unit or unit)[0] for term in current['terms'][1:])
            return ('sum', tuple(terms)), unit
        tree, unit = lower(node)
        shape = metadata[0][1]
        if any(item[1] != shape for item in metadata):
            raise ValueError('Quantity expression operands have different physical axes')
        def expression_traits(current):
            if current[0] == 'leaf':
                return metadata[current[1]][2], False
            if current[0] == 'absolute':
                return False, True
            children = current[1] if current[0] == 'sum' else current[1:]
            traits = tuple(expression_traits(child) for child in children)
            return any(item[0] for item in traits), any(item[1] for item in traits)
        complex_output, has_absolute = expression_traits(tree)
        expression = OutputExpression(key, tuple(local_indices), tree, unit, shape,
                                      metadata[0][3], complex_output, has_absolute)
        expressions.append(expression)
        axis_records = []
        for axis_id, coordinates in metadata[0][4]:
            if axis_id == 'frequency':
                entries = [{'id': str(i), 'label': str(i), 'value': value, 'unit': 'hertz'}
                           for i, value in enumerate(coordinates)]
                axis_records.append({'id': axis_id, 'label': 'Frequency', 'unit': 'hertz', 'coordinates': entries})
            else:
                axis_records.append({'id': axis_id, 'label': axis_id.replace('_', ' '), 'unit': 'dimensionless',
                                     'coordinates': [{'id': value, 'label': value} for value in coordinates]})
        axis_records.append({'id': 'component', 'label': 'Component', 'unit': 'dimensionless',
                             'coordinates': [{'id': component, 'label': component, 'component': component}
                                             for component in (('real', 'imag') if complex_output else ('real',))]})
        leaves.append(QuantityLeaf(key, len(expressions) - 1, unit, canonical_json_bytes(axis_records), encoded))
    program = PhysicalLayout(model, views, descriptors, precision=precision, output_expressions=expressions)
    identity = {'schema': 'scnsim.quantity_program', 'schema_version': 1,
                'implementation': 'scnsim.jax.native_physical.v1', 'precision': precision,
                'plan_sha256': hashlib.sha256(canonical_json_bytes(plan_document)).hexdigest(),
                'declarations': [[key, json.loads(value)] for key, value in declarations],
                'mesh': json.loads(mesh_plan.derivation_bytes),
                'sections': [{'path': list(path), 'count': count} for path, count in mesh_plan.sections],
                'output_layout': program.output_layout.compact_record()}
    result = QuantityProgram(program, declarations, tuple(leaves), tuple(frequencies), tuple(starts), canonical_json_bytes(identity), tuple(observable_metadata))
    # Explicit owner admission retains declarations/functions only, never point
    # operands or checked frames. The caller owns registry lifetime.
    if program_owner is None:
        return result
    key = acquisition_key
    admitted = program_owner.get(key)
    if admitted is not None:
        return admitted
    if admit_program:
        program_owner.admit(key, result, kind='program', known_bytes=None)
        return program_owner.get(key, result)
    return result


def compile_point_quantity_program(plan_document, quantity_declarations, parameter_set, *,
                                   default_view=None, mesh=None, precision='float64',
                                   authorized=frozenset(), program_variants,
                                   program_owner=None, admit_program=False):
    """Ordinary-value variant admission using the existing actual-point mesh.

    Optimization and Sensitivity use their fixed mesh frontend instead. The
    operation owns this registry. Numeric scalar/material entries remain runtime
    operands; only actual section counts and structured metadata select a graph.
    """
    from .compiler import parameter_values, resolve_fields
    from .mesh import realize_line, mesh_from_record
    from .models import MeshSpec
    values = parameter_values(parameter_set)
    resolved, _ = resolve_fields(plan_document, values, authorized)
    policy = MeshSpec() if mesh is None else mesh_from_record(mesh)
    sections = []
    for leaf in plan_document['physical_leaves']:
        if leaf['model'] == 'transmission_line':
            path = tuple(leaf['path'])
            grid = realize_line(leaf, resolved[path, 'length'], resolved[path, 'rlgc'],
                                values, policy, {})
            sections.append({'path': list(path), 'count': grid['n_sections']})
    fixed = {'type': 'fixed_mesh', 'sections': sections, 'resolution': None}
    structured = []
    for row in parameter_set['bindings']:
        value = row['value']
        if value['type'] == 'rlgc':
            metadata = dict(value)
            for name in ('resistance', 'inductance', 'conductance', 'capacitance'):
                field = name + '_per_length'
                metadata[field] = {key: item for key, item in value[field].items() if key != 'values_f64'}
            structured.append({'parameter': row['parameter'], 'metadata': metadata})
    key = canonical_json_bytes({'plan': plan_document, 'quantities': quantity_declarations,
                                'view': default_view, 'mesh': fixed, 'precision': precision,
                                'structured_operands': structured, 'authorized': sorted(authorized)})
    result = program_variants.get(key)
    if result is None:
        result = compile_quantity_program(plan_document, quantity_declarations, parameter_set,
                                           default_view=default_view, mesh=fixed, precision=precision,
                                           authorized=authorized, program_owner=program_owner,
                                           admit_program=admit_program)
        result = program_variants.setdefault(key, result)
    return result


def _compile_mesh(plan, parameters, mesh, bounds, authorized):
    if mesh is None:
        return plan_mesh(plan, parameters, bounds=bounds, authorized=authorized)
    if mesh['type'] != 'fixed_mesh':
        raise TypeError('QuantityProgram mesh requires the canonical FixedMeshSpec record')
    if mesh['sections'] is not None:
        sections = tuple((tuple(row['path']), row['count']) for row in mesh['sections'])
        return MeshPlan(sections, 'declared_fixed', canonical_json_bytes(mesh), ())
    return plan_mesh(plan, parameters, bounds=bounds, authorized=authorized,
                     resolution=mesh['resolution'])
