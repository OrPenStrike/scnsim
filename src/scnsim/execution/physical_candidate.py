"""Same-frame optimizer evidence serialization, independent of controller math.

The ObjectiveProgram has already computed ordered expression values, residuals,
contributions and total cost. This adapter only encodes them and projects actual
leaf primal locators into the existing candidate envelope. It performs no model
lowering, objective aggregation, derivative computation, solve or persistence.
"""
from __future__ import annotations

import json

from ..numeric_encoding import bits, complex_record
from ..numerics.evidence import numerical_error
from .physical_analysis import point_contexts, project_primal_bodies
from .quantities import expression_leaves, quantity_body_id
from .quantity_policy import diagonal_passive_policy, project_quantity


def encode_physical_candidate(*, objective_program, outcome, parameters, objective_records, affine_evidence=()):
    """Encode a synchronized ObjectiveOutcome with original requested selectors.

    ``objective_records`` is the ordered finalized canonical request declaration,
    retaining the original magnitude selector even when zero-target loss uses a
    smooth raw squared operand. Derivative availability does not alter primal
    success; the coordinator owns local-stage partial termination and callbacks.
    """
    program = objective_program.quantity_program
    metadata = program.observable_metadata()
    inputs = program.inputs(parameters)
    contexts = point_contexts(program, inputs)
    bodies = project_primal_bodies(program, outcome.primal, inputs)
    dependencies = {}
    for row in metadata:
        result = bodies[row['observable_index']]
        if row['kind'] == 'diagonal_root':
            result = diagonal_passive_policy(result,
                node_count=len(program.program.views[
                    program.program.observables[row['observable_index']].view_index].node_ids),
                precision=program.program.precision)
            if result.failure is not None:
                raise numerical_error(result.failure)
            bodies[row['observable_index']] = result
        body_id, body = quantity_body_id(result)
        dependencies[body_id] = body
    objectives = []
    for ordinal, declaration in enumerate(objective_records):
        rows = [row for row in metadata if row['quantity_key'] == declaration['id'] and not row['hidden']]
        selectors = expression_leaves(declaration['quantity'])
        terms = []
        for term_ordinal, (selector, row) in enumerate(zip(selectors, rows, strict=True)):
            result = bodies[row['observable_index']]
            # Projection is presentation of the actual stored primitive; loss
            # remains the original same-frame result, including smooth abs².
            value = project_quantity(selector, result)
            actual = (result.root_omega_rad_s if result.root_omega_rad_s is not None else
                      result.response_value if result.response_value is not None else result.coupling_rad_s)
            terms.append({
                'term_ordinal': term_ordinal, 'selector': selector, 'status': 'success',
                'value_f64': bits(value), 'actual_complex': complex_record(actual),
                'body_id': quantity_body_id(result)[0],
                'lineage': contexts[row['observable_index']]['lineage'],
            })
        objectives.append({
            'id': declaration['id'], 'status': 'success',
            'value_f64': bits(float(outcome.values[ordinal])),
            'normalized_residual_f64': bits(float(outcome.normalized_residuals[ordinal])),
            'cost_f64': bits(float(outcome.contributions[ordinal])), 'terms': terms,
        })
    return {**({'affine_evidence':affine_evidence} if affine_evidence else {}),
            'parameters': parameters, 'objectives': objectives,
            'cost_f64': bits(float(outcome.total_cost)), 'failure': None,
            'dependencies': dependencies}


def encode_physical_candidate_failure(*, parameters, objective_records, error,
                                      observed_dependencies=None, observed_result=None,
                                      completed_objectives=(), objective_ordinal=None,
                                      term_ordinal=None, affine_evidence=()):
    """Encode an already classified actual numerical failure, never infer values.

    The coordinator alone decides which existing numerical failure codes are
    candidate failures. Infrastructure, callback and storage errors remain fatal.
    Unknown graph failure position leaves terms unevaluated rather than inventing
    a failed leaf or a typed root/coupling body that was never observed.
    """
    from ..numeric_encoding import record_bytes
    failure = getattr(error, 'failure', None)
    if failure is None:
        failure_record = {'kind': error.kind, 'stage': error.stage, 'detail': str(error),
                          'evidence_bytes': json.loads(record_bytes(error.evidence))}
    else:
        failure_record = {'kind': failure.kind, 'stage': failure.stage, 'detail': failure.detail,
                          'evidence_bytes': json.loads(failure.evidence_bytes)}
    failure_record['phase'] = 'quantity_evaluation'
    if objective_ordinal is not None:
        failure_record['objective_ordinal'] = objective_ordinal
    if term_ordinal is not None:
        failure_record['term_ordinal'] = term_ordinal
    dependencies = dict(observed_dependencies or {})
    result = observed_result if observed_result is not None else getattr(error, 'result', None)
    if result is not None:
        body_id, body = quantity_body_id(result)
        dependencies[body_id] = body
    objectives = list(completed_objectives)
    for ordinal in range(len(objectives), len(objective_records)):
        declaration = objective_records[ordinal]
        terms = []
        for index, selector in enumerate(expression_leaves(declaration['quantity'])):
            row = {'term_ordinal': index, 'selector': selector, 'status': 'unevaluated'}
            if ordinal == objective_ordinal and index == term_ordinal:
                row.update(status='failure', failure=failure_record)
            terms.append(row)
        row = {'id': declaration['id'], 'status': 'unevaluated', 'terms': terms}
        if ordinal == objective_ordinal:
            row.update(status='failure', failure=failure_record)
        objectives.append(row)
    return {**({'affine_evidence':affine_evidence} if affine_evidence else {}),
            'parameters': parameters, 'objectives': objectives,
            'cost_f64': bits(float('inf')), 'failure': failure_record,
            'dependencies': dependencies}
