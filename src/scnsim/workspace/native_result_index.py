"""Publish compact native Optimization projections at the Workspace boundary.

Julia result/ledger bytes remain numerical authority. This separate sealed index
copies only presentation scalars and routing locators, one generation at a time;
never dependencies, CMA populations, discretization, or all detailed candidates.
The attempt owner includes the returned root ref in its immutable completion.
Readers do not rebuild a missing index or fall back to replaying native history.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..canonical import canonical_json_bytes, sha256_hex
from .artifacts import _read_json_artifact
from .primitives import _inside
from .storage import _atomic_write, _fsync_directory
from .validation.common import _integrity


def _immutable(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class NativeResultIndex:
    root_ref: Mapping[str, object]
    metadata: Mapping[str, object]


def _quantity_fields(value):
    if not isinstance(value, Mapping):
        return {'value_f64': None, 'si_unit': None, 'dimensionality': None}
    return {'value_f64': value.get('si_value_f64'),
            'si_unit': value.get('si_unit'),
            'dimensionality': value.get('dimensionality')}


def _failure_summary(failure):
    if not isinstance(failure, Mapping):
        return None
    return {'kind': failure.get('kind'), 'stage': failure.get('stage'),
            'detail': failure.get('message')}


def native_candidate_summary(candidate):
    """Copy exact encoded scalar records without decode/re-encode arithmetic."""
    outcome = candidate['outcome']
    components = []
    for component in outcome['objective_components']:
        terms = []
        for term in component.get('terms', ()):
            terms.append({'term_ordinal': term.get('term_ordinal'),
                          'status': term.get('status'),
                          **_quantity_fields(term.get('value')),
                          'ref_lineage': term.get('ref_lineage'),
                          'failure': _failure_summary(term.get('failure'))})
        components.append({'objective_id': component.get('objective_id'),
                           'status': component.get('status'),
                           **_quantity_fields(component.get('value')),
                           'normalized_residual_f64': component.get('normalized_residual_f64'),
                           'weighted_cost_f64': component.get('weighted_cost_f64'),
                           'terms': terms})
    return {'evaluation_ordinal': candidate['evaluation_ordinal'],
            'generation': candidate['generation'],
            'status': outcome['status'], 'cost_f64': outcome.get('cost_f64'),
            'parameters': candidate['parameters'], 'components': components}


def _write(directory, relative, value, *, role, identifier):
    path = _inside(directory, relative)
    if path.exists():
        raise _integrity('Native index publication would overwrite existing evidence.', path=relative)
    raw = canonical_json_bytes(value)
    _atomic_write(path, raw)
    return {'id': identifier, 'path': relative, 'sha256': sha256_hex(raw),
            'byte_length': len(raw), 'media_type': 'application/json', 'role': role}


def build_native_result_index(*, binding_identity, request, attempt, receipt, result, directory):
    """Build an index before the caller's one immutable attempt publication.

    ``binding_identity`` contains external expected Plan and instance scalars.
    ``request/attempt/receipt/result`` are the actual validated native envelopes;
    ``directory`` is their allocated staging directory. The owner still verifies
    and atomically promotes the whole attempt and binds this root in completion.
    """
    directory = Path(directory)
    request_sha = sha256_hex(canonical_json_bytes(dict(request)))
    attempt_sha = sha256_hex(canonical_json_bytes(dict(attempt)))
    result_raw = canonical_json_bytes(dict(result))
    result_sha = sha256_hex(result_raw)
    if (request['plan_sha256'] != binding_identity['plan_sha256']
            or attempt['request_sha256'] != request_sha
            or receipt['request_sha256'] != request_sha
            or receipt['attempt_sha256'] != attempt_sha
            or receipt['result_sha256'] != result_sha
            or receipt['outcome'] != 'success'
            or result['request_sha256'] != request_sha
            or result['attempt_sha256'] != attempt_sha
            or result['result_kind'] != 'optimization'):
        raise _integrity('Native index inputs do not bind one successful Optimization attempt.')
    native_ref = {'id': 'native_result', 'path': 'result.json',
                  'sha256': result_sha, 'byte_length': len(result_raw),
                  'media_type': 'application/json', 'role': 'native_result'}
    attempt_identity = {'sha256': attempt_sha, 'ordinal': attempt['ordinal'],
                        'directory': attempt['directory']}
    identity = {'workspace_instance_id': binding_identity['workspace_instance_id'],
                'plan_sha256': binding_identity['plan_sha256'],
                'request_sha256': request_sha, 'attempt_identity': attempt_identity,
                'native_result_ref': native_ref}
    artifacts = _inside(directory, 'artifacts')
    index_directory = _inside(directory, 'artifacts/native-index')
    if not artifacts.exists():
        artifacts.mkdir(); _fsync_directory(directory)
    if index_directory.exists():
        raise _integrity('Native index staging directory already exists.')
    index_directory.mkdir(); _fsync_directory(artifacts)

    baseline = result['baseline']
    if baseline['evaluation_ordinal'] != 0 or baseline['generation'] != 0:
        raise _integrity('Native baseline locator is not the baseline occurrence.')
    baseline_locator = {'native_result_ref': native_ref, 'member': 'baseline',
                        'evaluation_ordinal': 0, 'generation': 0}
    baseline_ref = _write(directory, 'artifacts/native-index/baseline.json',
        {'schema': 'scnsim.native_baseline_index', 'schema_version': 1, **identity,
         'locator': baseline_locator, 'summary': native_candidate_summary(baseline)},
        role='native_baseline_index', identifier='native_baseline_index')
    baseline_locator = {**baseline_locator, 'summary_ref': baseline_ref}
    best_ordinal = result['best']['evaluation_ordinal']
    best_locator = baseline_locator if best_ordinal == 0 else None
    generations = []
    count = 0
    previous = None
    population = request['spec']['optimizer']['resolved_population_size']
    for generation, artifact in enumerate(result['ledger_artifacts'], 1):
        ledger = _read_json_artifact(directory, artifact)
        # Replayed ledgers retain their original source attempt identity. Their
        # exact hash chain, not the current attempt SHA, owns that ancestry.
        if (ledger['schema'] != 'scnsim.optimization_ledger'
                or ledger['schema_version'] != 4
                or ledger['request_sha256'] != request_sha
                or ledger['generation'] != generation
                or ledger['previous_ledger_sha256'] != previous
                or ledger['population_size'] != population
                or len(ledger['candidates']) != population):
            raise _integrity('Native index generation does not match its selected ledger chain.', generation=generation)
        rows = []
        first = count + 1
        for row_index, candidate in enumerate(ledger['candidates']):
            count += 1
            if candidate['evaluation_ordinal'] != count or candidate['generation'] != generation:
                raise _integrity('Native index candidate locator is not in declared ledger order.', generation=generation)
            locator = {'native_ledger_ref': dict(artifact), 'row_index': row_index,
                       'evaluation_ordinal': count, 'generation': generation,
                       'source_attempt_sha256': ledger['attempt_sha256']}
            rows.append({'locator': locator, 'summary': native_candidate_summary(candidate)})
            if count == best_ordinal:
                best_locator = locator
        block = {'schema': 'scnsim.native_generation_index', 'schema_version': 1,
                 **identity, 'generation': generation, 'native_ledger_ref': dict(artifact),
                 'previous_ledger_sha256': previous,
                 'source_attempt_sha256': ledger['attempt_sha256'],
                 'first_ordinal': first, 'row_count': len(rows), 'rows': rows}
        summary_ref = _write(directory, f'artifacts/native-index/generation-{generation:06d}.json',
            block, role='native_generation_index', identifier=f'native_generation_index_{generation:06d}')
        if best_locator is not None and best_locator['generation'] == generation:
            best_locator = {**best_locator, 'summary_ref': summary_ref}
        generations.append({'generation': generation, 'first_ordinal': first,
                            'row_count': len(rows), 'summary_ref': summary_ref,
                            'native_ledger_ref': dict(artifact)})
        previous = artifact['sha256']
        # No generation's decoded numerical/dependency body survives iteration.
        del ledger, rows, block, candidate
    if len(generations) != result['completed_generations'] or best_locator is None:
        raise _integrity('Native index count or best locator differs from the terminal result.')
    root = {'schema': 'scnsim.native_result_index', 'schema_version': 1, **identity,
            'generation_index_refs': generations, 'generation_count': len(generations),
            'population_candidate_count': count, 'baseline_locator': baseline_locator,
            'best_locator': best_locator}
    ref = _write(directory, 'artifacts/native-index/root.json', root,
                 role='native_result_index', identifier='native_result_index')
    return NativeResultIndex(_immutable(ref), _immutable(root))
