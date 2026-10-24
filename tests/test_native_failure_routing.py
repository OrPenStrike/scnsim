"""Owned checkify error transport; no solver or synthetic numerical Result."""
from __future__ import annotations

import unittest

import jax
from jax._src import checkify
import numpy as np

from scnsim.execution.physical_analysis import (
    _native_chunk_sparse_exception, _raise_unowned_native_chunk_error,
)
from scnsim.numerics.native_sparse import (
    ArchivedPrimalCheck, SparseEquationCheck, _archive_layout,
)


def _equation(site=(256,)):
    return SparseEquationCheck(
        None, site, 'primal', 7, 0, ('responses', 0, ('Z',)),
        np.int32(7), np.float64(np.inf), np.float64(np.inf),
        np.float64(np.inf), np.bool_(True),
    )


def _error(rows):
    """Construct the real Error envelope from real exception pytrees/predicates."""
    predicates, codes, metadata, payloads = {}, {}, {}, {}
    for code, (exception, predicate) in enumerate(rows, 1):
        values, tree = jax.tree.flatten(exception)
        effect = exception.get_effect_type()
        predicate = np.asarray(predicate, dtype=np.bool_)
        predicates[effect] = predicate
        codes[effect] = np.full(predicate.shape, code, dtype=np.int32)
        metadata[code] = tree
        payloads[effect] = [np.broadcast_to(np.asarray(value), predicate.shape + np.asarray(value).shape)
                            for value in values]
    return checkify.Error(predicates, codes, metadata, payloads)


def _payload_bits(error):
    return tuple((tuple(np.asarray(item).shape), np.asarray(item).dtype.str,
                  np.asarray(item).tobytes())
                 for values in error._payload.values() for item in values)


class NativeFailureRoutingTests(unittest.TestCase):
    def test_owned_scalar_and_batched_equations_keep_original_payload(self):
        for predicate in (True, [True], [False, True, False]):
            with self.subTest(predicate=predicate):
                error = _error([(_equation(), predicate)])
                original = _payload_bits(error)
                exception = error.get_exception()
                self.assertTrue(_native_chunk_sparse_exception(exception))
                _raise_unowned_native_chunk_error(error)
                self.assertEqual(_payload_bits(error), original)
                if np.asarray(predicate).ndim:
                    self.assertIs(type(exception), checkify.BatchedError)
                    expected = [index for index, flag in enumerate(predicate) if flag]
                    self.assertEqual(list(exception.error_mapping), [(index,) for index in expected])
                    self.assertEqual([int(item.payload[0]) for item in exception.error_mapping.values()],
                                     [7] * len(expected))

    def test_archived_equations_are_owned_in_scalar_and_batch_wrappers(self):
        for predicate in (True, [False, True]):
            with self.subTest(predicate=predicate):
                # Use the actual archive producer, including its lane/code payload.
                error = _archive_layout(_error([(_equation(), predicate)]))
                original = _payload_bits(error)
                self.assertIs(type(error.get_exception()), ArchivedPrimalCheck)
                self.assertTrue(_native_chunk_sparse_exception(error.get_exception()))
                _raise_unowned_native_chunk_error(error)
                self.assertEqual(_payload_bits(error), original)
                # The exact real batch container can also hold an archived leaf.
                batch = checkify.BatchedError({(0,): error.get_exception()})
                self.assertTrue(_native_chunk_sparse_exception(batch))

    def test_unknown_and_mixed_errors_preserve_original_throw(self):
        def unowned_check():
            checkify.check(False, 'unowned check')

        recorded, _ = checkify.checkify(unowned_check)()
        unknown = recorded.get_exception()
        # BatchedError takes its traceback from the first failing lane.
        owned = _equation()
        owned.traceback_info = unknown.traceback_info
        errors = [
            _error([(unknown, True)]),
            _error([(unknown, [True])]),
            _error([(owned, [True, False]), (unknown, [False, True])]),
            _archive_layout(_error([(unknown, True)])),
        ]
        for error in errors:
            with self.subTest(exception=type(error.get_exception()).__name__):
                self.assertFalse(_native_chunk_sparse_exception(error.get_exception()))
                original = _payload_bits(error)
                with self.assertRaises(checkify.JaxRuntimeError) as expected:
                    error.throw()
                with self.assertRaises(checkify.JaxRuntimeError) as actual:
                    _raise_unowned_native_chunk_error(error)
                self.assertEqual(str(actual.exception), str(expected.exception))
                self.assertIs(type(actual.exception.__cause__), type(expected.exception.__cause__))
                self.assertEqual(_payload_bits(error), original)

    def test_invalid_wrappers_are_not_admitted_and_inactive_checks_do_not_throw(self):
        batch = checkify.BatchedError({(0,): _equation()})
        batch.error_mapping.clear()
        self.assertFalse(_native_chunk_sparse_exception(batch))
        batch.error_mapping['invalid lane'] = _equation()
        self.assertFalse(_native_chunk_sparse_exception(batch))
        values, tree = jax.tree.flatten(_equation())
        archive = ArchivedPrimalCheck(tree, (*values, np.bool_(True), np.int32(1)))
        archive.payload = ()
        self.assertFalse(_native_chunk_sparse_exception(archive))
        self.assertFalse(_native_chunk_sparse_exception(None))
        error = _error([(_equation(), [False, False])])
        self.assertIsNone(error.get_exception())
        _raise_unowned_native_chunk_error(error)


if __name__ == '__main__':
    unittest.main()
