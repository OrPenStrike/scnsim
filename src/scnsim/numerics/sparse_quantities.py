"""Direct quantities on shared sparse operators and selected dense boundaries.

The host owns dependency order and continuation. Finished branch results are
consumed verbatim: residue evaluation never performs another root search.
Global/eliminated equations remain sparse; dense arrays are retained boundaries,
local cofactors/SVDs or actual right-hand sides. Numerical failures retain their
specific quantity stage; native resource/program errors escape unchanged.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix
from .sparse_direct import selected_state, schur_reciprocity, network, dense_solve, maximum_abs, ratio, residual, tau
from .sparse_root import element_root, _quotient
from .determinants import (QuantityError, fail, Scaled, local_pair, positive_product,
                           positive_sum, scaled_ratio, ratio_le, Denominator)


def checked_state(system, omega, assemble, measurements):
    state, code = selected_state(system, assemble(omega, loaded=True, derivative=True), measurements)
    if code:
        numerical_code(code)
    return state


def numerical_code(code, *, transfer=False):
    from .jax_core import FAILURES
    kind, stage = FAILURES[code]
    fail(kind, 'transfer_denominator' if transfer else stage,
         'numerical operation did not satisfy its existing solve contract', source_stage=stage)


def same_bits(left, right, rd):
    uint = np.uint32 if rd == np.float32 else np.uint64
    def bits(value):
        return np.asarray(value, dtype=rd).view(uint).item()
    return bits(left.real) == bits(right.real) and bits(left.imag) == bits(right.imag)


def _row_norm_product(matrix, rd):
    """Product of row Euclidean norms without restoring large SI scales.

    Normalize real/imaginary components before squaring; combining each row's
    scale in mantissa/exponent form retains the prescribed row-2-norm product.
    SVDs and residuals continue to use the original unscaled matrix.
    """
    row_norms = []
    for row in matrix:
        scale = rd(max(np.max(np.abs(row.real)), np.max(np.abs(row.imag))))
        if scale == 0:
            row_norms.append((rd(0), 0))
        else:
            real, imag = row.real/scale, row.imag/scale
            norm = rd(np.sqrt(np.sum(real*real+imag*imag, dtype=rd)))
            row_norms.append(positive_product((scale, norm), rd, 'newton_certificate'))
    mantissa, exponent = positive_product((norm[0] for norm in row_norms), rd, 'newton_certificate')
    return mantissa, exponent+sum(norm[1] for norm in row_norms)


def hybridized(system, start, assemble, measurements):
    cd, rd, q = system.complex_dtype, system.real_dtype, len(system.selected)
    omega = cd(start)
    if not (np.isfinite(omega) and omega.real > 0):
        fail('numerical_resolution_unresolved', 'newton', 'hybridized-pole initialization is nonfinite or nonpositive')
    with measurements.phase('root_control', quantity='hybridized_pole'):
        for step in range(32):
            state = checked_state(system, omega, assemble, measurements)
            h, hp, *_ = local_pair(state[0], state[1], cd, rd)
            if hp == 0:
                fail('root_slope_unresolved', 'newton', 'hybridized-pole determinant derivative is zero')
            candidate = cd(omega-cd(h/hp))
            stagnant = same_bits(candidate, omega, rd)
            omega = candidate
            if stagnant:
                break
        F, Fp, Q, _, bound, X, _, _ = checked_state(system, omega, assemble, measurements)
        h, hp, _, _, determinant, _ = local_pair(F, Fp, cd, rd)
        with measurements.phase('quantity_svd', quantity='hybridized_pole', dimension=q):
            try:
                U, singular, Vh = np.linalg.svd(F)
            except np.linalg.LinAlgError as error:
                raise QuantityError('numerical_resolution_unresolved', 'rank_certificate', 'hybridized-pole SVD did not converge') from error
        if not (len(singular) >= 2 and singular[0] > 0 and singular[-1]/singular[0] <= tau(q, rd)
                and singular[-2]/singular[0] > tau(q, rd)):
            fail('numerical_resolution_unresolved', 'rank_certificate',
                 'retained operator does not have exactly one machine-null direction', singular_values=singular)
        v, u = np.asarray(Vh[-1].conj(), dtype=cd), np.asarray(U[:, -1], dtype=cd)
        pivot = int(np.argmax(np.abs(v)))
        phase = cd(np.exp(cd(-cd(1j)*rd(np.angle(v[pivot])))))
        v, u = np.asarray(v*phase, dtype=cd), np.asarray(u*phase, dtype=cd)
        right = _quotient(maximum_abs(F @ v, rd), maximum_abs(np.abs(F) @ np.abs(v), rd), rd)
        left = _quotient(maximum_abs(u.conj().T @ F, rd), maximum_abs(np.abs(u) @ np.abs(F), rd), rd)
        rows = _row_norm_product(F, rd)
        eta_det = scaled_ratio(rd(abs(h)), rows, rd)
        slope = cd(np.vdot(u, Fp @ v))
        scale = rd(np.sum(np.abs(u)*(np.abs(Fp) @ np.abs(v)), dtype=rd))
        correction = rd(np.inf) if hp == 0 else rd(rd(abs(cd(h/hp)))/rd(abs(omega)))
        x = np.zeros(system.n, dtype=cd)
        x[system.selected] = v
        if len(system.eliminated):
            x[system.eliminated] = -X @ v
        eta_q = _quotient(maximum_abs(Q @ x, rd), maximum_abs(bound @ np.abs(x), rd), rd)
        facts = dict(determinant=determinant, right_residual=right, left_residual=left,
                     determinant_residual=eta_det, determinant_normalizer=rows, full_operator_residual=eta_q,
                     slope_scale=scale, correction=correction, singular_values=singular,
                     newton_steps=step+1)
        closed = (np.isfinite(h) and np.isfinite(hp) and ratio_le(eta_det, tau(q, rd), rd)
                  and np.isfinite(right) and right <= tau(q, rd) and np.isfinite(left) and left <= tau(q, rd)
                  and np.isfinite(eta_q) and eta_q <= tau(system.n, rd) and scale > 0
                  and rd(abs(slope))/scale > tau(q, rd) and np.isfinite(correction)
                  and correction <= tau(q, rd) and omega.real > 0 and omega.imag <= 0)
        if not closed:
            fail('numerical_resolution_unresolved', 'newton_certificate',
                 'hybridized-pole machine-resolution certificate did not close', **facts)
        return dict(root_omega_rad_s=omega, root_slope=slope, null_vector=v), facts


def _family_state(system, omega, family, assemble, measurements):
    cd, rd = system.complex_dtype, system.real_dtype
    compensated = family == 'Z' and not system.view.port_realizable
    if compensated:
        assembled = assemble(omega, loaded=True, derivative=True, compensated=True)
    else:
        assembled = assemble(omega, loaded=not system.view.port_realizable, derivative=True)
    if system.view.port_realizable:
        values, status = network(system, omega, assembled, family, measurements, derivative=True)
        if status:
            numerical_code(status, transfer=True)
        S, Y, Z, Sp, Yp, Zp, H, Hp = values
    else:
        if family == 'S':
            fail('port_realizability', 'selected_network', 'S transfer zero requires a Port-realizable View')
        state, status = selected_state(system, assembled, measurements, compensated=compensated)
        if status:
            numerical_code(status, transfer=True)
        F, Fp, Q, Qp, _, _, _, _ = state[:8]
        divisor = cd(-cd(1j)*omega)
        Y = np.asarray(F/divisor, dtype=cd)
        Yp = np.asarray((Fp*divisor+cd(1j)*F)/cd(divisor*divisor), dtype=cd)
        # This Z route uses projected Y/Yp; high-only H would misstate its authority.
        H, Hp = (None, None) if compensated else (
            Q/divisor, (Qp*divisor+cd(1j)*Q)/cd(divisor*divisor))
        S = Sp = Z = Zp = None
        if family == 'Z':
            Z, status, _ = dense_solve(Y, np.eye(len(Y), dtype=cd), len(Y), 7, status, rd, measurements)
            if status:
                numerical_code(status, transfer=True)
            Zp = np.asarray(-Z @ Yp @ Z, dtype=cd)
    matrices = {'S': (S, Sp), 'Y': (Y, Yp), 'Z': (Z, Zp)}
    return matrices[family], Y, Yp, None if H is None else H.tocsc(), None if Hp is None else Hp.tocsc(), assembled


def _denominator(matrix, prime, system, measurements):
    try:
        denominator = Denominator(matrix, prime, system, measurements)
        value, derivative, facts = denominator.certificate_pair(verify=False)
    except QuantityError as error:
        if error.stage.startswith('determinant_'):
            raise QuantityError('numerical_resolution_unresolved', 'transfer_denominator', error.detail, **error.facts) from error
        raise
    return denominator, value, derivative, facts


def _checked_denominator_solve(denominator, rhs, system, stage):
    solution = denominator.solve(rhs)
    eta = residual(denominator.matrix, solution, rhs, system.real_dtype)
    if not (np.all(np.isfinite(solution)) and np.isfinite(eta) and eta <= tau(denominator.matrix.shape[0], system.real_dtype)):
        fail('numerical_resolution_unresolved', stage, 'required denominator solve did not close', residual=eta)
    return solution


def transfer_at(system, omega, job, assemble, measurements, *, final=False):
    cd, rd, output, input_ = system.complex_dtype, system.real_dtype, job.output_index, job.input_index
    (V, Vp), Y, Yp, H, Hp, assembled = _family_state(system, omega, job.family, assemble, measurements)
    value, slope = cd(V[output, input_]), cd(Vp[output, input_])
    if not np.isfinite(value):
        fail('numerical_resolution_unresolved', 'transfer_numerator_scale', 'transfer value is nonfinite')
    if not np.isfinite(slope):
        fail('root_slope_unresolved', 'transfer_slope_rank', 'transfer slope is nonfinite')
    selected, eliminated = system.selected, system.eliminated
    denominator_facts = None
    denominator = None
    denominator_input = None
    if job.family == 'Y':
        HRR, HRRp = H[selected][:, selected].toarray(), Hp[selected][:, selected].toarray()
        if len(eliminated):
            AD, ADp = H[eliminated][:, eliminated].tocsc(), Hp[eliminated][:, eliminated].tocsc()
            denominator, detd, detdp, denominator_facts = _denominator(AD, ADp, system, measurements)
            HER, HERp = H[eliminated][:, selected].toarray(), Hp[eliminated][:, selected].toarray()
            HRE, HREp = H[selected][:, eliminated], Hp[selected][:, eliminated]
            X = _checked_denominator_solve(denominator, HER, system, 'transfer_denominator')
            Xp = _checked_denominator_solve(denominator, np.asarray(HERp-ADp @ X, dtype=cd), system, 'transfer_denominator')
            N = HRR*detd-HRE @ (detd*X)
            Np = HRRp*detd+HRR*detdp-HREp @ (detd*X)-HRE @ (detdp*X+detd*Xp)
        else:
            detd, detdp = cd(1), cd(0)
            N, Np = HRR, HRRp
            denominator_facts = dict(original_dimension=0, determinant={'mantissa': cd(1), 'exponent': 0}, identity_solve_residual=rd(0))
        AN = np.asarray([[N[output, input_]]], dtype=cd)
        ANp = np.asarray([[Np[output, input_]]], dtype=cd)
        bound = system.csc(assembled[2], measurements)
        if not (np.all(np.isfinite(bound.data)) and np.isfinite(abs(omega)) and abs(omega) > 0):
            fail('numerical_resolution_unresolved', 'transfer_numerator_scale', 'absolute operator bound is unresolved')
        BH = bound/rd(abs(omega))
        if not np.all(np.isfinite(BH.data)) or np.any((bound.data > 0) & (BH.data == 0)):
            fail('numerical_resolution_unresolved', 'transfer_numerator_scale', 'admittance bound is not representable')
        if system.view.port_realizable:
            # The source boundary uses abs(B) abs(Go) abs(B).T, not abs of a
            # prematurely cancelled physical Port contribution.
            B = system.B
            weights = np.abs(np.asarray(B.values, dtype=cd))
            columns = np.asarray(B.cols)
            pair = weights[:, None]*np.abs(system.Go)[columns[:, None], columns[None, :]]*weights[None, :]
            extra = np.zeros(len(system.rows), dtype=rd)
            np.add.at(extra, system.B_pair_map, pair.ravel())
            BH = BH+system.csc(extra, measurements)
        BF = BH[selected][:, selected].toarray()
        if len(eliminated):
            BF += BH[selected][:, eliminated] @ np.abs(X)
        if not np.all(np.isfinite(BF)):
            fail('numerical_resolution_unresolved', 'transfer_numerator_scale', 'selected admittance bound is nonfinite')
        normalizer = positive_product((abs(detd), np.linalg.norm(BF[output])), rd)
    elif job.family == 'Z':
        rows = [i for i in range(len(Y)) if i != input_]
        columns = [i for i in range(len(Y)) if i != output]
        AN = Y[np.ix_(rows, columns)].copy() if rows else np.eye(1, dtype=cd)
        ANp = Yp[np.ix_(rows, columns)].copy() if rows else np.zeros((1, 1), dtype=cd)
        if (output+input_) % 2:
            AN[0] *= cd(-1); ANp[0] *= cd(-1)
        normalizer = positive_product((np.linalg.norm(Y[row]) for row in rows), rd)
        denominator_input = (csc_matrix(Y), csc_matrix(Yp))
    else:
        eye = np.eye(len(Y), dtype=cd)
        P, N = eye+system.Dk @ Y @ system.Dk, eye-system.Dk @ Y @ system.Dk
        Pp = system.Dk @ Yp @ system.Dk
        AN, ANp = np.zeros((len(Y)+1, len(Y)+1), dtype=cd), np.zeros((len(Y)+1, len(Y)+1), dtype=cd)
        AN[:-1, :-1], AN[:-1, -1], AN[-1, output] = P, N[:, input_], cd(-1)
        ANp[:-1, :-1], ANp[:-1, -1] = Pp, -Pp[:, input_]
        normalizer = positive_product((np.linalg.norm(row) for row in AN), rd)
        denominator_input = (csc_matrix(P), csc_matrix(Pp))
    try:
        numerator, numerator_slope, ANscaled, ANpscaled, determinant_facts, terms = local_pair(AN, ANp, cd, rd)
    except QuantityError as error:
        raise QuantityError('root_slope_unresolved', 'transfer_numerator_scale', error.detail, **error.facts) from error
    if denominator_input is not None:
        denominator, detd, _, denominator_facts = _denominator(*denominator_input, system, measurements)
    if not (np.isfinite(abs(numerator)) and np.isfinite(abs(numerator_slope))):
        fail('numerical_resolution_unresolved', 'transfer_numerator_scale', 'numerator pair is nonfinite')
    eta_n = scaled_ratio(rd(abs(numerator)), normalizer, rd)
    residual_pass = ratio_le(eta_n, tau(len(AN), rd), rd)
    slope_terms = []
    for cofactor, derivative_entry in terms:
        cm, ce = positive_product((abs(cofactor.mantissa), abs(derivative_entry)), rd, 'transfer_slope_rank')
        slope_terms.append((cm, ce+cofactor.exponent))
    slope_scale = positive_sum(slope_terms, rd)
    derivative_record = determinant_facts['derivative']
    scaled_slope = Scaled(derivative_record['mantissa'],
                          derivative_record['exponent']+sum(determinant_facts['row_shifts']))
    # Compare mantissa/exponent directly, preserving a derivative whose
    # magnitude would under/overflow outside the scaled numerator equation.
    slope_ratio = scaled_ratio(rd(abs(scaled_slope.mantissa)), slope_scale, rd)
    slope_ratio = (slope_ratio[0], slope_ratio[1]+scaled_slope.exponent)
    with measurements.phase('quantity_svd', quantity='transfer_zero', dimension=len(AN)):
        try:
            sv = np.linalg.svd(AN, compute_uv=False)
        except np.linalg.LinAlgError as error:
            raise QuantityError('root_slope_unresolved', 'transfer_slope_rank', 'transfer numerator SVD did not converge') from error
    rank_min = None if len(sv) == 1 or sv[0] == 0 else rd(sv[-1]/sv[0])
    rank_gap = None if len(sv) == 1 or sv[0] == 0 else rd(sv[-2]/sv[0])
    slope_pass = slope_scale[0] > 0 and not ratio_le(slope_ratio, tau(len(AN), rd), rd)
    rank_pass = len(sv) == 1 or (rank_min is not None and rank_min <= tau(len(sv), rd) and rank_gap > tau(len(sv), rd))
    # Closure decisions retain the native ordering: numerator/cofactor/SVD
    # facts precede the complete required denominator solve certificate.
    if denominator is not None:
        denominator.verify_certificate()
    correction = scaled_ratio(rd(abs(value)), positive_product((abs(slope), rd(2*np.pi)), rd), rd)
    correction_pass = ratio_le(correction, rd(0.01), rd)
    facts = dict(numerator=determinant_facts, denominator=denominator_facts,
                 numerator_residual=eta_n, normalizer=normalizer, slope_ratio=slope_ratio,
                 slope_scale=slope_scale, rank_min=rank_min, rank_gap=rank_gap, correction_hz=correction)
    if job.family == 'Z' and not system.view.port_realizable:
        facts['arithmetic'] = dict(system.compensation_evidence)
    if final:
        if not (np.isfinite(omega) and omega.real > 0):
            fail('numerical_resolution_unresolved', 'transfer_frequency', 'transfer-zero frequency is unresolved', **facts)
        if not residual_pass:
            fail('numerical_resolution_unresolved', 'transfer_numerator_scale', 'transfer numerator residual did not close', **facts)
        if not (slope_pass and rank_pass):
            fail('root_slope_unresolved', 'transfer_slope_rank', 'transfer numerator slope/rank did not close', **facts)
        if not correction_pass:
            fail('numerical_resolution_unresolved', 'transfer_correction', 'transfer-zero correction exceeds the existing 0.01 Hz condition', **facts)
    return residual_pass and slope_pass and rank_pass and correction_pass, value, slope, numerator_slope, detd, facts


def transfer_zero(system, start, job, assemble, measurements):
    cd, rd = system.complex_dtype, system.real_dtype
    omega = cd(start)
    if not (np.isfinite(omega) and omega.real > 0):
        fail('numerical_resolution_unresolved', 'transfer_frequency', 'transfer-zero initialization is unresolved')
    with measurements.phase('root_control', quantity='transfer_zero'):
        for update in range(33):
            passed, value, slope, numerator_slope, denominator, facts = transfer_at(
                system, omega, job, assemble, measurements, final=update == 32)
            if passed and omega.real > 0:
                return dict(root_omega_rad_s=omega, root_slope=slope,
                            numerator_slope=numerator_slope, denominator=denominator), dict(**facts, newton_steps=update)
            if update == 32:
                raise RuntimeError('final transfer certificate returned without a failure')
            if slope == 0:
                fail('root_slope_unresolved', 'transfer_slope_rank', 'transfer-zero derivative is zero')
            candidate = cd(omega-cd(value/slope))
            if not np.isfinite(candidate):
                fail('numerical_resolution_unresolved', 'transfer_frequency', 'transfer-zero update is nonfinite')
            if same_bits(candidate, omega, rd):
                transfer_at(system, candidate, job, assemble, measurements, final=True)
                raise RuntimeError('stagnant transfer certificate returned without a failure')
            omega = candidate


def residue_coupling(system, job, assemble, measurements):
    cd, rd, q = system.complex_dtype, system.real_dtype, len(system.selected)
    roots, vectors, slopes, residues, scales = [], [], [], [], []
    for branch in job.branches:
        body = branch.result
        if body.failure is not None:
            raise ValueError('residue dependency must be a finished successful branch result')
        omega = cd(body.root_omega_rad_s)
        if branch.kind == 'diagonal_root':
            if omega.imag > 0:
                fail('numerical_resolution_unresolved', 'newton_certificate', 'residue diagonal root violates the passive imaginary-root policy')
            vector = np.zeros(q, dtype=cd)
            vector[branch.coordinate_index] = cd(1)
        else:
            vector = np.array(body.null_vector, dtype=cd, copy=True)
        vector /= rd(np.linalg.norm(vector))
        _, Fp, *_ = checked_state(system, omega, assemble, measurements)
        slope = cd(-vector.T @ Fp @ vector)
        scale = rd(np.sum(np.abs(vector)*(np.abs(Fp) @ np.abs(vector)), dtype=rd))
        if not (np.isfinite(slope) and np.isfinite(scale) and scale > 0 and rd(abs(slope))/scale > tau(q, rd)):
            fail('root_slope_unresolved', 'residue_slope', 'residue branch slope is unresolved', slope_scale=scale)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            residue = cd(-cd(1)/slope)
        if not np.isfinite(residue):
            fail('root_slope_unresolved', 'residue_slope', 'residue branch value is nonfinite',
                 slope=slope, slope_scale=scale, residue=residue)
        roots.append(omega); vectors.append(vector); slopes.append(slope); residues.append(residue); scales.append(scale)
    try:
        singular = np.linalg.svd(np.column_stack(vectors), compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise QuantityError('root_slope_unresolved', 'residue_rank', 'residue branch SVD did not converge') from error
    if not (len(singular) >= 2 and singular[1]/singular[0] > tau(q, rd)):
        fail('root_slope_unresolved', 'residue_rank', 'residue branch vectors are not independent', singular_values=singular)
    evaluation = cd(cd(roots[0]+roots[1])/cd(2)) if job.frequency_mode == 'complex_root_midpoint' else cd(job.evaluation_omega_rad_s)
    if not np.isfinite(evaluation):
        fail('root_slope_unresolved', 'residue_coupling', 'coupling evaluation location is nonfinite')
    reciprocity = []
    locations = {}
    for location in (*roots, evaluation):
        locations.setdefault(np.asarray(location, dtype=cd).tobytes(), location)
    for omega in locations.values():
        state = checked_state(system, omega, assemble, measurements)
        closed, certificate = schur_reciprocity(system, state)
        reciprocity.append(dict(omega=omega, **certificate))
        if not closed:
            fail('root_slope_unresolved', 'reciprocity', 'selected-operator numerical reciprocity check did not close', reciprocity=reciprocity)
    F, *_ = checked_state(system, evaluation, assemble, measurements)
    with np.errstate(over='ignore', invalid='ignore'):
        denominator = cd(np.sqrt(cd(slopes[0]*slopes[1])))
    if not (np.isfinite(denominator) and denominator != 0):
        fail('root_slope_unresolved', 'residue_coupling',
             'principal-square-root coupling denominator is nonfinite or zero', denominator=denominator)
    coupling = cd(cd(vectors[0].T @ F @ vectors[1])/denominator)
    if not np.isfinite(coupling):
        fail('root_slope_unresolved', 'residue_coupling', 'residue-normalized coupling is nonfinite')
    return dict(coupling_rad_s=coupling, residue_a=residues[0], residue_b=residues[1],
                branch_roots_rad_s=np.asarray(roots, dtype=cd), evaluation_omega_rad_s=evaluation), dict(
                reciprocity=reciprocity, branch_slope_scales=scales, branch_independence=singular)


def evaluate_quantity(system, job, assemble, measurements):
    cd = system.complex_dtype
    start = cd(job.omega_start_rad_s) if job.omega_start_rad_s is not None else (
        cd(2*np.pi*job.root_hint_hz) if job.root_hint_hz is not None else None)
    if job.kind == 'operator_element_root':
        omega, slope, certificate, code, steps = element_root(system, start, job.row_index, job.column_index, assemble, measurements)
        if code:
            numerical_code(code)
        return dict(root_omega_rad_s=omega, root_slope=slope), dict(certificate=certificate, newton_steps=steps)
    if job.kind == 'hybridized_pole':
        return hybridized(system, start, assemble, measurements)
    if job.kind == 'transfer_zero':
        return transfer_zero(system, start, job, assemble, measurements)
    if job.kind == 'residue_normalized_coupling':
        return residue_coupling(system, job, assemble, measurements)
    raise ValueError(f'unsupported Direct quantity: {job.kind}')
