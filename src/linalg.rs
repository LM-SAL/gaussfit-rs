use num_traits::Float;

/// Solve a 3x3 linear system `matrix * x = rhs` via Gaussian elimination with
/// partial pivoting. Returns `None` if the matrix is (near-)singular.
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_3x3<F: Float>(mut matrix: [[F; 3]; 3], mut rhs: [F; 3]) -> Option<[F; 3]> {
    let epsilon = F::epsilon();
    for pivot_col in 0..3 {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = matrix[pivot_col][pivot_col].abs();
        for row in (pivot_col + 1)..3 {
            let candidate = matrix[row][pivot_col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if pivot_abs <= epsilon || !pivot_abs.is_finite() {
            return None;
        }

        if pivot_row != pivot_col {
            matrix.swap(pivot_col, pivot_row);
            rhs.swap(pivot_col, pivot_row);
        }

        let pivot = matrix[pivot_col][pivot_col];
        for col in pivot_col..3 {
            matrix[pivot_col][col] = matrix[pivot_col][col] / pivot;
        }
        rhs[pivot_col] = rhs[pivot_col] / pivot;

        for row in 0..3 {
            if row == pivot_col {
                continue;
            }
            let factor = matrix[row][pivot_col];
            for col in pivot_col..3 {
                matrix[row][col] = matrix[row][col] - factor * matrix[pivot_col][col];
            }
            rhs[row] = rhs[row] - factor * rhs[pivot_col];
        }
    }

    Some(rhs)
}

/// Invert a 3x3 matrix via the explicit cofactor formula.
/// Returns `None` if the determinant is zero or non-finite.
pub(crate) fn invert_3x3<F: Float>(m: [[F; 3]; 3]) -> Option<[[F; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() <= F::epsilon() || !det.is_finite() {
        return None;
    }

    let inv_det = F::one() / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}
