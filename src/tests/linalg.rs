use crate::linalg::invert_3x3;

#[test]
fn invert_3x3_correct() {
    let m = [[4.0, 3.0, 2.0], [1.0, 5.0, 3.0], [2.0, 1.0, 6.0]];
    let inv = invert_3x3(m).expect("matrix is invertible");

    let inv_columns = [
        [inv[0][0], inv[1][0], inv[2][0]],
        [inv[0][1], inv[1][1], inv[2][1]],
        [inv[0][2], inv[1][2], inv[2][2]],
    ];
    for (row, m_row) in m.iter().enumerate() {
        for (col, inv_col) in inv_columns.iter().enumerate() {
            let dot: f32 = m_row
                .iter()
                .zip(inv_col.iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum();
            let expected = if row == col { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-5,
                "A*A^-1[{row}][{col}] = {dot}"
            );
        }
    }
}

#[test]
fn invert_3x3_singular_returns_none() {
    let singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]];
    assert!(invert_3x3(singular).is_none());
}
