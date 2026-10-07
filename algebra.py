"""循环群代数乘法表与 CRT 投影。"""

from __future__ import annotations

import numpy as np


def idx_to_coeffs(index: int, degree: int, p: int) -> np.ndarray:
    """将元素编号转换为低次项在前的系数向量。"""
    coeffs = np.zeros(degree, dtype=np.int64)
    for position in range(degree):
        coeffs[position] = index % p
        index //= p
    return coeffs


def coeffs_to_idx(coeffs: np.ndarray, p: int) -> int:
    """将低次项在前的系数向量转换为元素编号。"""
    index = 0
    for value in reversed(coeffs):
        index = index * p + int(value)
    return index


def cyclic_group_algebra_table(p: int, group_order: int) -> np.ndarray:
    """构造 F_p[Z_n] 的完整乘法表。"""
    ring_size = p ** group_order
    elements = np.stack([
        idx_to_coeffs(index, group_order, p)
        for index in range(ring_size)
    ])
    table = np.zeros((ring_size, ring_size), dtype=np.int64)
    for left in range(ring_size):
        for right in range(ring_size):
            product = np.zeros(group_order, dtype=np.int64)
            for i in range(group_order):
                if elements[left, i] == 0:
                    continue
                for j in range(group_order):
                    product[(i + j) % group_order] += (
                        elements[left, i] * elements[right, j]
                    )
            table[left, right] = coeffs_to_idx(product % p, p)
    return table


def poly_mod(coeffs, factor, p: int) -> np.ndarray:
    """计算 F_p[x] 中多项式除以首一 factor 的余式。"""
    remainder = np.asarray(coeffs, dtype=np.int64).copy() % p
    factor = np.asarray(factor, dtype=np.int64) % p
    while True:
        nonzero = np.flatnonzero(remainder)
        if len(nonzero) == 0 or nonzero[-1] < len(factor) - 1:
            break
        shift = nonzero[-1] - len(factor) + 1
        scale = remainder[nonzero[-1]]
        remainder[shift:shift + len(factor)] = (
            remainder[shift:shift + len(factor)] - scale * factor
        ) % p
    return remainder[:len(factor) - 1]


def build_crt_projections(p: int, degree: int, factors) -> list[np.ndarray]:
    """返回每个环元素在各个互素因子环中的编号。"""
    ring_size = p ** degree
    projections = []
    for factor in factors:
        values = np.zeros(ring_size, dtype=np.int64)
        for index in range(ring_size):
            remainder = poly_mod(idx_to_coeffs(index, degree, p), factor, p)
            values[index] = coeffs_to_idx(remainder, p)
        projections.append(values)
    return projections


def audit_projections(table, projections) -> dict:
    """穷举验证联合单射和乘法同态。"""
    ring_size = len(table)
    joint = list(zip(*(values.tolist() for values in projections)))
    if len(set(joint)) != ring_size:
        raise RuntimeError("CRT 联合投影不是单射")

    for values in projections:
        induced = {}
        for left in range(ring_size):
            for right in range(ring_size):
                key = (int(values[left]), int(values[right]))
                output = int(values[table[left, right]])
                previous = induced.setdefault(key, output)
                if previous != output:
                    raise RuntimeError("CRT 投影不满足乘法同态")
    return {
        "ring_size": ring_size,
        "factor_sizes": [int(np.unique(values).size) for values in projections],
        "joint_unique": len(set(joint)),
    }

