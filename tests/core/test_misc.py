"""Tests for core/misc.py utility functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.misc import (
    PaddingConfig,
    advanced_padding,
    assimilate_shape,
    batched_diag_construct,
    cast_floating_to_numpy,
    ensure_slice_tuple,
    expand_matrix,
    find_squarest_divisors,
    get_air_name,
    get_background_material_name,
    index_1d_array,
    index_by_slice,
    index_by_slice_take,
    index_by_slice_take_1d,
    invert_dict,
    is_float_divisible,
    is_index_in_slice,
    linear_interpolated_indexing,
    mask_1d_from_slice,
    normalize_polarization_for_source,
    prime_factorization,
)
from fdtdx.materials import Material


class TestExpandMatrix:
    """Tests for expand_matrix function."""

    def test_expand_2d_matrix(self):
        """Test expanding a 2D matrix."""
        matrix = jnp.ones((2, 3))
        result = expand_matrix(matrix, (2, 3, 1))

        # Should expand to (4, 9, 1)
        assert result.shape == (4, 9, 1)
        assert jnp.all(result == 1.0)

    def test_expand_3d_matrix(self):
        """Test expanding a 3D matrix."""
        matrix = jnp.ones((2, 3, 4))
        result = expand_matrix(matrix, (2, 2, 3))

        # Should expand to (4, 6, 12)
        assert result.shape == (4, 6, 12)
        assert jnp.all(result == 1.0)

    def test_expand_preserves_values(self):
        """Test that expansion preserves values correctly."""
        matrix = jnp.array([[1, 2], [3, 4]])
        result = expand_matrix(matrix, (2, 2, 1))

        # Check that values are repeated correctly
        # Result shape will be (4, 4, 1)
        assert result[0, 0, 0] == 1
        assert result[0, 1, 0] == 1  # Repeated horizontally
        assert result[1, 0, 0] == 1  # Repeated vertically
        # Value 4 at (1,1) expands to positions (2:4, 2:4, 0)
        assert result[2, 2, 0] == 4


class TestEnsureSliceTuple:
    """Tests for ensure_slice_tuple function."""

    def test_convert_ints_to_slices(self):
        """Test converting integers to slices."""
        result = ensure_slice_tuple([1, 2, 3])

        assert len(result) == 3
        assert result[0] == slice(1, 2)
        assert result[1] == slice(2, 3)
        assert result[2] == slice(3, 4)

    def test_pass_through_slices(self):
        """Test that slices are passed through unchanged."""
        input_slices = [slice(0, 5), slice(10, 20), slice(None)]
        result = ensure_slice_tuple(input_slices)

        assert result == tuple(input_slices)

    def test_convert_tuples_to_slices(self):
        """Test converting tuples to slices."""
        result = ensure_slice_tuple([(0, 5), (10, 20)])

        assert len(result) == 2
        assert result[0] == slice(0, 5)
        assert result[1] == slice(10, 20)

    def test_mixed_types(self):
        """Test with mixed input types."""
        result = ensure_slice_tuple([1, slice(5, 10), (20, 30)])

        assert len(result) == 3
        assert result[0] == slice(1, 2)
        assert result[1] == slice(5, 10)
        assert result[2] == slice(20, 30)

    def test_invalid_type_raises_error(self):
        """Test that invalid types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid location type"):
            ensure_slice_tuple([1.5])  # Float is invalid


class TestIsFloatDivisible:
    """Tests for is_float_divisible function."""

    def test_exact_division(self):
        """Test exact division."""
        assert is_float_divisible(10.0, 2.0)
        assert is_float_divisible(15.0, 3.0)

    def test_not_divisible(self):
        """Test non-divisible floats."""
        assert not is_float_divisible(10.0, 3.0)
        assert not is_float_divisible(7.5, 2.0)

    def test_with_tolerance(self):
        """Test division within tolerance."""
        # Nearly divisible due to float precision
        assert is_float_divisible(10.0000000001, 2.0, tolerance=1e-8)

    def test_zero_divisor(self):
        """Test that zero divisor returns False."""
        assert not is_float_divisible(10.0, 0.0)

    def test_zero_dividend(self):
        """Test zero dividend."""
        assert is_float_divisible(0.0, 5.0)


class TestIsIndexInSlice:
    """Tests for is_index_in_slice function."""

    def test_index_in_slice(self):
        """Test index within slice."""
        assert is_index_in_slice(5, slice(0, 10), 20)
        assert is_index_in_slice(0, slice(0, 10), 20)
        assert is_index_in_slice(9, slice(0, 10), 20)

    def test_index_not_in_slice(self):
        """Test index outside slice."""
        assert not is_index_in_slice(10, slice(0, 10), 20)
        assert not is_index_in_slice(15, slice(0, 10), 20)
        assert not is_index_in_slice(-1, slice(0, 10), 20)

    def test_negative_slice_indices(self):
        """Test with negative slice indices."""
        assert is_index_in_slice(15, slice(-5, None), 20)
        assert not is_index_in_slice(10, slice(-5, None), 20)


class TestCastFloatingToNumpy:
    """Tests for cast_floating_to_numpy function."""

    def test_cast_float_to_float32(self):
        """Test casting float arrays."""
        vals = {"a": np.array([1.0, 2.0], dtype=np.float64)}
        result = cast_floating_to_numpy(vals, np.float32)

        assert result["a"].dtype == np.float32

    def test_cast_complex_to_float(self):
        """Test casting complex to float takes real part."""
        vals = {"a": np.array([1 + 2j, 3 + 4j], dtype=np.complex64)}
        result = cast_floating_to_numpy(vals, np.float32)

        assert result["a"].dtype == np.float32
        assert np.allclose(result["a"], [1.0, 3.0])

    def test_preserve_complex_when_target_is_complex(self):
        """Test that complex arrays are preserved when target is complex."""
        vals = {"a": np.array([1 + 2j, 3 + 4j], dtype=np.complex64)}
        result = cast_floating_to_numpy(vals, np.complex128)

        assert result["a"].dtype == np.complex128
        assert np.allclose(result["a"], [1 + 2j, 3 + 4j])


class TestBatchedDiagConstruct:
    """Tests for batched_diag_construct function."""

    def test_single_diagonal(self):
        """Test creating single diagonal matrix."""
        arr = jnp.array([1, 2, 3])
        result = batched_diag_construct(arr)

        expected = jnp.diag(arr)
        assert jnp.allclose(result, expected)

    def test_batched_diagonals(self):
        """Test creating multiple diagonal matrices."""
        arr = jnp.array([[1, 2], [3, 4], [5, 6]])
        result = batched_diag_construct(arr)

        assert result.shape == (3, 2, 2)
        # Check that each diagonal matrix is correct
        assert result[0, 0, 0] == 1 and result[0, 1, 1] == 2
        assert result[1, 0, 0] == 3 and result[1, 1, 1] == 4
        assert result[2, 0, 0] == 5 and result[2, 1, 1] == 6

    def test_higher_dimensional_batch(self):
        """Test with higher dimensional batch."""
        arr = jnp.ones((2, 3, 4))
        result = batched_diag_construct(arr)

        assert result.shape == (2, 3, 4, 4)
        # Check that diagonals are set correctly
        for i in range(2):
            for j in range(3):
                assert jnp.allclose(jnp.diag(result[i, j]), jnp.ones(4))


class TestInvertDict:
    """Tests for invert_dict function."""

    def test_simple_dict_inversion(self):
        """Test simple dictionary inversion."""
        d = {"a": 1, "b": 2, "c": 3}
        result = invert_dict(d)

        assert result == {1: "a", 2: "b", 3: "c"}

    def test_empty_dict(self):
        """Test inverting empty dictionary."""
        result = invert_dict({})

        assert result == {}

    def test_single_element(self):
        """Test single element dictionary."""
        result = invert_dict({"key": "value"})

        assert result == {"value": "key"}


class TestPrimeFactorization:
    """Tests for prime_factorization function."""

    def test_small_primes(self):
        """Test factorization of small primes."""
        assert prime_factorization(2) == [2]
        assert prime_factorization(3) == [3]
        assert prime_factorization(5) == [5]

    def test_composite_numbers(self):
        """Test factorization of composite numbers."""
        assert prime_factorization(12) == [2, 2, 3]
        assert prime_factorization(30) == [2, 3, 5]
        assert prime_factorization(100) == [2, 2, 5, 5]

    def test_large_prime(self):
        """Test factorization of larger prime."""
        assert prime_factorization(97) == [97]

    def test_perfect_square(self):
        """Test factorization of perfect square."""
        assert prime_factorization(36) == [2, 2, 3, 3]


class TestFindSquarestDivisors:
    """Tests for find_squarest_divisors function."""

    def test_perfect_square(self):
        """Test with perfect square."""
        a, b = find_squarest_divisors(36)

        assert a * b == 36
        assert abs(a - b) <= 2  # Should be close (6, 6)

    def test_prime_number(self):
        """Test with prime number."""
        a, b = find_squarest_divisors(13)

        assert a * b == 13
        assert {a, b} == {1, 13}

    def test_non_square(self):
        """Test with non-square composite."""
        a, b = find_squarest_divisors(24)

        assert a * b == 24
        # Should find relatively square factors

    def test_power_of_two(self):
        """Test with power of 2."""
        a, b = find_squarest_divisors(64)

        assert a * b == 64
        assert a == 8 and b == 8  # Should be square


class TestIndex1dArray:
    """Tests for index_1d_array function."""

    def test_find_value_at_beginning(self):
        """Test finding value at beginning of array."""
        arr = jnp.array([5, 2, 3, 4])
        idx = index_1d_array(arr, 5)

        assert idx == 0

    def test_find_value_in_middle(self):
        """Test finding value in middle of array."""
        arr = jnp.array([1, 2, 5, 4])
        idx = index_1d_array(arr, 5)

        assert idx == 2

    def test_value_not_found(self):
        """Test when value is not in array."""
        arr = jnp.array([1, 2, 3, 4])
        idx = index_1d_array(arr, 10)

        # Returns 0 when not found (argmax of all False)
        assert idx == 0

    def test_raises_on_multidimensional(self):
        """Test that multidimensional array raises exception."""
        arr = jnp.array([[1, 2], [3, 4]])

        with pytest.raises(Exception, match="index only works on 1d-array"):
            index_1d_array(arr, 1)


class TestIndexBySlice:
    """Tests for index_by_slice function."""

    def test_slice_along_axis_0(self):
        """Test slicing along axis 0."""
        arr = jnp.arange(24).reshape((2, 3, 4))
        result = index_by_slice(arr, 0, 1, axis=0)

        assert result.shape == (1, 3, 4)
        assert jnp.allclose(result, arr[0:1, :, :])

    def test_slice_along_axis_1(self):
        """Test slicing along axis 1."""
        arr = jnp.arange(24).reshape((2, 3, 4))
        result = index_by_slice(arr, 1, 3, axis=1)

        assert result.shape == (2, 2, 4)

    def test_slice_with_step(self):
        """Test slicing with step."""
        arr = jnp.arange(10)
        result = index_by_slice(arr, 0, 10, axis=0, step=2)

        assert jnp.allclose(result, jnp.array([0, 2, 4, 6, 8]))


class TestIndexBySliceTake1d:
    """Tests for index_by_slice_take_1d function."""

    def test_simple_slice(self):
        """Test simple slice operation."""
        arr = jnp.arange(10)
        result = index_by_slice_take_1d(arr, slice(2, 5), axis=0)

        assert jnp.allclose(result, jnp.array([2, 3, 4]))

    def test_slice_entire_array(self):
        """Test that slicing entire array returns original."""
        arr = jnp.arange(10)
        result = index_by_slice_take_1d(arr, slice(0, 10, 1), axis=0)

        assert result is arr  # Should return same array

    def test_slice_with_step(self):
        """Test slice with step."""
        arr = jnp.arange(20).reshape((4, 5))
        result = index_by_slice_take_1d(arr, slice(0, 4, 2), axis=0)

        assert result.shape == (2, 5)

    def test_empty_slice_raises(self):
        """Test that empty slice raises exception."""
        arr = jnp.arange(10)

        with pytest.raises(Exception, match="Invalid slice"):
            index_by_slice_take_1d(arr, slice(5, 2), axis=0)


class TestIndexBySliceTake:
    """Tests for index_by_slice_take function."""

    def test_slice_multiple_axes(self):
        """Test slicing along multiple axes."""
        arr = jnp.arange(24).reshape((2, 3, 4))
        slices = [slice(0, 1), slice(1, 3), slice(0, 4)]
        result = index_by_slice_take(arr, slices)

        assert result.shape == (1, 2, 4)

    def test_full_slices_return_original(self):
        """Test that full slices return array unchanged."""
        arr = jnp.arange(24).reshape((2, 3, 4))
        slices = [slice(None), slice(None), slice(None)]
        result = index_by_slice_take(arr, slices)

        assert jnp.allclose(result, arr)


class TestMask1dFromSlice:
    """Tests for mask_1d_from_slice function."""

    def test_simple_slice_mask(self):
        """Test creating mask from simple slice."""
        mask = mask_1d_from_slice(slice(2, 5), axis_size=10)

        expected = jnp.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=jnp.bool_)
        assert jnp.allclose(mask, expected)

    def test_slice_from_start(self):
        """Test mask from start of array."""
        mask = mask_1d_from_slice(slice(0, 3), axis_size=5)

        expected = jnp.array([1, 1, 1, 0, 0], dtype=jnp.bool_)
        assert jnp.allclose(mask, expected)

    def test_slice_with_step(self):
        """Test mask with step."""
        mask = mask_1d_from_slice(slice(0, 10, 2), axis_size=10)

        expected = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=jnp.bool_)
        assert jnp.allclose(mask, expected)


class TestAssimilateShape:
    """Tests for assimilate_shape function."""

    def test_add_dimensions(self):
        """Test adding dimensions for broadcasting."""
        arr = jnp.ones((3, 4))
        ref_arr = jnp.ones((2, 3, 4, 5))
        result = assimilate_shape(arr, ref_arr, ref_axes=(1, 2))

        assert result.shape == (1, 3, 4, 1)

    def test_with_repeat(self):
        """Test with repeat_single_dims=True."""
        arr = jnp.ones((1, 4))
        ref_arr = jnp.ones((2, 3, 4, 5))
        result = assimilate_shape(arr, ref_arr, ref_axes=(1, 2), repeat_single_dims=True)

        assert result.shape == (1, 3, 4, 1)

    def test_invalid_axes_raises(self):
        """Test that invalid axes raise exception."""
        arr = jnp.ones((3, 4))
        ref_arr = jnp.ones((2, 3, 4, 5))

        with pytest.raises(Exception, match="Invalid axes"):
            assimilate_shape(arr, ref_arr, ref_axes=(0, 1, 2))  # Too many axes


class TestLinearInterpolatedIndexing:
    """Tests for linear_interpolated_indexing function."""

    def test_interpolate_at_grid_point(self):
        """Test interpolation exactly at grid point."""
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        point = jnp.array([0.0, 0.0])
        result = linear_interpolated_indexing(point, arr)

        assert jnp.isclose(result, 1.0)

    def test_interpolate_between_points(self):
        """Test interpolation between grid points."""
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        point = jnp.array([0.5, 0.5])
        result = linear_interpolated_indexing(point, arr)

        # Average of all four corners
        expected = (1.0 + 2.0 + 3.0 + 4.0) / 4.0
        assert jnp.isclose(result, expected)

    def test_out_of_bounds_point(self):
        """Test interpolation with out-of-bounds point."""
        arr = jnp.ones((3, 3))
        point = jnp.array([5.0, 5.0])  # Outside array
        result = linear_interpolated_indexing(point, arr)

        # Should handle gracefully (weights will be zero)
        assert jnp.isfinite(result)

    def test_invalid_dimensions_raises(self):
        """Test that mismatched dimensions raise exception."""
        arr = jnp.ones((3, 3))
        point = jnp.array([1.0, 2.0, 3.0])  # 3D point for 2D array

        with pytest.raises(Exception, match="Invalid shape"):
            linear_interpolated_indexing(point, arr)


class TestGetAirName:
    """Tests for get_air_name function."""

    def test_find_air_material(self):
        """Test finding air material."""
        materials = {
            "air": Material(permittivity=1.0, permeability=1.0),
            "glass": Material(permittivity=2.25, permeability=1.0),
        }

        result = get_air_name(materials)
        assert result == "air"

    def test_no_air_returns_first(self):
        """Test that first material is returned when no air found."""
        materials = {
            "glass": Material(permittivity=2.25, permeability=1.0),
            "silicon": Material(permittivity=12.0, permeability=1.0),
        }

        result = get_air_name(materials)
        assert result in ["glass", "silicon"]  # Should return first


class TestGetBackgroundMaterialName:
    """Tests for get_background_material_name function."""

    def test_find_lowest_permittivity(self):
        """Test finding material with lowest permittivity."""
        materials = {
            "air": Material(permittivity=1.0),
            "glass": Material(permittivity=2.25),
            "silicon": Material(permittivity=12.0),
        }

        result = get_background_material_name(materials)
        assert result == "air"

    def test_with_anisotropic_materials(self):
        """Test with anisotropic materials."""
        materials = {
            "aniso": Material(permittivity=(2.0, 3.0, 4.0)),
            "air": Material(permittivity=1.0),
        }

        result = get_background_material_name(materials)
        assert result == "air"

    def test_empty_dict_raises(self):
        """Test that empty dictionary raises exception."""
        with pytest.raises(Exception, match="Empty Material dictionary"):
            get_background_material_name({})


class TestAdvancedPadding:
    """Tests for advanced_padding function."""

    def test_constant_padding(self):
        """Test constant value padding."""
        arr = jnp.ones((3, 3))
        cfg = PaddingConfig(widths=[1], modes=["constant"], values=[0])

        result, slices = advanced_padding(arr, cfg)

        assert result.shape == (5, 5)
        # Check padding values
        assert result[0, 0] == 0  # Padded value
        assert result[1, 1] == 1  # Original value

    def test_edge_padding(self):
        """Test edge mode padding."""
        arr = jnp.arange(9).reshape((3, 3))
        cfg = PaddingConfig(widths=[1], modes=["edge"])

        result, slices = advanced_padding(arr, cfg)

        assert result.shape == (5, 5)

    def test_asymmetric_padding(self):
        """Test asymmetric padding on different sides."""
        arr = jnp.ones((3, 3))
        cfg = PaddingConfig(widths=[2, 1, 1, 2], modes=["constant"] * 4, values=[0] * 4)

        result, slices = advanced_padding(arr, cfg)

        # Padded 2 on left, 1 on right for axis 0
        # Padded 1 on left, 2 on right for axis 1
        assert result.shape == (6, 6)


class TestNormalizePolarizationForSource:
    """Tests for normalize_polarization_for_source function."""

    def test_normalize_e_polarization(self):
        """Test normalization with E polarization specified."""
        e_pol, h_pol = normalize_polarization_for_source(
            direction="+", propagation_axis=0, fixed_E_polarization_vector=(0.0, 1.0, 0.0)
        )

        # E should be normalized
        assert jnp.isclose(jnp.linalg.norm(e_pol), 1.0)
        # H should be orthogonal to E and propagation
        assert jnp.isclose(jnp.linalg.norm(h_pol), 1.0)

    def test_normalize_h_polarization(self):
        """Test normalization with H polarization specified."""
        e_pol, h_pol = normalize_polarization_for_source(
            direction="+", propagation_axis=0, fixed_H_polarization_vector=(0.0, 0.0, 1.0)
        )

        # Both should be normalized
        assert jnp.isclose(jnp.linalg.norm(e_pol), 1.0)
        assert jnp.isclose(jnp.linalg.norm(h_pol), 1.0)

    def test_both_none_raises(self):
        """Test that having neither E nor H polarization raises error."""
        with pytest.raises(Exception, match="Need to specify either E or H polarization"):
            normalize_polarization_for_source(direction="+", propagation_axis=0)

    def test_different_propagation_axes(self):
        """Test with different propagation axes."""
        # Test axis 0 (propagate along x, E along y)
        e_pol, h_pol = normalize_polarization_for_source(
            direction="+", propagation_axis=0, fixed_E_polarization_vector=(0.0, 1.0, 0.0)
        )
        assert jnp.isclose(jnp.linalg.norm(e_pol), 1.0)
        assert jnp.isclose(jnp.linalg.norm(h_pol), 1.0)

        # Test axis 1 (propagate along y, E along z)
        e_pol, h_pol = normalize_polarization_for_source(
            direction="+", propagation_axis=1, fixed_E_polarization_vector=(0.0, 0.0, 1.0)
        )
        assert jnp.isclose(jnp.linalg.norm(e_pol), 1.0)
        assert jnp.isclose(jnp.linalg.norm(h_pol), 1.0)

        # Test axis 2 (propagate along z, E along x)
        e_pol, h_pol = normalize_polarization_for_source(
            direction="+", propagation_axis=2, fixed_E_polarization_vector=(1.0, 0.0, 0.0)
        )
        assert jnp.isclose(jnp.linalg.norm(e_pol), 1.0)
        assert jnp.isclose(jnp.linalg.norm(h_pol), 1.0)