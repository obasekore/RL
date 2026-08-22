import numpy as np

from coppelia_rl.retargeting.contact_lock import contiguous_true_runs, lock_contacts


def test_contiguous_true_runs_empty():
    assert contiguous_true_runs(np.array([], dtype=bool)) == []


def test_contiguous_true_runs_all_false():
    assert contiguous_true_runs(np.array([False, False, False])) == []


def test_contiguous_true_runs_all_true():
    assert contiguous_true_runs(np.array([True, True, True])) == [(0, 2)]


def test_contiguous_true_runs_multiple():
    flags = np.array([False, True, True, False, False, True, False, True, True, True])
    assert contiguous_true_runs(flags) == [(1, 2), (5, 5), (7, 9)]


def test_locked_curve_constant_across_contact_run():
    curve = np.array([[float(i), 0.0, 0.0] for i in range(6)])
    contact = np.array([False, True, True, True, False, False])

    locked = lock_contacts({"foot": curve}, {"foot": contact}, {"foot"})

    np.testing.assert_array_equal(locked["foot"][1], curve[1])
    np.testing.assert_array_equal(locked["foot"][2], curve[1])
    np.testing.assert_array_equal(locked["foot"][3], curve[1])


def test_locked_curve_unaffected_outside_contact_run():
    curve = np.array([[float(i), 0.0, 0.0] for i in range(6)])
    contact = np.array([False, True, True, True, False, False])

    locked = lock_contacts({"foot": curve}, {"foot": contact}, {"foot"})

    np.testing.assert_array_equal(locked["foot"][0], curve[0])
    np.testing.assert_array_equal(locked["foot"][4], curve[4])
    np.testing.assert_array_equal(locked["foot"][5], curve[5])


def test_non_contact_capable_bone_untouched_even_with_contact_channel():
    curve = np.array([[float(i), 0.0, 0.0] for i in range(4)])
    contact = np.array([True, True, True, True])

    locked = lock_contacts({"hand": curve}, {"hand": contact}, contact_capable_bones=set())

    np.testing.assert_array_equal(locked["hand"], curve)


def test_bone_without_contact_channel_untouched():
    curve = np.array([[float(i), 0.0, 0.0] for i in range(4)])

    locked = lock_contacts({"tail": curve}, {}, {"tail"})

    np.testing.assert_array_equal(locked["tail"], curve)


def test_multiple_disjoint_runs_locked_independently():
    curve = np.array([[float(i), 0.0, 0.0] for i in range(8)])
    contact = np.array([True, True, False, False, True, True, True, False])

    locked = lock_contacts({"foot": curve}, {"foot": contact}, {"foot"})

    np.testing.assert_array_equal(locked["foot"][0], curve[0])
    np.testing.assert_array_equal(locked["foot"][1], curve[0])
    np.testing.assert_array_equal(locked["foot"][2], curve[2])
    np.testing.assert_array_equal(locked["foot"][3], curve[3])
    np.testing.assert_array_equal(locked["foot"][4], curve[4])
    np.testing.assert_array_equal(locked["foot"][5], curve[4])
    np.testing.assert_array_equal(locked["foot"][6], curve[4])
    np.testing.assert_array_equal(locked["foot"][7], curve[7])


def test_lock_contacts_does_not_mutate_input():
    curve = np.array([[float(i), 0.0, 0.0] for i in range(4)])
    original = curve.copy()
    contact = np.array([True, True, True, True])

    lock_contacts({"foot": curve}, {"foot": contact}, {"foot"})

    np.testing.assert_array_equal(curve, original)
