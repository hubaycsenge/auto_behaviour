"""The cleaning pipeline every engine's output has to survive."""

import unittest

from abcoder.common.ethogram import EPISODE_CATEGORY, POINT, STATE, Behavior, Ethogram, Subject
from abcoder.common.events import (
    Event,
    ObservationResult,
    clean,
    close_open_states,
    coerce_subjects,
    deduplicate,
    drop_unknown_behaviors,
    enforce_exclusivity,
    fuse,
    merge_adjacent_states,
)


def ethogram(subjects=("Dog",)):
    etho = Ethogram(
        behaviors=[
            Behavior("Bark", POINT),
            Behavior("Growl", POINT),
            Behavior("Sit", STATE),
            Behavior("Stand", STATE, excluded="Sit"),
            Behavior("Phase1", STATE, category=EPISODE_CATEGORY),
        ],
        subjects=[Subject(s) for s in subjects],
        categories=[EPISODE_CATEGORY],
    )
    return etho


class TestCleaning(unittest.TestCase):
    def test_invented_behaviours_are_dropped(self):
        kept, warnings = drop_unknown_behaviors(
            [Event("Bark", 1.0), Event("Teleport", 2.0)], ethogram())
        self.assertEqual([e.behavior for e in kept], ["Bark"])
        self.assertIn("Teleport", warnings[0])

    def test_single_subject_absorbs_unattributed_events(self):
        out, _ = coerce_subjects([Event("Bark", 1.0)], ethogram())
        self.assertEqual(out[0].subject, "Dog")

    def test_trial_phases_keep_no_focal_subject(self):
        out, _ = coerce_subjects([Event("Phase1", 1.0, subject="Dog")], ethogram())
        self.assertEqual(out[0].subject, "")

    def test_unknown_subject_is_reassigned_and_flagged(self):
        out, warnings = coerce_subjects(
            [Event("Bark", 1.0, subject="Ghost")], ethogram(("Dog", "Owner")))
        self.assertEqual(out[0].subject, "Dog")
        self.assertIn("reassigned", out[0].comment)
        self.assertTrue(warnings)

    def test_no_subjects_means_no_focal_subject(self):
        out, _ = coerce_subjects([Event("Bark", 1.0, subject="Dog")], ethogram(()))
        self.assertEqual(out[0].subject, "")

    def test_open_state_closes_at_the_next_one(self):
        out, warnings = close_open_states(
            [Event("Sit", 10.0), Event("Sit", 25.0)], ethogram(), duration=100.0)
        self.assertEqual([(e.start, e.stop) for e in out], [(10.0, 25.0), (25.0, 100.0)])
        self.assertEqual(len(warnings), 2)

    def test_state_ending_before_it_starts_is_repaired(self):
        out, warnings = close_open_states(
            [Event("Sit", 10.0, stop=5.0)], ethogram(), duration=100.0)
        self.assertGreater(out[0].stop, out[0].start)
        self.assertTrue(warnings)

    def test_duplicates_within_tolerance_collapse(self):
        out = deduplicate([Event("Bark", 10.0, confidence=0.4),
                           Event("Bark", 10.1, confidence=0.9)], tolerance=0.25)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].confidence, 0.9)

    def test_duplicate_states_take_the_union(self):
        out = deduplicate([Event("Sit", 10.0, stop=15.0), Event("Sit", 10.1, stop=22.0)])
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].start, out[0].stop), (10.0, 22.0))

    def test_adjacent_states_merge_across_a_gap(self):
        out = merge_adjacent_states(
            [Event("Sit", 0.0, stop=5.0), Event("Sit", 5.4, stop=9.0)], gap=0.5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].stop, 9.0)

    def test_excluded_states_do_not_overlap(self):
        out, warnings = enforce_exclusivity(
            [Event("Sit", 0.0, stop=20.0, subject="Dog"),
             Event("Stand", 10.0, stop=30.0, subject="Dog")], ethogram())
        sit = next(e for e in out if e.behavior == "Sit")
        self.assertEqual(sit.stop, 10.0)
        self.assertTrue(warnings)

    def test_exclusivity_truncates_the_right_occurrence(self):
        # Two identical Sit events: Event compares by value, so a value-based
        # lookup would truncate whichever came first regardless of overlap.
        out, _ = enforce_exclusivity(
            [Event("Sit", 0.0, stop=5.0, subject="Dog"),
             Event("Sit", 100.0, stop=200.0, subject="Dog"),
             Event("Stand", 150.0, stop=160.0, subject="Dog")], ethogram())
        sits = sorted((e for e in out if e.behavior == "Sit"), key=lambda e: e.start)
        self.assertEqual(sits[0].stop, 5.0, "the non-overlapping Sit must be untouched")
        self.assertEqual(sits[1].stop, 150.0, "the overlapping Sit must be truncated")

    def test_exclusivity_is_per_subject(self):
        out, _ = enforce_exclusivity(
            [Event("Sit", 0.0, stop=20.0, subject="Dog"),
             Event("Stand", 10.0, stop=30.0, subject="Owner")],
            ethogram(("Dog", "Owner")))
        sit = next(e for e in out if e.behavior == "Sit")
        self.assertEqual(sit.stop, 20.0)

    def test_events_past_the_end_are_dropped_and_states_trimmed(self):
        out, _ = clean([Event("Bark", 500.0), Event("Sit", 90.0, stop=140.0)],
                       ethogram(), duration=100.0)
        self.assertEqual([e.behavior for e in out], ["Sit"])
        self.assertEqual(out[0].stop, 100.0)

    def test_clean_is_ordered_by_time(self):
        out, _ = clean([Event("Bark", 30.0), Event("Growl", 5.0)],
                       ethogram(), duration=100.0)
        self.assertEqual([e.start for e in out], [5.0, 30.0])

    def test_nan_and_infinity_are_discarded(self):
        out, _ = clean([Event("Bark", float("nan")), Event("Bark", float("inf")),
                        Event("Bark", 3.0)], ethogram(), duration=100.0)
        self.assertEqual(len(out), 1)


class TestFusion(unittest.TestCase):
    def _result(self, name, events):
        return ObservationResult(observation_id="x", events=events, engine={"name": name})

    def test_ownership_filters_out_the_non_owner(self):
        etho = ethogram()
        results = [
            self._result("audio", [Event("Bark", 10.0, confidence=0.9)]),
            self._result("vlm_vllm", [Event("Bark", 10.0, confidence=0.3),
                                      Event("Sit", 1.0, stop=5.0)]),
        ]
        events, _ = fuse(results, etho, {"Bark": "audio"}, duration=60.0)
        barks = [e for e in events if e.behavior == "Bark"]
        self.assertEqual(len(barks), 1)
        self.assertEqual(barks[0].source, "audio")
        self.assertTrue(any(e.behavior == "Sit" for e in events))

    def test_unowned_behaviours_pool_and_deduplicate(self):
        etho = ethogram()
        results = [self._result("a", [Event("Growl", 10.0, confidence=0.5)]),
                   self._result("b", [Event("Growl", 10.1, confidence=0.8)])]
        events, _ = fuse(results, etho, {}, duration=60.0)
        self.assertEqual(len(events), 1)

    def test_ownership_naming_an_unknown_behaviour_warns(self):
        _, warnings = fuse([self._result("a", [])], ethogram(), {"Nope": "a"}, 10.0)
        self.assertTrue(any("Nope" in w for w in warnings))

    def test_source_is_recorded(self):
        events, _ = fuse([self._result("audio", [Event("Bark", 1.0)])],
                         ethogram(), {}, 10.0)
        self.assertEqual(events[0].source, "audio")


if __name__ == "__main__":
    unittest.main()
