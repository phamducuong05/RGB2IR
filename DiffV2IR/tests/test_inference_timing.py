import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_timing import InferenceTimer, format_duration


class FormatDurationTests(unittest.TestCase):
    def test_formats_seconds_minutes_and_hours(self):
        self.assertEqual(format_duration(42.3), "42.3s")
        self.assertEqual(format_duration(125), "2m 05s")
        self.assertEqual(format_duration(7384), "2h 03m 04s")


class InferenceTimerTests(unittest.TestCase):
    def test_records_only_completed_images_and_estimates_remaining_time(self):
        timer = InferenceTimer(total_images=5)

        first = timer.record(10.0)
        second = timer.record(20.0)

        self.assertEqual(first.completed, 1)
        self.assertEqual(first.average_seconds, 10.0)
        self.assertEqual(first.eta_seconds, 40.0)
        self.assertEqual(second.completed, 2)
        self.assertEqual(second.average_seconds, 15.0)
        self.assertEqual(second.eta_seconds, 45.0)

    def test_builds_a_readable_per_image_log(self):
        timer = InferenceTimer(total_images=5)
        stats = timer.record(42.3)

        message = stats.format_log(index=1, key="FLIR_00001_PreviewData")

        self.assertEqual(
            message,
            ">> [1/5] FLIR_00001_PreviewData: hoan thanh | "
            "anh nay: 42.3s | trung binh: 42.3s | ETA: 2m 49s",
        )


if __name__ == "__main__":
    unittest.main()
