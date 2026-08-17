"""Small, dependency-free helpers for reporting inference timing."""

from dataclasses import dataclass


def format_duration(seconds):
    """Format a duration compactly for a long-running terminal log."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"

    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


@dataclass(frozen=True)
class InferenceTimingStats:
    completed: int
    total_images: int
    image_seconds: float
    average_seconds: float
    eta_seconds: float

    def format_log(self, index, key):
        return (
            f">> [{index}/{self.total_images}] {key}: hoan thanh | "
            f"anh nay: {format_duration(self.image_seconds)} | "
            f"trung binh: {format_duration(self.average_seconds)} | "
            f"ETA: {format_duration(self.eta_seconds)}"
        )


class InferenceTimer:
    """Track successful inference durations without counting skips or failures."""

    def __init__(self, total_images):
        self.total_images = max(0, int(total_images))
        self.completed = 0
        self.total_seconds = 0.0

    def record(self, image_seconds):
        image_seconds = max(0.0, float(image_seconds))
        self.completed += 1
        self.total_seconds += image_seconds
        average = self.total_seconds / self.completed
        remaining = max(0, self.total_images - self.completed)
        return InferenceTimingStats(
            completed=self.completed,
            total_images=self.total_images,
            image_seconds=image_seconds,
            average_seconds=average,
            eta_seconds=average * remaining,
        )
