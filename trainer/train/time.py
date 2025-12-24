import time

class ETATimer:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = time.perf_counter()

    def update(self, completed_steps: int) -> float | None:
        """
        Returns ETA in seconds, or None if not enough info yet.
        """
        if completed_steps <= 0:
            return None

        elapsed = time.perf_counter() - self.start_time
        steps_left = self.total_steps - completed_steps

        if steps_left <= 0:
            return 0.0

        avg_step_time = elapsed / completed_steps
        eta_seconds = steps_left * avg_step_time
        return eta_seconds
