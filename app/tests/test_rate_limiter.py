import unittest
from datetime import datetime, timedelta
import time
from app.api.sync_service import RateLimiter

class TestRateLimiter(unittest.TestCase):
    def test_rate_limiter_allows_requests_up_to_max(self):
        # Create a rate limiter with max 3 requests per hour
        limiter = RateLimiter(max_requests_per_hour=3)

        # Should allow first 3 requests
        self.assertTrue(limiter.can_make_request())
        limiter.record_request()

        self.assertTrue(limiter.can_make_request())
        limiter.record_request()

        self.assertTrue(limiter.can_make_request())
        limiter.record_request()

        # Should deny 4th request
        self.assertFalse(limiter.can_make_request())

    def test_rate_limiter_expires_old_requests(self):
        # Create a rate limiter with max 2 requests per hour
        limiter = RateLimiter(max_requests_per_hour=2)

        # Mock old timestamps (2 hours ago)
        old_time = datetime.now() - timedelta(hours=2)
        limiter.request_timestamps = [old_time, old_time]

        # Should allow requests since old ones expired
        self.assertTrue(limiter.can_make_request())

    def test_get_next_available_time(self):
        # Create a rate limiter with max 1 request per hour
        limiter = RateLimiter(max_requests_per_hour=1)

        # Record a request 30 minutes ago
        now = datetime.now()
        thirty_min_ago = now - timedelta(minutes=30)
        limiter.request_timestamps = [thirty_min_ago]

        # Next available time should be 30 minutes from now
        next_time = limiter.get_next_available_time()
        expected_time = thirty_min_ago + timedelta(hours=1)

        # Allow a small difference due to test execution time
        time_diff = abs((next_time - expected_time).total_seconds())
        self.assertTrue(time_diff < 5, f"Time difference too large: {time_diff} seconds")

if __name__ == '__main__':
    unittest.main()
