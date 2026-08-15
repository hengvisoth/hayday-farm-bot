import unittest

from app import WorkerController


class FakeBot:
    def __init__(self):
        self.reset_count = 0
        self.stop_requested = False

    def reset_stop(self):
        self.reset_count += 1
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def bot_loop(self):
        return None


class FakeThread:
    created = []

    def __init__(self, target, daemon):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.alive = False
        self.__class__.created.append(self)

    def start(self):
        self.started = True
        self.alive = True

    def is_alive(self):
        return self.alive


class WorkerControllerTests(unittest.TestCase):
    def setUp(self):
        FakeThread.created.clear()

    def test_start_refuses_second_live_worker(self):
        bot = FakeBot()
        controller = WorkerController(bot, thread_factory=FakeThread)

        first_started = controller.start()
        second_started = controller.start()

        self.assertTrue(first_started)
        self.assertFalse(second_started)
        self.assertEqual(len(FakeThread.created), 1)
        self.assertEqual(bot.reset_count, 1)

    def test_start_allows_new_worker_after_previous_worker_stops(self):
        bot = FakeBot()
        controller = WorkerController(bot, thread_factory=FakeThread)
        controller.start()
        FakeThread.created[0].alive = False

        self.assertTrue(controller.start())
        self.assertEqual(len(FakeThread.created), 2)

    def test_stop_signals_bot_without_discarding_live_thread(self):
        bot = FakeBot()
        controller = WorkerController(bot, thread_factory=FakeThread)
        controller.start()

        controller.stop()

        self.assertTrue(bot.stop_requested)
        self.assertTrue(controller.is_running())


if __name__ == "__main__":
    unittest.main()
