import multiprocessing as mp
import time
import traceback
from ..log import sayd, sayw


__all__ = ['Worker', 'WorkerManager']


class WorkerManager:
    def __init__(self):
        self._manager = mp.Manager()
        self._worker = self.spawn_worker()

    def __del__(self):
        self.disconnect()

    def connect(self):
        self.start_worker()
        while not self._worker.isalive:
            time.sleep(0.01)

    def disconnect(self):
        self.kill_worker()

    #################

    def spawn_worker(self):
        return NotImplemented

    def start_worker(self):
        if self._worker is None:
            self._worker = self.spawn_worker()

        if not self._worker.isalive:
            self._worker.start()
        return 0

    def kill_worker(self):
        if self._worker is None:
            return 0
        if self._worker.isalive:
            try:
                self._worker.job_Q.put(None)
            except FileNotFoundError:
                sayd('During killing worker, file not found error.')
            except EOFError:
                sayd('During killing worker, EOFError.')
            except BrokenPipeError:
                sayd('During killing worker, BrokenPipeError.')
            self._worker.join()
            self._worker = None

        return 0

    def reboot_worker(self):
        self.kill_worker()
        self.start_worker()

    def assign_job_worker(self, job):
        self._worker.job_Q.put(job)

    def receive_result_worker(self):
        return self._worker.result_Q.get()


class Worker(mp.Process):
    def __init__(self, manager):
        super().__init__()

        self.job_Q = manager.Queue()
        self.result_Q = manager.Queue()
        self.status = manager.dict()

        self.status['isalive'] = False

    @property
    def isalive(self):
        try:
            return self.status['isalive']
        except:
            return False

    @isalive.setter
    def isalive(self, v):
        self.status['isalive'] = v

    ###########################################

    def On_run(self):
        pass

    def On_job(self, job):
        raise NotImplementedError()

    def run(self):
        sayd('Worker start!')

        try:
            self.On_run()

            while True:
                job = self.job_Q.get()
                if job is None:
                    sayd('Worker stopped.')
                    return

                if self.On_job(job):
                    break

        except KeyboardInterrupt:
            sayd('Keyboard Interrupt detected.')
        except (SystemExit, EOFError, BrokenPipeError) as e:
            sayw('Worker killed by %s.' % type(e).__name__)
        except Exception as e:
            sayw('Worker killed with unexpected reason : %s.' % type(e).__name__)
            sayw(traceback.format_exc())
