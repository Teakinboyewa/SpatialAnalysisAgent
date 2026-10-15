"""Run generated code on the QGIS main (GUI) thread.

QGIS layers, renderers, the project and the map canvas may only be created
and changed on the main thread. The plugin's AI work runs in worker threads
(QThread), so code that touches QGIS is handed to the main thread here and
the worker waits until it has finished. Running it in the worker instead
makes QGIS crash intermittently ("Windows fatal exception: access violation").
"""
from qgis.PyQt.QtCore import QCoreApplication, QObject, QThread, Qt, pyqtSignal, pyqtSlot


class _MainThreadExecutor(QObject):
    _run_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        # Blocking: the emitting (worker) thread waits until the slot has run
        # on the thread this object lives in (the main thread).
        self._run_requested.connect(self._run, Qt.BlockingQueuedConnection)

    @pyqtSlot(object)
    def _run(self, job):
        try:
            exec(job['code'], job['namespace'])
        except BaseException as e:  # re-raised in the worker thread
            job['error'] = e

    def run(self, code, namespace):
        job = {'code': code, 'namespace': namespace, 'error': None}
        self._run_requested.emit(job)
        if job['error'] is not None:
            raise job['error']


_executor = None


def exec_in_main_thread(code, namespace):
    """exec(code, namespace) on the main thread; exceptions propagate to the caller.

    Called from the main thread (or without a Qt application) it simply runs
    the code. Called from a worker thread it blocks until the main thread has
    run it, so the main thread must not be waiting on that worker meanwhile.
    """
    global _executor
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() == app.thread():
        exec(code, namespace)
        return
    if _executor is None:
        executor = _MainThreadExecutor()
        executor.moveToThread(app.thread())
        _executor = executor
    _executor.run(code, namespace)
