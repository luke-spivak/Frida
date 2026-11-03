"""Standalone SDK entry point: Runpod ships this tiny module to the Pod."""


def execute(request):
    from remote.worker import run_job
    return run_job(request)
