import asyncio
from psutil import cpu_percent, virtual_memory
from datetime import datetime
import time
from statistics import mean
from os import path
from os.path import dirname


def writeToFile(file="", content=""):
    if file != "":
        with open(file, 'a+') as f:
            f.write(str(content) + '\n')
    else:
        print(content)


class SystemMonitoring:
    def __init__(self) -> None:
        self.main_path = None
        self.execution_stats: list = []
        self._cpu_usage_data: list = []
        self._ram_usage_data: list = []
        self._sampling_task: asyncio.Task
        self.max_cpu_used: float = 0
        self.avg_cpu_used: float = 0
        self.max_ram_used: float = 0
        self.avg_ram_used: float = 0
        self.duration: float = 0
        self.start_time: float = 0
        self.finish_time: float = 0
        self.stats: dict = {}
        self._sampling: bool = False
        self.baseline_cpu: float = 0
        self.baseline_ram: float = 0

    def writeReport(self, datasetName, modelName, encoderName):
        # content = {
        #     'avg_cpu': self.avg_cpu_used,
        #     'max_cpu': self.max_cpu_used,
        #     'avg_ram': self.max_cpu_used,
        #     'runtime': self.duration,

        # }
        content = self.stats
        writeToFile('results/' + datasetName + '/' + encoderName +
                    '/system_metrics/' + modelName + 'AutoGluon.txt', content)

    async def _get_current_state(self, interval: float = 0.1):
        self._sampling = True
        while self._sampling:
            try:
                self._cpu_usage_data.append(cpu_percent())
                self._ram_usage_data.append(virtual_memory().percent)
                await asyncio.sleep(interval)
            except Exception as exc:
                print(f"Measuring cpu and ram failed due to {exc}")
                raise exc

    async def _calculate_stats(self):
        self.max_cpu_used = max(self._cpu_usage_data)
        self.avg_cpu_used = mean(self._cpu_usage_data)
        self.max_ram_used = max(self._ram_usage_data)
        self.avg_ram_used = mean(self._ram_usage_data)

    async def _get_baselinestate(self):
        self.baseline_cpu = cpu_percent(0.5)
        self.baseline_ram = virtual_memory().percent

    async def start(self, interval: float = 0.1) -> bool:
        try:
            # Check is sampling is not happening
            if not self._sampling:
                await self._get_baselinestate()
                self.start_time = time.time()
                self._sampling_task = asyncio.create_task(
                    self._get_current_state(interval)
                )

            return True
        except Exception as exc:
            print(f"SystemMonitoring failed to start due to {exc}")
            raise exc

    async def stop(self) -> dict:
        try:
            # Check is sampling is happening
            if self._sampling:
                # Change _sampling to false and await _sampling_task
                self._sampling = False
                await self._sampling_task

                # Get finish time, duration and rest of stats
                self.finish_time = time.time()
                self.duration = self.finish_time - self.start_time
                await self._calculate_stats()

                self.stats.update(
                    {
                        "duration": round(float(self.duration), 2),
                        "max_cpu": round(float(self.max_cpu_used), 2),
                        "avg_cpu": round(float(self.avg_cpu_used), 2),
                        "max_ram": round(float(self.max_ram_used), 2),
                        "avg_ram": round(float(self.avg_ram_used), 2),
                        "baseline_cpu": round(float(self.baseline_cpu), 2),
                        "baseline_ram": round(float(self.baseline_ram), 2),
                    }
                )

                return self.stats
            else:
                return {}
        except Exception as exc:
            print(f"SystemMonitoring failed to stop due to {exc}")
            raise exc
