import re
import subprocess
from typing import List
from citadel_shared.logging_setup import get_logger
from proto.python import registry_pb2
from datetime import datetime, timezone
from google.protobuf.timestamp_pb2 import Timestamp


def ssh_bin() -> str:
    return "ssh"


def nvidia_bin() -> str:
    return "nvidia-smi"


def wrap_remote(host: str, cmd: list[str]) -> list[str]:
    return cmd if host == "localhost" else [ssh_bin(), host] + cmd


class GPUCollector:

    def __init__(self, worker_id: str = "unknown"):
        self.worker_id = worker_id
        self.logger = get_logger(f"gpu_collector.{worker_id}")

    def query_nvidia_smi(self, host: str) -> tuple[list[str], list[str]]:
        base = nvidia_bin()
        metrics = [base, "--query-gpu=timestamp,uuid,name,index,utilization.gpu,utilization.memory", "--format=csv,noheader"]
        metrics = wrap_remote(host, metrics)
        try:
            m = subprocess.check_output(metrics, text=True, stderr=subprocess.STDOUT).strip().split("\n")
            return m
        except subprocess.CalledProcessError as e:
            self.logger.error(f"query error on {host}: {e.output}")
            return [], []

    def parse_metrics(self, metric_lines: List[str], host: str, worker_id) -> List[registry_pb2.GpuMetricsData]:
        msgs: List[registry_pb2.GpuMetricsData] = []

        for mline_raw in metric_lines:
            mline = (mline_raw or "").strip()
            if not mline:
                continue

            try:
                parts = [p.strip() for p in mline.split(",")]
                if len(parts) < 6:
                    self.logger.error(f"parse error on {host}: not enough fields -> {mline}")
                    continue

                ts_s, uuid, name, idx_s, ug_s, um_s = parts[:6]

                m_ug = re.search(r"\d+", ug_s)
                m_um = re.search(r"\d+", um_s)
                if not m_ug or not m_um:
                    self.logger.error(f"parse error on {host}: bad util fields -> {mline}")
                    continue
                ug = int(m_ug.group())
                um = int(m_um.group())

                try:
                    dt = datetime.strptime(ts_s, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                except ValueError:
                    dt = datetime.strptime(ts_s, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)

                ts_pb = Timestamp()
                dt_local = datetime.now(timezone.utc)
                ts_pb.FromDatetime(dt_local)

                msg = registry_pb2.GpuMetricsData(
                    timestamp=ts_pb,
                    uuid=uuid,
                    gpu_index=int(idx_s),
                    worker_id=worker_id,
                    gpu_name=name,
                    util_gpu=ug,
                    util_mem=um,
                )
                
                msgs.append(msg)

            except Exception as e:
                self.logger.error(f"parse error on {host}: {mline}  - {e}")

        return msgs


