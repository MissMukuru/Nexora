import numpy as np
import polars as pl
from .config import *


def initialize(w):
    w.backlog = np.zeros(6)
    w.pressure = np.ones(6)
    w.oprows = []


def daily(w, day, regional_orders):
    r = w.r("operations", day)
    ext = w.ext
    regions = np.arange(6)
    base_staff = np.bincount(w.er[w.ed == 2], minlength=6)
    # Workload units normalized to staff productive capacity, not one order per employee.
    orders = np.asarray(regional_orders)
    workload = orders * 0.30 + r.gamma(5, 5, 6)
    dow = int((date(day).astype(int) + 3) % 7)
    shift = np.where(dow >= 5, 0.60, 1.0)
    scheduled = np.maximum(4, (base_staff * shift).astype(int))
    absentee = np.clip(r.beta(2, 35, 6) + 0.0005 * ext["rainfall"][day], 0, 0.45)
    incident = r.poisson(0.5 + ext["rainfall"][day] / 130, 6)
    downtime = r.gamma(1.2, 4, 6) + incident * r.gamma(2, 3, 6)
    event = 438 <= day <= 469
    if event:
        absentee[0] += 0.19
        downtime[0] += r.gamma(5, 15)
        w.ep.benchmark("EVT-001", "daily_operations", 1)
    factor = w.ep.factor("operations", day, regions, table="daily_operations")
    absentee = np.clip(absentee * factor, 0, 0.65)
    downtime = np.clip(downtime * factor, 0, 460)
    staffing = np.maximum(1, np.rint(scheduled * (1 - absentee))).astype(int)
    capacity = staffing * 5.6 * np.maximum(0.15, 1 - downtime / 480) / np.sqrt(factor)
    overtime = np.minimum(
        staffing * 2.5, np.maximum(0, workload + w.backlog - capacity) / 3
    )
    capacity += overtime * 2.3
    completed = np.minimum(workload + w.backlog, capacity)
    before = w.backlog.copy()
    w.backlog = np.maximum(0, w.backlog + workload - completed)
    processing = (
        12
        + r.gamma(3, 2, 6)
        + 8 * np.log1p(w.backlog / np.maximum(capacity, 1))
        + downtime * 0.13
        + ext["traffic"][day] * 4
    )
    if event:
        processing[0] *= 1.35
    sla = np.clip(
        sigmoid(-4 + 0.06 * processing + 1.4 * w.backlog / np.maximum(capacity, 1)),
        0,
        1,
    )
    cost = money(
        staffing * 2900 * (1 + 0.006 * (ext["inflation"][day] - 5))
        + overtime * 650
        + workload * ext["fuel"][day] * 0.16
    )
    w.pressure = np.clip(
        1 + processing / 30 + sla + np.log1p(w.backlog / np.maximum(capacity, 1)), 1, 12
    )
    frame = pl.DataFrame(
        dict(
            date=np.repeat(date(day), 6),
            region=REGIONS,
            workload_units=workload,
            completed_units=completed,
            backlog_units=w.backlog,
            opening_backlog_units=before,
            processing_time_minutes=processing,
            downtime_minutes=downtime,
            incident_count=incident,
            staffing_level=staffing,
            scheduled_staffing=scheduled,
            absenteeism_rate=absentee,
            overtime_hours=overtime,
            SLA_breach_rate=sla,
            operating_cost=cost,
        )
    )
    assert np.allclose(before + workload - completed, w.backlog)
    return frame
