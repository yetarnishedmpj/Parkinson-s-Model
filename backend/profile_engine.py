#!/usr/bin/env python3
"""
Profiling script for the Digital Twin engine.
Run this to get baseline performance metrics.
"""
import time
import cProfile
import pstats
import io
from app.services.engine import engine

def simulate_ticks(n_ticks=1000):
    """Simulate n_ticks and return elapsed time."""
    start = time.perf_counter()
    for _ in range(n_ticks):
        engine.generate_reading()
    end = time.perf_counter()
    return end - start

def main():
    print("Profiling Digital Twin engine performance...")
    # Warm-up
    for _ in range(100):
        engine.generate_reading()

    # Reset engine state for consistent baseline? Not trivial; we'll just run as-is.
    # For better baseline, we could reinitialize engine, but let's keep simple.

    # Run simulation
    n_ticks = 5000  # Simulate 250 seconds at 20 Hz
    print(f"Simulating {n_ticks} ticks...")
    elapsed = simulate_ticks(n_ticks)
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Average time per tick: {elapsed/n_ticks*1000:.3f} ms")
    print(f"Ticks per second: {n_ticks/elapsed:.1f}")

    # Profiling
    print("\n--- cProfile output (top 20 functions) ---")
    pr = cProfile.Profile()
    pr.enable()
    simulate_ticks(1000)  # Profile 1000 ticks
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

if __name__ == "__main__":
    main()